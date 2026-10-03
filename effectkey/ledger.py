"""Write-ahead effect ledger.

An effect key is only half the problem. The other half is what you do with it
at the moment of execution, and the naive answer (keep a set of seen keys and
skip anything already in it) is wrong in two directions:

* a call that **failed** must stay retryable, so a key cannot be burned on
  reservation alone;
* a call whose transport died **after** the request reached the backend is
  genuinely ambiguous, and silently retrying it is how one invoice becomes two.

So the ledger is a small state machine with an explicit uncertain state, and a
TTL, because "send the reminder again tomorrow" is a legitimate repeat of
byte-identical arguments and must not be suppressed forever.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type

IN_FLIGHT = "in_flight"
COMMITTED = "committed"
UNCERTAIN = "uncertain"
FRESH = "fresh"

BLOCK = "block"
RETRY = "retry"


class UncertainEffect(RuntimeError):
    """Raised when a key is parked in the uncertain state and policy is BLOCK.

    The prior attempt may or may not have reached the backend. Resolving it
    needs a read against the backend or a human, not another write.
    """

    def __init__(self, key: str) -> None:
        super().__init__(
            f"effect key {key[:12]} is unresolved from a previous attempt; "
            "reconcile with the backend before retrying"
        )
        self.key = key


@dataclass
class _Entry:
    state: str
    created: float
    result: Any = None
    attempts: int = 0


@dataclass
class LedgerStats:
    """Counters for what the ledger actually did."""

    executed: int = 0
    deduped: int = 0
    joined: int = 0
    aborted: int = 0
    blocked_uncertain: int = 0
    expired: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "executed": self.executed,
            "deduped": self.deduped,
            "joined": self.joined,
            "aborted": self.aborted,
            "blocked_uncertain": self.blocked_uncertain,
            "expired": self.expired,
        }


@dataclass
class Slot:
    """A reservation handed to the caller by :meth:`EffectLedger.guard`."""

    key: str
    state: str
    result: Any = None
    _committed: bool = field(default=False, repr=False)

    @property
    def duplicate(self) -> bool:
        """True when the effect already landed and ``result`` is the cached one."""
        return self.state == COMMITTED


class EffectLedger:
    """Key-scoped execution guard with reserve, commit, abort and TTL.

    Thread safe. Concurrent attempts on one key join the first attempt rather
    than running a second side effect.
    """

    def __init__(
        self,
        ttl_seconds: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
        on_uncertain: str = BLOCK,
        join_timeout: float = 30.0,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if on_uncertain not in (BLOCK, RETRY):
            raise ValueError(f"on_uncertain must be {BLOCK!r} or {RETRY!r}")
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.on_uncertain = on_uncertain
        self.join_timeout = join_timeout
        self.stats = LedgerStats()
        self._entries: Dict[str, _Entry] = {}
        self._condition = threading.Condition(threading.RLock())

    def _expired(self, entry: _Entry) -> bool:
        return (
            entry.state == COMMITTED
            and self.clock() - entry.created >= self.ttl_seconds
        )

    def state_of(self, key: str) -> str:
        with self._condition:
            entry = self._entries.get(key)
            if entry is None or self._expired(entry):
                return FRESH
            return entry.state

    def reserve(self, key: str) -> Slot:
        """Claim a key, or report that the effect already landed.

        Blocks while another thread holds the key in flight, up to
        ``join_timeout``.
        """
        deadline = self.clock() + self.join_timeout
        counted_join = False
        with self._condition:
            while True:
                entry = self._entries.get(key)

                if entry is not None and self._expired(entry):
                    self.stats.expired += 1
                    del self._entries[key]
                    entry = None

                if entry is None:
                    self._entries[key] = _Entry(
                        state=IN_FLIGHT, created=self.clock(), attempts=1
                    )
                    return Slot(key=key, state=IN_FLIGHT)

                if entry.state == COMMITTED:
                    self.stats.deduped += 1
                    return Slot(key=key, state=COMMITTED, result=entry.result)

                if entry.state == UNCERTAIN:
                    if self.on_uncertain == BLOCK:
                        self.stats.blocked_uncertain += 1
                        raise UncertainEffect(key)
                    entry.state = IN_FLIGHT
                    entry.created = self.clock()
                    entry.attempts += 1
                    return Slot(key=key, state=IN_FLIGHT)

                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out joining in-flight effect {key[:12]}"
                    )
                if not counted_join:
                    self.stats.joined += 1
                    counted_join = True
                self._condition.wait(timeout=remaining)

    def commit(self, key: str, result: Any = None) -> None:
        """Record that the effect landed, and cache its result for the TTL."""
        with self._condition:
            entry = self._entries.get(key)
            attempts = entry.attempts if entry else 1
            self._entries[key] = _Entry(
                state=COMMITTED, created=self.clock(), result=result, attempts=attempts
            )
            self.stats.executed += 1
            self._condition.notify_all()

    def abort(self, key: str) -> None:
        """Record that the effect definitely did not land. The key frees up."""
        with self._condition:
            self._entries.pop(key, None)
            self.stats.aborted += 1
            self._condition.notify_all()

    def mark_uncertain(self, key: str) -> None:
        """Park a key whose outcome is unknown. No automatic retry under BLOCK."""
        with self._condition:
            entry = self._entries.get(key)
            attempts = entry.attempts if entry else 1
            self._entries[key] = _Entry(
                state=UNCERTAIN, created=self.clock(), attempts=attempts
            )
            self._condition.notify_all()

    def resolve_uncertain(self, key: str, landed: bool, result: Any = None) -> None:
        """Close out an uncertain key after reconciling with the backend."""
        if landed:
            self.commit(key, result)
        else:
            with self._condition:
                self._entries.pop(key, None)
                self._condition.notify_all()

    @contextmanager
    def guard(
        self,
        key: str,
        ambiguous_errors: Tuple[Type[BaseException], ...] = (TimeoutError,),
    ) -> Iterator[Slot]:
        """Execute a side effect at most once per key per TTL window.

        Usage::

            with ledger.guard(spec.key(args)) as slot:
                if not slot.duplicate:
                    slot.result = send_invoice(**args)
                return slot.result

        Exceptions in ``ambiguous_errors`` park the key as uncertain. Any other
        exception aborts the reservation, leaving the call retryable.
        """
        slot = self.reserve(key)
        if slot.duplicate:
            yield slot
            return
        try:
            yield slot
        except ambiguous_errors:
            self.mark_uncertain(key)
            raise
        except BaseException:
            self.abort(key)
            raise
        else:
            self.commit(key, slot.result)
