"""Trace replay harness and effect-level metrics.

A strategy is scored on what the backend actually did, not on whether its keys
matched. Two numbers matter and they pull against each other:

duplicate landings
    The side effect fired more than once for one intent. One invoice, two
    charges.
lost intents
    The side effect never fired, because dedup suppressed the only attempt that
    would have worked. One invoice, zero charges.

Any strategy can drive one of these to zero by sacrificing the other, so both
are always reported together.

One case is deliberately not counted against a strategy. If two separate
intents carry byte-identical effect values and the second arrives inside the
dedup window, suppressing it is what the window is for, and no key-based scheme
could do otherwise. Those are tallied as correctly suppressed rather than lost,
because charging a strategy for them would measure the limits of idempotency
keys in general rather than the quality of this key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .ledger import BLOCK, EffectLedger, UncertainEffect
from .workload import AMBIGUOUS_FAIL, CLEAN_FAIL, SUCCESS, Workload

MECH_NONE = "none"
MECH_SEEN_SET = "seen_set"
MECH_LEDGER = "ledger"


class KeyStrategy(Protocol):
    name: str

    def key(self, tool: str, args: Any) -> str:  # pragma: no cover - protocol
        ...


@dataclass
class EvalResult:
    """Outcome of replaying one trace under one strategy and mechanism."""

    strategy: str
    mechanism: str
    attempts: int
    intents: int
    landings: int
    duplicate_landings: int
    lost_intents: int
    deduped: int
    blocked_uncertain: int
    correctly_suppressed: int = 0
    key_fields: Dict[str, List[str]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def duplicates_per_1k(self) -> float:
        return 1000.0 * self.duplicate_landings / self.attempts if self.attempts else 0.0

    @property
    def lost_pct(self) -> float:
        return 100.0 * self.lost_intents / self.intents if self.intents else 0.0

    @property
    def duplicate_pct(self) -> float:
        return (
            100.0 * self.duplicate_landings / self.intents if self.intents else 0.0
        )

    @property
    def effect_error_pct(self) -> float:
        """Intents whose backend state is wrong, in either direction."""
        if not self.intents:
            return 0.0
        return 100.0 * (self.duplicate_landings + self.lost_intents) / self.intents

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "mechanism": self.mechanism,
            "attempts": self.attempts,
            "intents": self.intents,
            "landings": self.landings,
            "duplicate_landings": self.duplicate_landings,
            "lost_intents": self.lost_intents,
            "deduped": self.deduped,
            "blocked_uncertain": self.blocked_uncertain,
            "correctly_suppressed": self.correctly_suppressed,
            "duplicates_per_1k": round(self.duplicates_per_1k, 4),
            "duplicate_pct": round(self.duplicate_pct, 4),
            "lost_pct": round(self.lost_pct, 4),
            "effect_error_pct": round(self.effect_error_pct, 4),
            "key_fields": self.key_fields,
            **self.extra,
        }


def evaluate(
    workload: Workload,
    strategy: Optional[KeyStrategy] = None,
    mechanism: str = MECH_LEDGER,
    ttl_seconds: float = 900.0,
    reconcile_rate: float = 1.0,
    key_fields: Optional[Dict[str, List[str]]] = None,
) -> EvalResult:
    """Replay a trace and count what the backend ended up doing.

    Args:
        strategy: the keying strategy. ``None`` means no dedup at all.
        mechanism: what guards execution once a key exists.
            ``MECH_LEDGER`` is reserve, commit, abort, TTL and an uncertain
            state. ``MECH_SEEN_SET`` is the common shortcut: record the key
            before doing the work and never expire it.
        reconcile_rate: fraction of ambiguous outcomes a reconciliation job
            resolves against the backend. Held at 1.0 for strategy comparisons
            so ambiguity handling is identical everywhere and only the key
            varies.
    """
    if mechanism not in (MECH_NONE, MECH_SEEN_SET, MECH_LEDGER):
        raise ValueError(f"unknown mechanism: {mechanism!r}")
    if strategy is None and mechanism != MECH_NONE:
        raise ValueError("a dedup mechanism requires a key strategy")

    attempts = sorted(workload.attempts, key=lambda a: (a.timestamp, a.attempt_id))
    now = {"t": attempts[0].timestamp if attempts else 0.0}
    ledger = EffectLedger(
        ttl_seconds=ttl_seconds,
        clock=lambda: now["t"],
        on_uncertain=BLOCK,
        join_timeout=0.0,
    )
    seen: Dict[str, Any] = {}
    landings: Dict[str, int] = {i: 0 for i in workload.effect_of_intent}
    landed_at: List[tuple] = []
    deduped = 0
    blocked = 0
    rng_state = 0

    for attempt in attempts:
        now["t"] = attempt.timestamp

        if mechanism == MECH_NONE:
            _apply(attempt, landings, landed_at)
            continue

        key = strategy.key(attempt.tool, attempt.args)

        if mechanism == MECH_SEEN_SET:
            if key in seen:
                deduped += 1
                continue
            seen[key] = True
            _apply(attempt, landings, landed_at)
            continue

        try:
            slot = ledger.reserve(key)
        except UncertainEffect:
            blocked += 1
            continue
        except TimeoutError:
            blocked += 1
            continue

        if slot.duplicate:
            deduped += 1
            continue

        landed = _apply(attempt, landings, landed_at)
        if attempt.outcome == SUCCESS:
            ledger.commit(key, result=attempt.attempt_id)
        elif attempt.outcome == CLEAN_FAIL:
            ledger.abort(key)
        else:
            ledger.mark_uncertain(key)
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            if rng_state / 0x7FFFFFFF < reconcile_rate:
                ledger.resolve_uncertain(key, landed=landed, result=attempt.attempt_id)

    duplicate_landings = sum(max(0, count - 1) for count in landings.values())
    lost, suppressed = _classify_zero_landings(
        workload, landings, landed_at, ttl_seconds
    )

    return EvalResult(
        strategy=strategy.name if strategy else "no_dedup",
        mechanism=mechanism,
        attempts=len(attempts),
        intents=len(landings),
        landings=sum(landings.values()),
        duplicate_landings=duplicate_landings,
        lost_intents=lost,
        deduped=deduped,
        blocked_uncertain=blocked,
        correctly_suppressed=suppressed,
        key_fields=key_fields or {},
        extra={"ledger_stats": ledger.stats.as_dict()} if mechanism == MECH_LEDGER else {},
    )


def _apply(attempt, landings: Dict[str, int], landed_at=None) -> bool:
    """Run one attempt against the backend and record whether it landed."""
    if attempt.landed_if_executed:
        landings[attempt.intent_id] += 1
        if landed_at is not None:
            landed_at.append((attempt.timestamp, attempt.intent_id))
        return True
    return False


def _classify_zero_landings(workload, landings, landed_at, ttl_seconds):
    """Split zero-landing intents into genuinely lost and correctly suppressed.

    An intent that never landed is only a loss if no earlier landing carried the
    same effect signature inside the dedup window. Otherwise the suppression was
    the window doing its job.
    """
    signature = workload.effect_of_intent
    first_attempt: Dict[str, float] = {}
    for attempt in workload.attempts:
        prior = first_attempt.get(attempt.intent_id)
        if prior is None or attempt.timestamp < prior:
            first_attempt[attempt.intent_id] = attempt.timestamp

    by_signature: Dict[str, List[float]] = {}
    for timestamp, intent_id in landed_at:
        by_signature.setdefault(signature[intent_id], []).append(timestamp)
    for stamps in by_signature.values():
        stamps.sort()

    lost = 0
    suppressed = 0
    for intent_id, count in landings.items():
        if count:
            continue
        start = first_attempt.get(intent_id, 0.0)
        stamps = by_signature.get(signature[intent_id], ())
        if any(start - ttl_seconds <= t <= start for t in stamps):
            suppressed += 1
        else:
            lost += 1
    return lost, suppressed
