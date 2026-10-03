"""Ledger state machine: dedup, retryability, ambiguity, TTL, concurrency."""

from __future__ import annotations

import threading

import pytest

from effectkey.ledger import (
    COMMITTED,
    FRESH,
    RETRY,
    UNCERTAIN,
    EffectLedger,
    UncertainEffect,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def run(ledger, key, effects, fail=None):
    with ledger.guard(key) as slot:
        if not slot.duplicate:
            effects.append(key)
            if fail:
                raise fail
            slot.result = f"result-{len(effects)}"
        return slot.result


def test_second_call_with_same_key_does_not_re_execute():
    ledger = EffectLedger(ttl_seconds=10, clock=Clock())
    effects = []
    assert run(ledger, "k", effects) == run(ledger, "k", effects)
    assert len(effects) == 1


def test_key_expires_after_ttl():
    clock = Clock()
    ledger = EffectLedger(ttl_seconds=10, clock=clock)
    effects = []
    run(ledger, "k", effects)
    clock.now = 9.0
    run(ledger, "k", effects)
    assert len(effects) == 1
    clock.now = 10.0
    run(ledger, "k", effects)
    assert len(effects) == 2


def test_clean_failure_leaves_the_call_retryable():
    """Burning the key on reservation would lose the side effect entirely."""
    ledger = EffectLedger(ttl_seconds=10, clock=Clock())
    effects = []
    with pytest.raises(ValueError):
        run(ledger, "k", effects, fail=ValueError("backend refused"))
    assert ledger.state_of("k") == FRESH
    run(ledger, "k", effects)
    assert len(effects) == 2


def test_ambiguous_failure_blocks_instead_of_firing_twice():
    ledger = EffectLedger(ttl_seconds=10, clock=Clock())
    effects = []
    with pytest.raises(TimeoutError):
        run(ledger, "k", effects, fail=TimeoutError("response lost"))
    assert ledger.state_of("k") == UNCERTAIN
    with pytest.raises(UncertainEffect):
        run(ledger, "k", effects)
    assert len(effects) == 1


def test_reconciling_an_uncertain_key_unblocks_it():
    ledger = EffectLedger(ttl_seconds=10, clock=Clock())
    effects = []
    with pytest.raises(TimeoutError):
        run(ledger, "k", effects, fail=TimeoutError("response lost"))
    ledger.resolve_uncertain("k", landed=True, result="reconciled")
    assert ledger.state_of("k") == COMMITTED
    assert run(ledger, "k", effects) == "reconciled"
    assert len(effects) == 1


def test_uncertain_key_not_landed_becomes_retryable():
    ledger = EffectLedger(ttl_seconds=10, clock=Clock())
    effects = []
    with pytest.raises(TimeoutError):
        run(ledger, "k", effects, fail=TimeoutError("response lost"))
    ledger.resolve_uncertain("k", landed=False)
    run(ledger, "k", effects)
    assert len(effects) == 2


def test_retry_policy_reruns_uncertain_keys():
    ledger = EffectLedger(ttl_seconds=10, clock=Clock(), on_uncertain=RETRY)
    effects = []
    with pytest.raises(TimeoutError):
        run(ledger, "k", effects, fail=TimeoutError("response lost"))
    run(ledger, "k", effects)
    assert len(effects) == 2


def test_concurrent_attempts_produce_one_effect():
    ledger = EffectLedger(ttl_seconds=60)
    effects = []
    gate = threading.Event()

    def worker():
        with ledger.guard("k") as slot:
            if not slot.duplicate:
                gate.wait(0.5)
                effects.append(1)
                slot.result = "x"

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join()

    assert len(effects) == 1
    assert ledger.stats.deduped == 7


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        EffectLedger(ttl_seconds=0)
    with pytest.raises(ValueError):
        EffectLedger(on_uncertain="whatever")
