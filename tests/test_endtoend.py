"""End to end on a small trace: inference, auditing, and replay scoring."""

from __future__ import annotations

from effectkey.active import random_labels_per_tool, select_labels_per_tool
from effectkey.baselines import CanonicalHashKey, InferredKey, RawHashKey
from effectkey.constraints import CallRecord
from effectkey.infer import infer_effect_keys
from effectkey.metrics import MECH_LEDGER, MECH_NONE, MECH_SEEN_SET, evaluate
from effectkey.normalize import Normalizer
from effectkey.verify import verify_and_repair
from effectkey.workload import WorkloadGenerator


def build(n_intents=320):
    train = WorkloadGenerator(seed=11).generate(n_intents)
    test = WorkloadGenerator(seed=12).generate(n_intents)
    calls = [
        CallRecord(a.attempt_id, a.tool, a.args) for a in train.attempts
    ]
    intent_of = {a.attempt_id: a.intent_id for a in train.attempts}
    signature = train.effect_of_intent

    def oracle(left, right):
        return signature[intent_of[left.call_id]] == signature[intent_of[right.call_id]]

    normalizer = Normalizer.sniffed([c.args for c in calls])
    return train, test, calls, oracle, normalizer


def test_generated_trace_is_internally_consistent():
    train, _, calls, _, _ = build()
    assert len(calls) == len(train.attempts)
    assert len(train.intents()) == len(train.effect_of_intent)
    # Retries carry the same intent and later timestamps than their original.
    for attempts in train.intents().values():
        stamps = [a.timestamp for a in attempts]
        assert stamps == sorted(stamps)
        assert sum(1 for a in attempts if not a.is_retry) == 1


def test_no_dedup_produces_duplicate_landings():
    _, test, _, _, _ = build()
    result = evaluate(test, None, MECH_NONE)
    assert result.duplicate_landings > 0
    assert result.lost_intents == 0


def test_raw_and_canonical_hashes_are_defeated_by_rewrites():
    _, test, _, _, _ = build()
    baseline = evaluate(test, None, MECH_NONE).duplicate_landings
    for strategy in (RawHashKey(), CanonicalHashKey()):
        result = evaluate(test, strategy, MECH_LEDGER)
        assert result.duplicate_landings >= baseline * 0.9


def test_audited_key_beats_every_baseline_on_effect_error():
    _, test, calls, oracle, normalizer = build()
    labels, _ = select_labels_per_tool(
        calls, oracle, normalizer, budget_per_tool=120, seed=7
    )
    specs = infer_effect_keys(calls, labels, normalizer=normalizer)
    audited, _, _ = verify_and_repair(specs, calls, oracle, seed=3)

    best = evaluate(test, InferredKey(audited, "effectkey"), MECH_LEDGER)
    none = evaluate(test, None, MECH_NONE)
    raw = evaluate(test, RawHashKey(), MECH_LEDGER)

    assert best.effect_error_pct < none.effect_error_pct / 4
    assert best.effect_error_pct < raw.effect_error_pct / 4


def test_active_selection_beats_random_at_a_larger_budget():
    _, test, calls, oracle, normalizer = build()
    active, traces = select_labels_per_tool(
        calls, oracle, normalizer, budget_per_tool=120, seed=7
    )
    used = sum(t.queries for t in traces.values())
    random_labels = random_labels_per_tool(
        calls, oracle, budget_per_tool=used, seed=7
    )
    normalizer_ = normalizer
    active_specs = infer_effect_keys(calls, active, normalizer=normalizer_)
    random_specs = infer_effect_keys(calls, random_labels, normalizer=normalizer_)

    active_error = evaluate(
        test, InferredKey(active_specs, "a"), MECH_LEDGER
    ).effect_error_pct
    random_error = evaluate(
        test, InferredKey(random_specs, "r"), MECH_LEDGER
    ).effect_error_pct
    assert active_error < random_error


def test_naive_seen_set_loses_side_effects_the_ledger_keeps():
    _, test, calls, oracle, normalizer = build()
    labels, _ = select_labels_per_tool(
        calls, oracle, normalizer, budget_per_tool=120, seed=7
    )
    specs = infer_effect_keys(calls, labels, normalizer=normalizer)
    audited, _, _ = verify_and_repair(specs, calls, oracle, seed=3)
    strategy = InferredKey(audited, "effectkey")

    ledger = evaluate(test, strategy, MECH_LEDGER)
    seen_set = evaluate(test, strategy, MECH_SEEN_SET)
    assert seen_set.lost_intents > ledger.lost_intents * 5
