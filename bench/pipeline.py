"""Benchmark pipeline: infer, audit, and score every keying strategy.

Oracle queries are the cost being compared, so the random-label baseline is
given the same total query budget that active selection plus verification
spends. Comparing a tuned method against a starved baseline would not say
anything.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effectkey.active import random_labels_per_tool, select_labels_per_tool
from effectkey.baselines import (
    CanonicalHashKey,
    InferredKey,
    NormalizedAllFieldsKey,
    RawHashKey,
    SimilarityKey,
)
from effectkey.constraints import CallRecord
from effectkey.infer import SOLVER_EXACT, SOLVER_GREEDY, infer_effect_keys
from effectkey.metrics import MECH_LEDGER, MECH_NONE, MECH_SEEN_SET, evaluate
from effectkey.normalize import Normalizer
from effectkey.verify import verify_and_repair
from effectkey.workload import WorkloadGenerator

TTL_SECONDS = 900.0
TRAIN_SEED = 1
TEST_SEED = 2
N_INTENTS = 900
ACTIVE_BUDGET = 120
MERGE_PROBES = 40
SPLIT_PROBES = 40


@dataclass
class Bundle:
    """Everything one benchmark run needs, built once and reused."""

    train: Any
    test: Any
    calls: List[CallRecord]
    normalizer: Normalizer
    oracle: Callable[[CallRecord, CallRecord], bool]
    truth: Dict[str, set]


def build_bundle(n_intents: int = N_INTENTS) -> Bundle:
    train = WorkloadGenerator(seed=TRAIN_SEED).generate(n_intents)
    test = WorkloadGenerator(seed=TEST_SEED).generate(n_intents)
    calls = [
        CallRecord(call_id=a.attempt_id, tool=a.tool, args=a.args)
        for a in train.attempts
    ]
    intent_of = {a.attempt_id: a.intent_id for a in train.attempts}
    signature = train.effect_of_intent

    def oracle(left: CallRecord, right: CallRecord) -> bool:
        """Did these two calls produce the same backend effect?

        Stands in for a backend conflict check, a reconciliation query, or an
        operator. Every call to it is counted and reported.
        """
        return signature[intent_of[left.call_id]] == signature[intent_of[right.call_id]]

    return Bundle(
        train=train,
        test=test,
        calls=calls,
        normalizer=Normalizer.sniffed([c.args for c in calls]),
        oracle=oracle,
        truth={t.name: set(t.effect_fields) for t in train.tools},
    )


def field_set_accuracy(specs, truth) -> Tuple[int, float]:
    """Exact field-set matches, and mean Jaccard against ground truth."""
    exact = sum(1 for tool in truth if set(specs[tool].key_fields) == truth[tool])
    total = 0.0
    for tool, want in truth.items():
        got = set(specs[tool].key_fields)
        union = got | want
        total += len(got & want) / len(union) if union else 1.0
    return exact, total / len(truth)


def tune_similarity(bundle: Bundle) -> Tuple[float, Dict[float, float]]:
    """Pick the similarity threshold that minimizes effect error on train.

    The threshold baseline gets a fair shot: swept on the training trace, then
    frozen for the test trace, same as every other strategy.
    """
    sweep: Dict[float, float] = {}
    for step in range(5, 20):
        threshold = step / 20.0
        strategy = SimilarityKey(bundle.normalizer, threshold=threshold)
        result = evaluate(bundle.train, strategy, MECH_LEDGER, ttl_seconds=TTL_SECONDS)
        sweep[threshold] = result.effect_error_pct
    best = min(sweep, key=lambda t: (sweep[t], -t))
    return best, sweep


def run_inference(bundle: Bundle) -> Dict[str, Any]:
    """Infer keys three ways and audit the active one."""
    active_labels, traces = select_labels_per_tool(
        bundle.calls,
        bundle.oracle,
        bundle.normalizer,
        budget_per_tool=ACTIVE_BUDGET,
        seed=7,
    )
    active_queries = sum(t.queries for t in traces.values())

    specs_active = infer_effect_keys(
        bundle.calls, active_labels, normalizer=bundle.normalizer, solver=SOLVER_EXACT
    )
    specs_greedy = infer_effect_keys(
        bundle.calls, active_labels, normalizer=bundle.normalizer, solver=SOLVER_GREEDY
    )
    specs_verified, trail, verify_queries = verify_and_repair(
        specs_active,
        bundle.calls,
        bundle.oracle,
        merge_probes=MERGE_PROBES,
        split_probes=SPLIT_PROBES,
        seed=3,
    )

    total_queries = active_queries + verify_queries
    # Give random labeling the same oracle budget, split evenly across tools.
    matched_budget = max(1, total_queries // max(1, len(bundle.truth)))
    random_labels = random_labels_per_tool(
        bundle.calls, bundle.oracle, budget_per_tool=matched_budget, seed=7
    )
    specs_random = infer_effect_keys(
        bundle.calls, random_labels, normalizer=bundle.normalizer, solver=SOLVER_EXACT
    )

    return {
        "specs_active": specs_active,
        "specs_greedy": specs_greedy,
        "specs_verified": specs_verified,
        "specs_random": specs_random,
        "active_queries": active_queries,
        "verify_queries": verify_queries,
        "total_queries": total_queries,
        "random_queries": len(random_labels),
        "traces": traces,
        "audit_trail": trail,
    }


def learning_curve(
    bundle: Bundle, budgets=(1, 2, 3, 4, 6, 9, 14, 30, 60, 120)
) -> Dict[str, list]:
    """Effect error on the test trace as a function of oracle queries spent.

    Active selection saturates: it stops querying once no pair can still resolve
    a field, so its curve is reported at the queries it actually used rather
    than at the budget it was offered.
    """
    curve: Dict[str, list] = {"active": [], "random": []}
    for budget in budgets:
        labels, traces = select_labels_per_tool(
            bundle.calls,
            bundle.oracle,
            bundle.normalizer,
            budget_per_tool=budget,
            seed=7,
        )
        used = sum(t.queries for t in traces.values())
        specs = infer_effect_keys(
            bundle.calls, labels, normalizer=bundle.normalizer, solver=SOLVER_EXACT
        )
        result = evaluate(
            bundle.test,
            InferredKey(specs, "active"),
            MECH_LEDGER,
            ttl_seconds=TTL_SECONDS,
        )
        if not curve["active"] or curve["active"][-1][0] != used:
            curve["active"].append((used, result.effect_error_pct))

        per_tool = max(1, budget)
        random_labels = random_labels_per_tool(
            bundle.calls, bundle.oracle, budget_per_tool=per_tool, seed=7
        )
        specs_r = infer_effect_keys(
            bundle.calls, random_labels, normalizer=bundle.normalizer, solver=SOLVER_EXACT
        )
        result_r = evaluate(
            bundle.test,
            InferredKey(specs_r, "random"),
            MECH_LEDGER,
            ttl_seconds=TTL_SECONDS,
        )
        curve["random"].append((len(random_labels), result_r.effect_error_pct))
    return curve


def score_all(bundle: Bundle, inferred: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate every strategy on the held-out test trace."""
    threshold, sweep = tune_similarity(bundle)
    norm = bundle.normalizer

    plans = [
        ("no dedup", None, MECH_NONE),
        ("raw argument hash", RawHashKey(), MECH_LEDGER),
        ("canonical JSON hash", CanonicalHashKey(), MECH_LEDGER),
        ("all fields normalized", NormalizedAllFieldsKey(norm), MECH_LEDGER),
        (
            f"similarity threshold {threshold:.2f}",
            SimilarityKey(norm, threshold=threshold),
            MECH_LEDGER,
        ),
        (
            "inferred, random labels",
            InferredKey(inferred["specs_random"], "inferred_random_labels"),
            MECH_LEDGER,
        ),
        (
            "inferred, active labels",
            InferredKey(inferred["specs_active"], "inferred_active_labels"),
            MECH_LEDGER,
        ),
        (
            "effectkey, audited",
            InferredKey(inferred["specs_verified"], "effectkey_audited"),
            MECH_LEDGER,
        ),
        (
            "effectkey, naive seen set",
            InferredKey(inferred["specs_verified"], "effectkey_seen_set"),
            MECH_SEEN_SET,
        ),
    ]

    rows: List[Dict[str, Any]] = []
    for label, strategy, mechanism in plans:
        result = evaluate(
            bundle.test, strategy, mechanism, ttl_seconds=TTL_SECONDS
        )
        row = result.as_dict()
        row["label"] = label
        rows.append(row)

    rows.append({"label": "_similarity_sweep", "sweep": sweep, "chosen": threshold})
    return rows
