"""Constraint building, key inference, active selection, and auditing."""

from __future__ import annotations

import pytest

from effectkey.active import select_labels
from effectkey.constraints import CallRecord, PairLabel, build_constraints
from effectkey.infer import ConflictError, infer_effect_keys
from effectkey.normalize import Normalizer
from effectkey.verify import repair_key, verify_key


def call(call_id, **args):
    return CallRecord(call_id=call_id, tool="t", args=args)


BASE = dict(email="a@b.example", number="INV-1", note="hello")


def test_same_pair_forbids_every_field_it_differs_on():
    calls = [call("1", **BASE), call("2", **{**BASE, "note": "hi there"})]
    instance = build_constraints(calls, [PairLabel("1", "2", True)])["t"]
    assert "note" in instance.forbidden
    assert "email" not in instance.forbidden


def test_distinct_pair_differing_in_one_field_forces_it():
    calls = [call("1", **BASE), call("2", **{**BASE, "number": "INV-2"})]
    specs = infer_effect_keys(calls, [PairLabel("1", "2", False)])
    assert specs["t"].key_fields == ("number",)


def test_key_is_stable_across_formatting_rewrites():
    specs = infer_effect_keys(
        [call("1", **BASE), call("2", **{**BASE, "number": "INV-2"})],
        [PairLabel("1", "2", False)],
    )
    spec = specs["t"]
    left = {"number": "INV-1", "email": "a@b.example", "note": "hello"}
    right = {"note": "HELLO  ", "email": "<A@B.EXAMPLE>", "number": " INV-1 "}
    assert spec.key(left) == spec.key(right)


def test_contradictory_labels_are_reported_not_absorbed():
    calls = [call("1", **BASE), call("2", **{**BASE, "number": "INV-2"})]
    labels = [PairLabel("1", "2", True), PairLabel("1", "2", False)]
    specs = infer_effect_keys(calls, labels)
    assert specs["t"].conflicts
    with pytest.raises(ConflictError):
        infer_effect_keys(calls, labels, strict=True)


def test_labels_referencing_unknown_calls_raise():
    with pytest.raises(KeyError):
        build_constraints([call("1", **BASE)], [PairLabel("1", "missing", True)])


def test_tool_without_distinct_evidence_is_flagged_degenerate():
    calls = [call("1", **BASE), call("2", **{**BASE, "note": "other"})]
    spec = infer_effect_keys(calls, [PairLabel("1", "2", True)])["t"]
    assert spec.degenerate


def test_active_selection_prefers_decisive_pairs():
    """A pair differing in one field resolves that field in a single query."""
    calls = [
        call("1", **BASE),
        call("2", **{**BASE, "note": "reworded"}),
        call("3", **{**BASE, "number": "INV-2"}),
    ]
    truth = {"number", "email"}

    def oracle(left, right):
        project = lambda c: tuple(sorted((k, v) for k, v in c.args.items() if k in truth))
        return project(left) == project(right)

    normalizer = Normalizer.sniffed([c.args for c in calls])
    trace = select_labels(calls, oracle, normalizer, budget=10, window=4,
                          random_pairs=20, seed=0)
    assert trace.queries <= 10
    assert trace.resolved["note"] == "excluded"
    assert trace.resolved["number"] == "forced"


def test_audit_finds_a_collision_and_repair_fixes_it():
    calls = [
        call("1", number="INV-1", region="eu"),
        call("2", number="INV-1", region="us"),
    ]
    # A key that ignores region merges two calls the oracle calls distinct.
    spec = infer_effect_keys(calls, [PairLabel("1", "2", True)])["t"]
    spec.key_fields = ("number",)

    def oracle(left, right):
        return left.args["region"] == right.args["region"]

    report = verify_key(spec, calls, oracle, merge_probes=5, split_probes=5, seed=0)
    assert report.collisions == 1
    assert not report.clean
    repaired = repair_key(spec, report)
    assert "region" in repaired.key_fields
    assert repaired.key(calls[0].args) != repaired.key(calls[1].args)


def test_explain_shows_which_fields_the_key_ignored():
    spec = infer_effect_keys(
        [call("1", **BASE), call("2", **{**BASE, "number": "INV-2"})],
        [PairLabel("1", "2", False)],
    )["t"]
    detail = spec.explain(BASE)
    assert detail["key_fields"] == ["number"]
    assert "note" in detail["ignored_fields"]
