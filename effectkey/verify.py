"""Audit an inferred key before trusting it, and repair it if it fails.

Inference is only as good as the pairs it was shown. A field that the SAME
evidence excluded early can never be re-examined, because every pair that would
have argued for it looks uninformative once the field is ruled out. The failure
is silent: the key looks clean, reports no conflicts, and quietly merges calls
that the backend treats as different.

So the key gets audited in both directions against the same oracle:

collision
    The key merges two calls the oracle calls DISTINCT. The key is too small
    and will suppress a real side effect.
false split
    The key separates two calls the oracle calls SAME. The key is too large and
    will fire a side effect twice.

Collisions carry their own repair: the field that should have been in the key is
somewhere in the difference between the two calls, including among the fields
that were excluded. Re-admitting the fewest such fields is another minimum
hitting set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .constraints import CallRecord, difference_fields
from .hittingset import DEFAULT_NODE_BUDGET, min_hitting_set
from .infer import EffectKeySpec

Oracle = Callable[[CallRecord, CallRecord], bool]


@dataclass
class VerificationReport:
    """Audit result for one tool's key."""

    tool: str
    merge_probes: int = 0
    collisions: int = 0
    collision_diffs: List[Tuple[str, ...]] = field(default_factory=list)
    irreducible_collisions: int = 0
    split_probes: int = 0
    false_splits: int = 0
    false_split_fields: List[str] = field(default_factory=list)
    queries: int = 0

    @property
    def clean(self) -> bool:
        return self.collisions == 0 and self.false_splits == 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "tool": self.tool,
            "queries": self.queries,
            "merge_probes": self.merge_probes,
            "collisions": self.collisions,
            "irreducible_collisions": self.irreducible_collisions,
            "split_probes": self.split_probes,
            "false_splits": self.false_splits,
            "false_split_fields": sorted(set(self.false_split_fields)),
            "clean": self.clean,
        }


def verify_key(
    spec: EffectKeySpec,
    calls: Sequence[CallRecord],
    oracle: Oracle,
    merge_probes: int = 40,
    split_probes: int = 40,
    seed: int = 0,
) -> VerificationReport:
    """Probe an inferred key for collisions and false splits.

    Only calls for ``spec.tool`` are used. Merge probes are drawn from inside
    key groups, split probes from pairs the key separated, so every query tests
    a decision the key actually made.
    """
    rng = random.Random(seed)
    report = VerificationReport(tool=spec.tool)
    pool = [c for c in calls if c.tool == spec.tool]
    if len(pool) < 2:
        return report

    groups: Dict[str, List[CallRecord]] = {}
    for call in pool:
        groups.setdefault(spec.key(call.args), []).append(call)

    merged = [g for g in groups.values() if len(g) > 1]
    rng.shuffle(merged)
    for group in merged:
        if report.merge_probes >= merge_probes:
            break
        left, right = rng.sample(group, 2)
        report.merge_probes += 1
        report.queries += 1
        if oracle(left, right):
            continue
        report.collisions += 1
        diff = difference_fields(spec.normalizer, left.args, right.args)
        if diff:
            report.collision_diffs.append(tuple(sorted(diff)))
        else:
            # Identical after normalization yet different effects. No field set
            # can separate these, so adding fields would not help.
            report.irreducible_collisions += 1

    keys = list(groups)
    if len(keys) > 1:
        for _ in range(split_probes):
            left_key, right_key = rng.sample(keys, 2)
            left = rng.choice(groups[left_key])
            right = rng.choice(groups[right_key])
            report.split_probes += 1
            report.queries += 1
            if not oracle(left, right):
                continue
            report.false_splits += 1
            diff = difference_fields(spec.normalizer, left.args, right.args)
            report.false_split_fields.extend(sorted(diff & set(spec.key_fields)))
    return report


def repair_key(
    spec: EffectKeySpec,
    report: VerificationReport,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> EffectKeySpec:
    """Re-admit the fewest fields needed to resolve observed collisions.

    Previously excluded fields are back in scope here. An exclusion came from a
    SAME pair that differed on the field, which means the field is noisy, but a
    collision is direct proof it also carries effect identity. Re-admitting it
    trades a possible duplicate for a side effect that would otherwise be lost.
    """
    if not report.collision_diffs:
        return spec

    patch = min_hitting_set(
        report.collision_diffs,
        allowed=None,
        weights=spec.instability,
        node_budget=node_budget,
    )
    repaired = tuple(sorted(set(spec.key_fields) | set(patch.elements)))
    return replace(
        spec,
        key_fields=repaired,
        optimal=spec.optimal and patch.optimal,
    )


def verify_and_repair(
    specs: Dict[str, EffectKeySpec],
    calls: Sequence[CallRecord],
    oracle: Oracle,
    merge_probes: int = 40,
    split_probes: int = 40,
    seed: int = 0,
    rounds: int = 2,
) -> Tuple[Dict[str, EffectKeySpec], Dict[str, List[VerificationReport]], int]:
    """Audit and repair every tool's key, iterating until the audit is clean.

    Returns the repaired specs, the audit trail per tool, and the total number
    of oracle queries the audit cost.
    """
    if rounds < 1:
        raise ValueError("rounds must be at least 1")

    out = dict(specs)
    trail: Dict[str, List[VerificationReport]] = {t: [] for t in specs}
    queries = 0
    for tool, spec in sorted(specs.items()):
        current = spec
        for round_index in range(rounds):
            report = verify_key(
                current,
                calls,
                oracle,
                merge_probes=merge_probes,
                split_probes=split_probes,
                seed=seed + round_index,
            )
            trail[tool].append(report)
            queries += report.queries
            if not report.collision_diffs:
                break
            current = repair_key(current, report)
        out[tool] = current
    return out, trail, queries
