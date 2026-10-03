"""Infer a per-tool effect key from a labeled call log."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .constraints import CallRecord, ConstraintSet, PairLabel, build_constraints
from .hittingset import DEFAULT_NODE_BUDGET, greedy_hitting_set, min_hitting_set
from .normalize import Normalizer

SOLVER_EXACT = "exact"
SOLVER_GREEDY = "greedy"


class ConflictError(ValueError):
    """Raised in strict mode when the labels cannot all be satisfied."""


@dataclass
class EffectKeySpec:
    """The inferred effect key for one tool, and how well it is supported."""

    tool: str
    key_fields: Tuple[str, ...]
    normalizer: Normalizer
    solver: str
    optimal: bool
    lower_bound: int
    nodes: int
    forbidden: Tuple[str, ...]
    conflicts: Tuple[Tuple[str, str], ...]
    instability: Dict[str, float]
    same_pairs: int
    distinct_pairs: int

    @property
    def degenerate(self) -> bool:
        """True when every call to this tool maps to one key.

        Happens when the log contains no DISTINCT pair for the tool. The key is
        then unsupported by evidence and will suppress real second calls, so a
        caller should widen the log before trusting it.
        """
        return not self.key_fields

    def key(self, args: Mapping[str, Any]) -> str:
        """Stable hex key for a call's arguments."""
        flat = self.normalizer.normalize_call(args)
        payload = [
            [path, _jsonable(flat[path])]
            for path in sorted(self.key_fields)
            if path in flat
        ]
        blob = json.dumps(
            {"tool": self.tool, "fields": payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def explain(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        """The projection that produced the key, for audit logs."""
        flat = self.normalizer.normalize_call(args)
        return {
            "tool": self.tool,
            "key": self.key(args),
            "key_fields": list(sorted(self.key_fields)),
            "projection": {
                path: _jsonable(flat[path])
                for path in sorted(self.key_fields)
                if path in flat
            },
            "ignored_fields": sorted(set(flat) - set(self.key_fields)),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def solve_constraints(
    constraints: ConstraintSet,
    normalizer: Normalizer,
    solver: str = SOLVER_EXACT,
    node_budget: int = DEFAULT_NODE_BUDGET,
    use_weights: bool = True,
) -> EffectKeySpec:
    """Solve one tool's instance into an :class:`EffectKeySpec`.

    Field weights are the observed instability of each field: how often it
    differs across SAME pairs. Among keys of equal minimum size the solver
    prefers the fields the agent rewrites least, since those are the fields
    whose formatting is least likely to split one intent into two keys.
    """
    if solver not in (SOLVER_EXACT, SOLVER_GREEDY):
        raise ValueError(f"unknown solver: {solver!r}")

    weights = dict(constraints.instability) if use_weights else None
    solve = min_hitting_set if solver == SOLVER_EXACT else greedy_hitting_set
    kwargs = {"node_budget": node_budget} if solver == SOLVER_EXACT else {}
    result = solve(
        constraints.distinct_diffs,
        allowed=constraints.allowed,
        weights=weights,
        **kwargs,
    )

    return EffectKeySpec(
        tool=constraints.tool,
        key_fields=result.elements,
        normalizer=normalizer,
        solver=solver,
        optimal=result.optimal,
        lower_bound=result.lower_bound,
        nodes=result.nodes,
        forbidden=constraints.forbidden,
        conflicts=constraints.conflicts,
        instability=dict(constraints.instability),
        same_pairs=constraints.same_pairs,
        distinct_pairs=constraints.distinct_pairs,
    )


def infer_effect_keys(
    calls: Sequence[CallRecord],
    labels: Sequence[PairLabel],
    normalizer: Optional[Normalizer] = None,
    solver: str = SOLVER_EXACT,
    forbid_rate: float = 0.0,
    node_budget: int = DEFAULT_NODE_BUDGET,
    use_weights: bool = True,
    strict: bool = False,
) -> Dict[str, EffectKeySpec]:
    """Infer an effect key for every tool present in the log.

    Args:
        strict: raise :class:`ConflictError` if any DISTINCT pair cannot be
            separated using the fields the SAME pairs leave available. Those
            pairs are otherwise dropped and reported on the spec, never
            silently absorbed.
    """
    if normalizer is None:
        normalizer = Normalizer.sniffed([c.args for c in calls])

    constraints = build_constraints(calls, labels, normalizer, forbid_rate=forbid_rate)
    specs: Dict[str, EffectKeySpec] = {}
    for tool, instance in constraints.items():
        spec = solve_constraints(
            instance,
            normalizer,
            solver=solver,
            node_budget=node_budget,
            use_weights=use_weights,
        )
        if strict and spec.conflicts:
            raise ConflictError(
                f"tool {tool!r}: {len(spec.conflicts)} DISTINCT pairs cannot be "
                f"separated once SAME pairs forbid "
                f"{sorted(spec.forbidden)}; first conflict {spec.conflicts[0]}"
            )
        specs[tool] = spec
    return specs
