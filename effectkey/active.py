"""Active pair selection for effect-key inference.

Randomly sampled pair labels underdetermine the key, and badly. Two calls drawn
at random differ in a dozen fields at once, so the constraint "differ somewhere
in here" is nearly free to satisfy: a single high-cardinality field such as
``amount`` separates almost every random pair, and the minimum hitting set
happily returns it alone. The key then collapses distinct calls together and
suppresses real side effects.

The informative pair is the near miss: two calls that differ in exactly one
field. Labeling that pair pins that field down.

* oracle says DISTINCT, so one field is the whole difference, so the key must
  contain it. The field is forced.
* oracle says SAME, so the field cannot be in the key. The field is excluded.

Either answer resolves a field completely. So instead of labeling random pairs,
ask for the pairs whose answer is decisive, and spend the labeling budget where
it moves the key.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .constraints import CallRecord, PairLabel, difference_fields
from .normalize import Normalizer

FORCED = "forced"
EXCLUDED = "excluded"
UNDETERMINED = "undetermined"

Oracle = Callable[[CallRecord, CallRecord], bool]


@dataclass
class SelectionTrace:
    """What the selector spent and what it learned."""

    labels: List[PairLabel] = field(default_factory=list)
    queries: int = 0
    singleton_queries: int = 0
    resolved: Dict[str, str] = field(default_factory=dict)
    contradictions: List[str] = field(default_factory=list)

    def status_counts(self) -> Dict[str, int]:
        counts = {FORCED: 0, EXCLUDED: 0, UNDETERMINED: 0}
        for status in self.resolved.values():
            counts[status] += 1
        return counts


def candidate_pairs(
    calls: Sequence[CallRecord],
    window: int = 6,
    random_pairs: int = 1500,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """Build a pool of pairs worth considering.

    Calls that are close together in the log are the ones most likely to be a
    retry or a near miss, which is where the decisive pairs live. A random
    sample is mixed in so the pool is not purely local.
    """
    rng = random.Random(seed)
    pool: Set[Tuple[int, int]] = set()
    count = len(calls)
    for i in range(count):
        for j in range(i + 1, min(i + window + 1, count)):
            pool.add((i, j))
    for _ in range(random_pairs):
        if count < 2:
            break
        i, j = rng.sample(range(count), 2)
        pool.add((min(i, j), max(i, j)))
    return sorted(pool)


def select_labels(
    calls: Sequence[CallRecord],
    oracle: Oracle,
    normalizer: Normalizer,
    budget: int = 120,
    window: int = 6,
    random_pairs: int = 1500,
    seed: int = 0,
) -> SelectionTrace:
    """Spend a labeling budget on the most decisive pairs for one tool.

    Args:
        calls: the recorded calls for a single tool.
        oracle: answers whether two calls produced the same effect. In
            production this is a backend conflict check, a reconciliation
            query, or an operator. Here it is the cost being minimized.
        budget: maximum oracle queries.

    The loop is greedy on how close a pair is to being decisive. Only the
    excluded fields may be subtracted from a pair's difference: a DISTINCT pair
    is already satisfied by any field in the key, so a pair that still differs
    in four candidate fields teaches nothing about which one matters. Forcing
    requires the candidate set to be a true singleton.

    Exclusions cascade. Ruling out a noise field shrinks every pair that
    mentioned it, which turns previously useless pairs into decisive ones.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")

    trace = SelectionTrace()
    if len(calls) < 2:
        return trace

    flat = [normalizer.normalize_call(c.args) for c in calls]
    all_fields: Set[str] = set()
    for row in flat:
        all_fields.update(row)
    trace.resolved = {f: UNDETERMINED for f in sorted(all_fields)}

    pool = candidate_pairs(calls, window=window, random_pairs=random_pairs, seed=seed)
    diffs: Dict[Tuple[int, int], Set[str]] = {
        pair: difference_fields(normalizer, calls[pair[0]].args, calls[pair[1]].args)
        for pair in pool
    }

    forced: Set[str] = set()
    excluded: Set[str] = set()
    remaining = [p for p in pool if diffs[p]]

    def informative(pair: Tuple[int, int]) -> Optional[Tuple[Set[str], Set[str]]]:
        """Candidate separating fields and the still-unresolved ones among them."""
        candidates = diffs[pair] - excluded
        if not candidates:
            return None
        unresolved = candidates - forced
        if not unresolved:
            return None
        return candidates, unresolved

    while trace.queries < budget and remaining:
        best_pair = None
        best_score = None
        for pair in remaining:
            probe = informative(pair)
            if probe is None:
                continue
            candidates, unresolved = probe
            score = (len(candidates), -len(unresolved), pair)
            if best_score is None or score < best_score:
                best_pair, best_score = pair, score

        if best_pair is None:
            break

        candidates = diffs[best_pair] - excluded
        left, right = calls[best_pair[0]], calls[best_pair[1]]
        same = oracle(left, right)
        trace.queries += 1
        if len(candidates) == 1:
            trace.singleton_queries += 1
        trace.labels.append(PairLabel(left.call_id, right.call_id, same))

        if same:
            clash = diffs[best_pair] & forced
            if clash:
                # A field already proven necessary cannot also be irrelevant.
                # The cause is a value-level equivalence normalization missed,
                # so surface it instead of letting the key quietly change.
                trace.contradictions.extend(sorted(clash))
            excluded |= diffs[best_pair]
            forced -= excluded
        elif len(candidates) == 1:
            forced |= candidates

        remaining = [
            p for p in remaining if p != best_pair and informative(p) is not None
        ]

    for name in trace.resolved:
        if name in excluded:
            trace.resolved[name] = EXCLUDED
        elif name in forced:
            trace.resolved[name] = FORCED
    return trace


def select_labels_per_tool(
    calls: Sequence[CallRecord],
    oracle: Oracle,
    normalizer: Normalizer,
    budget_per_tool: int = 120,
    window: int = 6,
    random_pairs: int = 1500,
    seed: int = 0,
) -> Tuple[List[PairLabel], Dict[str, SelectionTrace]]:
    """Run :func:`select_labels` independently for every tool in the log."""
    by_tool: Dict[str, List[CallRecord]] = {}
    for call in calls:
        by_tool.setdefault(call.tool, []).append(call)

    labels: List[PairLabel] = []
    traces: Dict[str, SelectionTrace] = {}
    for offset, (tool, tool_calls) in enumerate(sorted(by_tool.items())):
        trace = select_labels(
            tool_calls,
            oracle,
            normalizer,
            budget=budget_per_tool,
            window=window,
            random_pairs=random_pairs,
            seed=seed + offset,
        )
        traces[tool] = trace
        labels.extend(trace.labels)
    return labels, traces


def random_labels_per_tool(
    calls: Sequence[CallRecord],
    oracle: Oracle,
    budget_per_tool: int = 120,
    seed: int = 0,
) -> List[PairLabel]:
    """Uniformly random pair labels, the obvious thing to do instead."""
    rng = random.Random(seed)
    by_tool: Dict[str, List[CallRecord]] = {}
    for call in calls:
        by_tool.setdefault(call.tool, []).append(call)

    labels: List[PairLabel] = []
    for tool, tool_calls in sorted(by_tool.items()):
        if len(tool_calls) < 2:
            continue
        for _ in range(budget_per_tool):
            left, right = rng.sample(tool_calls, 2)
            labels.append(
                PairLabel(left.call_id, right.call_id, oracle(left, right))
            )
    return labels
