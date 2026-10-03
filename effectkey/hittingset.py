"""Minimum hitting set over field-difference sets.

Effect-key inference reduces to a hitting set problem. Each pair of calls that
is known to have produced *different* side effects contributes a constraint:
the key must contain at least one field on which that pair differs, otherwise
the two calls collide and the second one is wrongly suppressed. Each pair known
to have produced the *same* effect forbids every field on which it differs.

Minimality is not cosmetic. Every field in the key is a field whose
reformatting on retry can split one intent into two keys and fire the side
effect twice. So the smallest feasible key is the one that tolerates the most
paraphrase, which is why this module solves the problem exactly instead of
settling for the greedy answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULT_NODE_BUDGET = 200_000


@dataclass(frozen=True)
class HittingSetResult:
    """Outcome of a hitting set solve."""

    elements: Tuple[str, ...]
    optimal: bool
    lower_bound: int
    nodes: int
    feasible: bool = True
    unhittable: Tuple[Tuple[str, ...], ...] = field(default=())

    @property
    def size(self) -> int:
        return len(self.elements)


def _bit_iter(mask: int) -> Iterable[int]:
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _encode(difference_sets, allowed):
    """Map field names to bit positions and drop disallowed fields.

    Returns the per-constraint bitmasks, the index-to-name table, the reverse
    map, and the constraints left with no allowed field (infeasible evidence).
    """
    allowed_set = None if allowed is None else set(allowed)

    names: List[str] = []
    index: Dict[str, int] = {}
    for diff in difference_sets:
        for name in diff:
            if allowed_set is not None and name not in allowed_set:
                continue
            if name not in index:
                index[name] = len(names)
                names.append(name)

    masks: List[int] = []
    unhittable: List[Tuple[str, ...]] = []
    for diff in difference_sets:
        mask = 0
        for name in diff:
            bit = index.get(name)
            if bit is not None:
                mask |= 1 << bit
        if mask == 0:
            unhittable.append(tuple(sorted(diff)))
        else:
            masks.append(mask)
    return masks, names, index, unhittable


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


def _reduce(masks: Sequence[int]) -> Tuple[List[int], int]:
    """Drop duplicate and dominated constraints.

    If constraint A is a subset of constraint B then hitting A always hits B,
    so B carries no information and is removed.
    """
    unique = sorted(set(masks), key=lambda m: (_popcount(m), m))
    kept: List[int] = []
    for mask in unique:
        if any(sub & mask == sub for sub in kept):
            continue
        kept.append(mask)
    forced = 0
    for mask in kept:
        if _popcount(mask) == 1:
            forced |= mask
    return kept, forced


def _lower_bound(masks: Sequence[int]) -> int:
    """Greedy packing of pairwise disjoint constraints.

    Disjoint constraints share no field, so each one needs its own element and
    their count is a valid lower bound on the hitting set size.
    """
    used = 0
    count = 0
    for mask in sorted(masks, key=_popcount):
        if mask & used:
            continue
        used |= mask
        count += 1
    return count


def greedy_hitting_set(
    difference_sets: Sequence[Iterable[str]],
    allowed: Optional[Iterable[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> HittingSetResult:
    """Classic max-coverage greedy. Fast, and the baseline to beat."""
    masks, names, _, unhittable = _encode(difference_sets, allowed)
    reduced, forced = _reduce(masks)
    weights = weights or {}

    chosen: Set[int] = set(_bit_iter(forced))
    remaining = [m for m in reduced if not (m & forced)]

    while remaining:
        best_bit = None
        best_score = None
        for bit in range(len(names)):
            probe = 1 << bit
            covered = sum(1 for mask in remaining if mask & probe)
            if covered == 0:
                continue
            score = (-covered, weights.get(names[bit], 0.0), names[bit])
            if best_score is None or score < best_score:
                best_bit, best_score = bit, score
        if best_bit is None:
            break
        chosen.add(best_bit)
        probe = 1 << best_bit
        remaining = [mask for mask in remaining if not (mask & probe)]

    return HittingSetResult(
        elements=tuple(sorted(names[bit] for bit in chosen)),
        optimal=False,
        lower_bound=_lower_bound(reduced),
        nodes=0,
        feasible=not unhittable,
        unhittable=tuple(unhittable),
    )


def min_hitting_set(
    difference_sets: Sequence[Iterable[str]],
    allowed: Optional[Iterable[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> HittingSetResult:
    """Exact minimum hitting set by branch and bound.

    Ties on size are broken by total field weight, so an unstable field (one the
    agent reformats often) loses to a stable field of equal coverage.

    If ``node_budget`` is exhausted the best solution found so far is returned
    with ``optimal=False``. It is still a valid hitting set, just not proven
    minimum.
    """
    masks, names, index, unhittable = _encode(difference_sets, allowed)
    weights = weights or {}
    weight_of = [weights.get(name, 0.0) for name in names]

    reduced, forced = _reduce(masks)
    global_lb = _lower_bound(reduced)

    if not reduced:
        return HittingSetResult(
            elements=(),
            optimal=True,
            lower_bound=0,
            nodes=0,
            feasible=not unhittable,
            unhittable=tuple(unhittable),
        )

    seed = greedy_hitting_set(difference_sets, allowed, weights)
    best_bits: Set[int] = {index[name] for name in seed.elements if name in index}
    best_size = len(best_bits)
    best_weight = sum(weight_of[bit] for bit in best_bits)

    forced_bits = sorted(_bit_iter(forced))
    start_remaining = tuple(m for m in reduced if not (m & forced))
    state = {"nodes": 0, "exhausted": False}

    def search(remaining, chosen, chosen_weight):
        nonlocal best_bits, best_size, best_weight

        if state["nodes"] >= node_budget:
            state["exhausted"] = True
            return
        state["nodes"] += 1

        if not remaining:
            if (len(chosen), chosen_weight) < (best_size, best_weight):
                best_bits = set(chosen)
                best_size = len(chosen)
                best_weight = chosen_weight
            return

        bound = len(chosen) + _lower_bound(remaining)
        if bound > best_size:
            return
        if bound == best_size and chosen_weight >= best_weight:
            return

        pivot = min(remaining, key=_popcount)
        order = sorted(
            _bit_iter(pivot),
            key=lambda bit: (
                -sum(1 for mask in remaining if mask & (1 << bit)),
                weight_of[bit],
                names[bit],
            ),
        )
        for bit in order:
            probe = 1 << bit
            chosen.append(bit)
            search(
                tuple(mask for mask in remaining if not (mask & probe)),
                chosen,
                chosen_weight + weight_of[bit],
            )
            chosen.pop()

    search(start_remaining, list(forced_bits), sum(weight_of[b] for b in forced_bits))

    return HittingSetResult(
        elements=tuple(sorted(names[bit] for bit in best_bits)),
        optimal=not state["exhausted"],
        lower_bound=global_lb,
        nodes=state["nodes"],
        feasible=not unhittable,
        unhittable=tuple(unhittable),
    )
