"""Turn a labeled call log into a hitting set instance.

The log is pair-labeled, which is what production actually gives you cheaply:

SAME
    The backend reported a conflict, returned the id of an already existing
    resource, or an operator filed a duplicate incident. Two calls, one effect.
DISTINCT
    Two calls that both landed and both were intended. Two calls, two effects.

From those labels alone the effect-determining field set is recoverable without
anyone hand-annotating a schema:

* a DISTINCT pair must differ somewhere inside the key, or it collapses into one
  key and the second call is wrongly suppressed;
* a SAME pair must not differ anywhere inside the key, or it splits into two
  keys and the side effect fires twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .normalize import Normalizer

MISSING = "\x00<missing>"


@dataclass(frozen=True)
class CallRecord:
    """One observed tool call."""

    call_id: str
    tool: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class PairLabel:
    """Observed relationship between two calls to the same tool."""

    a: str
    b: str
    same_effect: bool


@dataclass
class ConstraintSet:
    """A hitting set instance for one tool, plus the evidence behind it."""

    tool: str
    fields: Tuple[str, ...]
    forbidden: Tuple[str, ...]
    distinct_diffs: Tuple[Tuple[str, ...], ...]
    instability: Dict[str, float]
    same_pairs: int
    distinct_pairs: int
    conflicts: Tuple[Tuple[str, str], ...] = field(default=())

    @property
    def allowed(self) -> Tuple[str, ...]:
        forbidden = set(self.forbidden)
        return tuple(f for f in self.fields if f not in forbidden)


def difference_fields(
    normalizer: Normalizer,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Set[str]:
    """Fields whose canonical values differ between two calls.

    A field present in one call and absent in the other counts as a difference.
    Agents add and drop optional fields freely when they rewrite a call, and
    whether that matters is exactly what the SAME labels decide.
    """
    flat_left = normalizer.normalize_call(left)
    flat_right = normalizer.normalize_call(right)
    diff: Set[str] = set()
    for path in set(flat_left) | set(flat_right):
        if flat_left.get(path, MISSING) != flat_right.get(path, MISSING):
            diff.add(path)
    return diff


def build_constraints(
    calls: Sequence[CallRecord],
    labels: Sequence[PairLabel],
    normalizer: Optional[Normalizer] = None,
    forbid_rate: float = 0.0,
) -> Dict[str, ConstraintSet]:
    """Build one :class:`ConstraintSet` per tool.

    Args:
        calls: the observed call log.
        labels: pair labels over call ids. Pairs spanning two tools are skipped.
        normalizer: value canonicalizer. Sniffed from the log when omitted.
        forbid_rate: a field is forbidden from the key when it differs in more
            than this fraction of the SAME pairs that contain it. ``0.0`` means
            one SAME pair is enough, which is correct for clean labels. Raise it
            to tolerate mislabeled pairs.
    """
    if not 0.0 <= forbid_rate < 1.0:
        raise ValueError("forbid_rate must be in [0.0, 1.0)")

    by_id = {call.call_id: call for call in calls}
    missing = {lbl.a for lbl in labels if lbl.a not in by_id}
    missing |= {lbl.b for lbl in labels if lbl.b not in by_id}
    if missing:
        raise KeyError(f"labels reference unknown call ids: {sorted(missing)[:5]}")

    if normalizer is None:
        normalizer = Normalizer.sniffed([c.args for c in calls])

    per_tool: Dict[str, List[CallRecord]] = {}
    for call in calls:
        per_tool.setdefault(call.tool, []).append(call)

    same_diff_counts: Dict[str, Dict[str, int]] = {t: {} for t in per_tool}
    same_support: Dict[str, int] = {t: 0 for t in per_tool}
    distinct_diffs: Dict[str, List[Tuple[str, ...]]] = {t: [] for t in per_tool}
    distinct_pairs: Dict[str, int] = {t: 0 for t in per_tool}
    conflicts_src: Dict[str, List[Tuple[str, str]]] = {t: [] for t in per_tool}

    for label in labels:
        left, right = by_id[label.a], by_id[label.b]
        if left.tool != right.tool:
            continue
        tool = left.tool
        diff = difference_fields(normalizer, left.args, right.args)
        if label.same_effect:
            same_support[tool] += 1
            counts = same_diff_counts[tool]
            for path in diff:
                counts[path] = counts.get(path, 0) + 1
        else:
            distinct_pairs[tool] += 1
            distinct_diffs[tool].append((tuple(sorted(diff)), label.a, label.b))

    out: Dict[str, ConstraintSet] = {}
    for tool, tool_calls in per_tool.items():
        fields = sorted(
            set().union(*(normalizer.normalize_call(c.args).keys() for c in tool_calls))
            if tool_calls
            else set()
        )
        support = max(same_support[tool], 1)
        instability = {
            path: same_diff_counts[tool].get(path, 0) / support for path in fields
        }
        forbidden = tuple(
            path for path in fields if instability.get(path, 0.0) > forbid_rate
        )
        allowed = set(fields) - set(forbidden)

        diffs: List[Tuple[str, ...]] = []
        conflicts: List[Tuple[str, str]] = []
        for diff, a_id, b_id in distinct_diffs[tool]:
            if set(diff) & allowed:
                diffs.append(diff)
            else:
                conflicts.append((a_id, b_id))

        out[tool] = ConstraintSet(
            tool=tool,
            fields=tuple(fields),
            forbidden=forbidden,
            distinct_diffs=tuple(diffs),
            instability=instability,
            same_pairs=same_support[tool],
            distinct_pairs=distinct_pairs[tool],
            conflicts=tuple(conflicts),
        )
    return out
