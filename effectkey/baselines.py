"""Keying strategies to compare against inferred effect keys.

Each strategy answers one question: given this tool call, what key names the
side effect it is about to cause? Everything downstream (the ledger, the TTL,
the uncertain state) is held identical across strategies, so the benchmark
measures the key and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .infer import EffectKeySpec
from .normalize import Normalizer


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=False, separators=(",", ":"), default=repr)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RawHashKey:
    """SHA of the raw argument object as the model emitted it.

    Defeated by key reordering alone, which is the single most common rewrite.
    """

    name = "raw_hash"

    def key(self, tool: str, args: Mapping[str, Any]) -> str:
        return _digest([tool, args])


class CanonicalHashKey:
    """SHA of the argument object with keys sorted recursively.

    Fixes key order. Does nothing about value formatting, so ``1200`` and
    ``$1,200.00`` still split.
    """

    name = "canonical_hash"

    def key(self, tool: str, args: Mapping[str, Any]) -> str:
        return _digest([tool, _sorted(args)])


def _sorted(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _sorted(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_sorted(v) for v in value]
    return value


class NormalizedAllFieldsKey:
    """SHA of every field after type-driven canonicalization.

    Immune to formatting noise, but still splits whenever the agent reworders a
    field the backend never looks at, such as an invoice note.
    """

    name = "normalized_all_fields"

    def __init__(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer

    def key(self, tool: str, args: Mapping[str, Any]) -> str:
        flat = self.normalizer.normalize_call(args)
        return _digest([tool, sorted((k, repr(v)) for k, v in flat.items())])


class InferredKey:
    """Effect key from the inferred per-tool field set."""

    def __init__(self, specs: Mapping[str, EffectKeySpec], name: str) -> None:
        self.specs = dict(specs)
        self.name = name

    def key(self, tool: str, args: Mapping[str, Any]) -> str:
        spec = self.specs.get(tool)
        if spec is None:
            return _digest([tool, _sorted(args)])
        return spec.key(args)


@dataclass
class SimilarityKey:
    """Threshold dedup over normalized field shingles.

    The obvious alternative to inferring a field set: call two requests the same
    effect when they look similar enough. It trades one failure for the other,
    because a pair that differs in exactly one effect field and nothing else
    scores very high on similarity.
    """

    normalizer: Normalizer
    threshold: float = 0.85
    window: int = 600
    name: str = "similarity_threshold"
    _recent: List[Tuple[str, str, Set[str]]] = field(default_factory=list, repr=False)

    def _shingles(self, args: Mapping[str, Any]) -> Set[str]:
        flat = self.normalizer.normalize_call(args)
        return {f"{path}={value!r}" for path, value in flat.items()}

    def key(self, tool: str, args: Mapping[str, Any]) -> str:
        shingles = self._shingles(args)
        best_key, best_score = None, 0.0
        for cached_tool, cached_key, cached in self._recent:
            if cached_tool != tool:
                continue
            union = len(shingles | cached)
            if union == 0:
                continue
            score = len(shingles & cached) / union
            if score > best_score:
                best_key, best_score = cached_key, score

        if best_key is not None and best_score >= self.threshold:
            return best_key

        fresh = _digest([tool, sorted(shingles)])
        self._recent.append((tool, fresh, shingles))
        if len(self._recent) > self.window:
            del self._recent[: len(self._recent) - self.window]
        return fresh

    def reset(self) -> None:
        self._recent.clear()
