"""Flatten nested tool-call arguments into dotted field paths.

A tool call argument object is a JSON-like dict. Effect-key inference operates
on a flat namespace of field paths so that a key can name a single leaf such as
``recipient.email`` or ``line_items[*].sku``.

List handling is deliberate. Two conventions are supported:

``ORDERED``
    ``items[0]``, ``items[1]``, ... Position is part of the identity.
``SET``
    ``items[*]`` with a single canonical multiset value. Reordering the list
    does not change the field value.

Agents reorder list arguments constantly when they rewrite a call, so ``SET``
is the default for lists of scalars. Lists of objects keep positional paths
unless the schema marks them as a set.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Set

ORDERED = "ordered"
SET = "set"

_SCALAR = (str, int, float, bool, type(None))


def _is_scalar(value: Any) -> bool:
    return isinstance(value, _SCALAR)


def flatten(
    obj: Mapping[str, Any],
    set_paths: Optional[Set[str]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Return a dict mapping dotted field path to leaf value.

    Args:
        obj: the argument object.
        set_paths: field paths (in ``a.b[*]`` form) whose lists are treated as
            unordered multisets. Lists of scalars are always treated as sets.
        prefix: internal recursion prefix.
    """
    set_paths = set_paths or set()
    out: Dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        _flatten_value(path, value, set_paths, out)
    return out


def _flatten_value(
    path: str,
    value: Any,
    set_paths: Set[str],
    out: Dict[str, Any],
) -> None:
    if isinstance(value, Mapping):
        if not value:
            out[path] = _EMPTY_OBJECT
            return
        for key, sub in value.items():
            _flatten_value(f"{path}.{key}", sub, set_paths, out)
        return

    if isinstance(value, (list, tuple)):
        star = f"{path}[*]"
        all_scalar = all(_is_scalar(item) for item in value)
        if all_scalar or star in set_paths:
            out[star] = _as_multiset(value)
            return
        for index, item in enumerate(value):
            _flatten_value(f"{path}[{index}]", item, set_paths, out)
        return

    out[path] = value


class _EmptyObject:
    """Sentinel leaf for an empty nested object, so the path still exists."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<empty-object>"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _EmptyObject)

    def __hash__(self) -> int:
        return hash("<empty-object>")


_EMPTY_OBJECT = _EmptyObject()


class MultiSet:
    """An order-free list leaf.

    The raw element values are kept so that per-type normalization still runs
    on each element. Canonical ordering is applied after normalization, in
    :mod:`effectkey.normalize`, not here.
    """

    __slots__ = ("values",)

    def __init__(self, values: Iterable[Any]) -> None:
        self.values = tuple(values)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"MultiSet({list(self.values)!r})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MultiSet) and self.values == other.values

    def __hash__(self) -> int:
        return hash(("multiset", self.values))


def _as_multiset(values: Iterable[Any]) -> "MultiSet":
    """Order-free representation of a list of scalars."""
    return MultiSet(values)


def field_paths(objs: Iterable[Mapping[str, Any]], set_paths: Optional[Set[str]] = None) -> Set[str]:
    """Union of field paths present across a collection of argument objects."""
    paths: Set[str] = set()
    for obj in objs:
        paths.update(flatten(obj, set_paths).keys())
    return paths
