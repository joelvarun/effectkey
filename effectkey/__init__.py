"""effectkey: inferred idempotency keys for LLM agent tool calls.

An agent retries a tool call and the side effect fires twice, because the agent
rewrote the arguments and the gateway hashed them. The fix is not a better hash.
It is knowing which argument fields the backend's effect actually depends on,
and that field set can be inferred from a pair-labeled call log by solving a
constrained minimum hitting set.

Typical use::

    from effectkey import EffectLedger, infer_effect_keys

    specs = infer_effect_keys(calls, labels)
    ledger = EffectLedger(ttl_seconds=900)

    with ledger.guard(specs["send_invoice"].key(args)) as slot:
        if not slot.duplicate:
            slot.result = send_invoice(**args)
"""

from __future__ import annotations

from .active import select_labels, select_labels_per_tool
from .constraints import CallRecord, ConstraintSet, PairLabel, build_constraints
from .hittingset import greedy_hitting_set, min_hitting_set
from .infer import ConflictError, EffectKeySpec, infer_effect_keys
from .ledger import EffectLedger, LedgerStats, UncertainEffect
from .normalize import Normalizer
from .verify import repair_key, verify_and_repair, verify_key

__version__ = "0.1.0"

__all__ = [
    "CallRecord",
    "ConflictError",
    "ConstraintSet",
    "EffectKeySpec",
    "EffectLedger",
    "LedgerStats",
    "Normalizer",
    "PairLabel",
    "UncertainEffect",
    "build_constraints",
    "greedy_hitting_set",
    "infer_effect_keys",
    "min_hitting_set",
    "repair_key",
    "select_labels",
    "select_labels_per_tool",
    "verify_and_repair",
    "verify_key",
]
