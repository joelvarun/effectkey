"""A billing agent that invoices one customer twice, and the fix.

Run it:

    python examples/demo_agent.py

The tool is a fake backend that records every invoice it accepts. Nothing here
is mocked away: the duplicate in the first run is a real second entry in the
backend's ledger, caused by the agent rewriting arguments it had already sent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effectkey.constraints import CallRecord, PairLabel
from effectkey.infer import infer_effect_keys
from effectkey.ledger import EffectLedger, UncertainEffect

RULE = "-" * 74


class BillingBackend:
    """Accepts invoices. Has no idea two requests meant the same thing."""

    def __init__(self) -> None:
        self.issued = []
        self.fail_next_response = False

    def send_invoice(self, **args):
        self.issued.append(args)
        if self.fail_next_response:
            self.fail_next_response = False
            # The invoice was accepted. The response never made it back.
            raise TimeoutError("gateway closed the connection after accept")
        return {"invoice_id": f"srv-{len(self.issued)}"}


# The same intent, written twice by a model that does not repeat itself.
FIRST_ATTEMPT = {
    "customer": {"email": "dana.shaw@northwind.example", "name": "Dana Shaw"},
    "invoice_number": "INV-10432",
    "amount": 1200,
    "note": "Please find the attached statement.",
    "due_date": "2026-10-03",
}

RETRY_ATTEMPT = {
    "amount": "$1,200.00",
    "note": "Attached is the statement for your records.",
    "invoice_number": " INV-10432 ",
    "due_date": "Oct 3, 2026",
    "customer": {"name": "dana shaw", "email": "<DANA.SHAW@northwind.example>"},
}

# A different invoice to the same customer. This one must go through.
DIFFERENT_INVOICE = dict(FIRST_ATTEMPT, invoice_number="INV-10433", amount=480)


def without_effectkey() -> int:
    print(RULE)
    print("Run 1: the agent retries a call whose response was lost")
    print(RULE)
    backend = BillingBackend()
    backend.fail_next_response = True
    try:
        backend.send_invoice(**FIRST_ATTEMPT)
    except TimeoutError as error:
        print(f"  attempt 1 raised {type(error).__name__}: {error}")
        print("  the agent concludes the call failed and rewrites it")
    result = backend.send_invoice(**RETRY_ATTEMPT)
    print(f"  attempt 2 returned {result}")
    print(f"\n  invoices actually issued: {len(backend.issued)}")
    for index, issued in enumerate(backend.issued, start=1):
        print(f"    {index}. {issued['invoice_number']!r} for {issued['amount']!r}")
    print("  the customer was invoiced twice for one intent\n")
    return len(backend.issued)


def infer_key():
    """Infer the effect key from four labeled pairs.

    Each pair answers one question about one field. That is what the active
    selector asks for in a real deployment; here the pairs are written out so
    the evidence is visible.
    """
    calls = [
        CallRecord("c1", "send_invoice", FIRST_ATTEMPT),
        CallRecord("c2", "send_invoice", dict(FIRST_ATTEMPT, note="Reworded.")),
        CallRecord("c3", "send_invoice", dict(FIRST_ATTEMPT, due_date="2026-11-01")),
        CallRecord("c4", "send_invoice", dict(FIRST_ATTEMPT, invoice_number="INV-9")),
        CallRecord("c5", "send_invoice", dict(FIRST_ATTEMPT, amount=999)),
        CallRecord(
            "c6", "send_invoice",
            {**FIRST_ATTEMPT, "customer": {**FIRST_ATTEMPT["customer"],
                                           "email": "other@northwind.example"}},
        ),
        CallRecord(
            "c7", "send_invoice",
            {**FIRST_ATTEMPT, "customer": {**FIRST_ATTEMPT["customer"],
                                           "name": "D. Shaw"}},
        ),
    ]
    labels = [
        PairLabel("c1", "c2", True),   # rewording the note changed nothing
        PairLabel("c1", "c3", True),   # moving the due date changed nothing
        PairLabel("c1", "c7", True),   # the display name changed nothing
        PairLabel("c1", "c4", False),  # a different invoice number is a new invoice
        PairLabel("c1", "c5", False),  # a different amount is a new invoice
        PairLabel("c1", "c6", False),  # a different recipient is a new invoice
    ]
    spec = infer_effect_keys(calls, labels)["send_invoice"]

    print(RULE)
    print("Inferred effect key for send_invoice")
    print(RULE)
    print(f"  key fields:     {', '.join(spec.key_fields)}")
    print(f"  ruled out:      {', '.join(spec.forbidden)}")
    print(f"  proven minimum: {spec.optimal} "
          f"(lower bound {spec.lower_bound}, {spec.nodes} search nodes)")
    print(f"  labels used:    {spec.same_pairs} same, {spec.distinct_pairs} distinct")
    print()
    print(f"  key(attempt 1) = {spec.key(FIRST_ATTEMPT)[:16]}")
    print(f"  key(attempt 2) = {spec.key(RETRY_ATTEMPT)[:16]}")
    print(f"  identical:       {spec.key(FIRST_ATTEMPT) == spec.key(RETRY_ATTEMPT)}")
    print(f"  key(INV-10433) = {spec.key(DIFFERENT_INVOICE)[:16]}  (separate)")
    print()
    return spec


def with_effectkey(spec) -> int:
    print(RULE)
    print("Run 2: same trace, guarded by the inferred key")
    print(RULE)
    backend = BillingBackend()
    ledger = EffectLedger(ttl_seconds=900.0)

    def guarded(args, label):
        key = spec.key(args)
        try:
            with ledger.guard(key) as slot:
                if slot.duplicate:
                    print(f"  {label}: deduped, returning the stored result "
                          f"{slot.result}")
                    return slot.result
                slot.result = backend.send_invoice(**args)
                print(f"  {label}: executed, returned {slot.result}")
                return slot.result
        except TimeoutError as error:
            print(f"  {label}: raised {type(error).__name__}, key parked as uncertain")
            return None
        except UncertainEffect as error:
            print(f"  {label}: blocked, {error}")
            return None

    backend.fail_next_response = True
    guarded(FIRST_ATTEMPT, "attempt 1")
    guarded(RETRY_ATTEMPT, "attempt 2")

    print("\n  reconciling the uncertain key against the backend")
    ledger.resolve_uncertain(
        spec.key(FIRST_ATTEMPT), landed=True, result={"invoice_id": "srv-1"}
    )
    guarded(RETRY_ATTEMPT, "attempt 3")
    guarded(DIFFERENT_INVOICE, "new invoice INV-10433")

    print(f"\n  invoices actually issued: {len(backend.issued)}")
    for index, issued in enumerate(backend.issued, start=1):
        print(f"    {index}. {issued['invoice_number']!r} for {issued['amount']!r}")
    print(f"  ledger: {ledger.stats.as_dict()}")
    return len(backend.issued)


def main() -> int:
    duplicated = without_effectkey()
    spec = infer_key()
    guarded = with_effectkey(spec)

    print()
    print(RULE)
    print(f"  unguarded:  1 intent  -> {duplicated} invoices issued "
          f"({duplicated - 1} duplicate)")
    print(f"  guarded:    2 intents -> {guarded} invoices issued "
          f"(INV-10432 once, INV-10433 once)")
    print("  the retry collapsed onto the first invoice; the new one still went out")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
