"""Value canonicalization: formatting collapses, meaning survives."""

from __future__ import annotations

import pytest

from effectkey.flatten import flatten
from effectkey.normalize import EMAIL, MONEY, Normalizer, sniff_type


def norm(field_type, *values):
    normalizer = Normalizer({"f": field_type})
    return {normalizer.normalize_value("f", v) for v in values}


def test_money_formatting_variants_collapse():
    assert len(norm(MONEY, "1200", 1200, 1200.0, "1,200.00", "$1,200.00", " 1200 ")) == 1


def test_money_keeps_real_differences():
    assert len(norm(MONEY, "1200.00", "1200.01")) == 2


def test_date_formats_collapse():
    assert len(norm("date", "2026-10-03", "Oct 3, 2026", "October 3, 2026",
                    "03/10/2026", "2026/10/03")) == 1


def test_email_wrappers_and_case_collapse():
    assert len(norm(EMAIL, "A.B@Corp.Example", "<a.b@corp.example>",
                    "  a.b@corp.example ")) == 1


def test_text_folds_formatting_but_not_wording():
    folded = norm("text", "Hi  there", "hi there", " Hi there\n")
    assert len(folded) == 1
    assert len(norm("text", "hi there", "hello there")) == 2


def test_list_order_does_not_matter():
    normalizer = Normalizer({"tags[*]": "string"})
    first = normalizer.normalize_call({"tags": ["b", "a", "c"]})
    second = normalizer.normalize_call({"tags": ["c", "b", "a"]})
    assert first == second


def test_nested_objects_flatten_to_dotted_paths():
    flat = flatten({"customer": {"email": "a@b.example", "name": "Dana"}})
    assert set(flat) == {"customer.email", "customer.name"}


def test_sniffing_survives_formatting_noise():
    """A single angle-bracketed address must not demote a column of emails."""
    calls = [
        {"to": "a.b@corp.example"},
        {"to": "<c.d@corp.example>"},
        {"to": "  E.F@CORP.EXAMPLE "},
    ]
    assert Normalizer.sniffed(calls).type_of("to") == EMAIL


def test_sniffing_handles_mixed_numbers_and_numeric_strings():
    calls = [{"amount": 1200}, {"amount": "$1,200.00"}, {"amount": "1200.00"}]
    normalizer = Normalizer.sniffed(calls)
    assert normalizer.type_of("amount") == MONEY
    assert len({normalizer.normalize_call(c)["amount"] for c in calls}) == 1


def test_unknown_field_type_is_rejected():
    with pytest.raises(ValueError):
        Normalizer({"f": "not-a-type"})


def test_sniff_type_on_empty_sample_is_stable():
    assert sniff_type("anything", [None, ""]) == "string"
