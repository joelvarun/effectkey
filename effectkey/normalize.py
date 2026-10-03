"""Type-driven canonicalization of tool-call argument values.

When a language model retries a tool call it rarely reproduces the original
arguments byte for byte. It reorders keys, reflows whitespace, swaps straight
quotes for curly ones, writes ``1,200.00`` where it previously wrote ``1200``,
and switches date formats. None of that changes what the backend will do, so
none of it should change the effect key.

This module turns a raw leaf value into a canonical form per field type. The
split between "formatting noise" and "meaning" is the whole point: a reworded
email body is a *semantic* change and must survive normalization, while
``Oct 3, 2026`` and ``2026-10-03`` must collapse to the same value.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping, Optional

from .flatten import MultiSet, flatten

STRING = "string"
TEXT = "text"
NUMBER = "number"
MONEY = "money"
BOOL = "bool"
EMAIL = "email"
DATE = "date"
PHONE = "phone"
URL = "url"
IDENT = "ident"
OPAQUE = "opaque"

FIELD_TYPES = (
    STRING, TEXT, NUMBER, MONEY, BOOL, EMAIL, DATE, PHONE, URL, IDENT, OPAQUE,
)

_QUOTE_FOLD = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x2013: "-", 0x2014: "-", 0x2212: "-", 0x00A0: " ",
}

_WS = re.compile(r"\s+")
_NON_DIGIT = re.compile(r"[^0-9]")
_MONEY_STRIP = re.compile(r"[^0-9.\-]")
_TRUE = frozenset({"true", "yes", "y", "1", "on", "t"})
_FALSE = frozenset({"false", "no", "n", "0", "off", "f"})

_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y",
    "%b %d %Y", "%B %d %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
)


def fold_text(value: str) -> str:
    """Unicode and whitespace folding shared by every string-ish type."""
    folded = value.translate(_QUOTE_FOLD)
    folded = unicodedata.normalize("NFKC", folded)
    return _WS.sub(" ", folded).strip()


def _norm_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return fold_text(value).casefold()


def _norm_text(value: Any) -> Any:
    """Free text: fold formatting noise but keep wording and case-insensitivity.

    Rewording is a real change and is intentionally preserved.
    """
    if not isinstance(value, str):
        return value
    return fold_text(value).casefold()


def _norm_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return _canon_decimal(Decimal(str(value)))
    if isinstance(value, str):
        text = fold_text(value).replace(",", "").replace("_", "")
        try:
            return _canon_decimal(Decimal(text))
        except InvalidOperation:
            return fold_text(value).casefold()
    return value


def _norm_money(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return _quantize_money(Decimal(str(value)))
    if isinstance(value, str):
        text = fold_text(value)
        negative = text.startswith("(") and text.endswith(")")
        stripped = _MONEY_STRIP.sub("", text)
        if not stripped or stripped in {"-", ".", "-."}:
            return text.casefold()
        try:
            amount = Decimal(stripped)
        except InvalidOperation:
            return text.casefold()
        if negative:
            amount = -amount
        return _quantize_money(amount)
    return value


def _canon_decimal(amount: Decimal) -> str:
    """Trailing-zero-free canonical decimal string, so 1200 == 1200.00."""
    normalized = amount.normalize()
    if normalized == normalized.to_integral_value():
        normalized = normalized.quantize(Decimal(1))
    return format(normalized, "f")


def _quantize_money(amount: Decimal) -> str:
    return format(amount.quantize(Decimal("0.01")), "f")


def _norm_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = fold_text(value).casefold()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        return text
    return value


def _norm_email(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = fold_text(value).strip("<>").casefold()
    if "@" not in text:
        return text
    local, _, domain = text.rpartition("@")
    return f"{local}@{domain.rstrip('.')}"


def _norm_date(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    if not isinstance(value, str):
        return value
    text = fold_text(value)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text.casefold()


def _norm_phone(value: Any) -> Any:
    if not isinstance(value, str):
        value = str(value)
    text = fold_text(value)
    plus = text.lstrip().startswith("+")
    digits = _NON_DIGIT.sub("", text)
    return ("+" + digits) if plus else digits


def _norm_url(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = fold_text(value)
    scheme, sep, rest = text.partition("://")
    if sep:
        scheme = scheme.casefold()
    else:
        scheme, rest = "", text
    path, _, query = rest.partition("?")
    host, slash, tail = path.partition("/")
    canonical = f"{scheme}://{host.casefold()}" if sep else host.casefold()
    tail = tail.rstrip("/")
    if slash and tail:
        canonical = f"{canonical}/{tail}"
    if query:
        params = sorted(part for part in query.split("&") if part)
        canonical = f"{canonical}?{'&'.join(params)}"
    return canonical


def _norm_ident(value: Any) -> Any:
    if isinstance(value, str):
        return fold_text(value)
    return value


def _norm_opaque(value: Any) -> Any:
    return value


_NORMALIZERS = {
    STRING: _norm_string,
    TEXT: _norm_text,
    NUMBER: _norm_number,
    MONEY: _norm_money,
    BOOL: _norm_bool,
    EMAIL: _norm_email,
    DATE: _norm_date,
    PHONE: _norm_phone,
    URL: _norm_url,
    IDENT: _norm_ident,
    OPAQUE: _norm_opaque,
}


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_URL_RE = re.compile(r"^(https?|ftp)://", re.IGNORECASE)
_MONEY_RE = re.compile(r"^[\s(]*[$€£¥]\s*-?[0-9][0-9,_]*(\.[0-9]+)?[\s)]*$")
_NUMBER_RE = re.compile(r"^-?[0-9][0-9,_]*(\.[0-9]+)?$")
_PHONE_RE = re.compile(r"^\+?[0-9][0-9\s().-]{6,}$")

_DATE_HINTS = ("date", "due", "when", "deadline", "at", "on", "day")
_MONEY_HINTS = ("amount", "price", "total", "cost", "fee", "charge", "value")
_TEXT_HINTS = ("body", "text", "message", "note", "comment", "description", "content")


def sniff_type(path: str, values: Any) -> str:
    """Infer a field type from its name and observed values.

    Used when no schema is supplied. Name hints only break ties; the observed
    values decide whenever they are unambiguous.
    """
    samples = [v for v in values if v is not None and v != ""]
    if not samples:
        return STRING

    leaf = path.rsplit(".", 1)[-1].removesuffix("[*]").casefold()

    if all(isinstance(v, bool) for v in samples):
        return BOOL
    if all(isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) for v in samples):
        return MONEY if any(h in leaf for h in _MONEY_HINTS) else NUMBER

    # Mixed numbers and numeric strings are the common rewrite case: the model
    # writes 1200 one time and "$1,200.00" the next. Classify on the union.
    if all(_numeric_like(v) for v in samples):
        if any(_money_like(v) for v in samples) or any(h in leaf for h in _MONEY_HINTS):
            return MONEY
        return NUMBER
    if all(_bool_like(v) for v in samples):
        return BOOL

    raw_strings = [v for v in samples if isinstance(v, str)]
    if len(raw_strings) != len(samples):
        return OPAQUE

    # Sniff on folded values. Type detection has to be invariant to exactly the
    # noise normalization removes, or a single angle-bracketed address is enough
    # to make a column of emails look like free text.
    strings = [fold_text(v) for v in raw_strings]
    unwrapped = [v.strip("<>") for v in strings]

    if all(_EMAIL_RE.match(v) for v in unwrapped):
        return EMAIL
    if all(_URL_RE.match(v) for v in strings):
        return URL
    if all(_MONEY_RE.match(v) for v in strings):
        return MONEY
    if all(_NUMBER_RE.match(v) for v in strings):
        return MONEY if any(h in leaf for h in _MONEY_HINTS) else NUMBER
    if all(_looks_like_date(v) for v in strings):
        return DATE
    if all(v.casefold() in _TRUE | _FALSE for v in strings):
        return BOOL
    if any(h in leaf for h in _PHONE_HINTS) and all(_PHONE_RE.match(v) for v in strings):
        return PHONE
    if any(h in leaf for h in _TEXT_HINTS):
        return TEXT
    if any(h in leaf for h in _DATE_HINTS) and all(_looks_like_date(v) for v in strings):
        return DATE
    return STRING



def _numeric_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, Decimal)):
        return True
    if isinstance(value, str):
        text = fold_text(value)
        return bool(_NUMBER_RE.match(text) or _MONEY_RE.match(text))
    return False


def _money_like(value: Any) -> bool:
    return isinstance(value, str) and bool(_MONEY_RE.match(fold_text(value)))


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return fold_text(value).casefold() in _TRUE | _FALSE
    return False


_PHONE_HINTS = ("phone", "mobile", "tel", "cell", "fax")


def _looks_like_date(value: str) -> bool:
    text = fold_text(value)
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    return False


class Normalizer:
    """Applies per-field canonicalization to a flattened argument object."""

    def __init__(
        self,
        types: Optional[Mapping[str, str]] = None,
        default_type: str = STRING,
        set_paths: Optional[set] = None,
    ) -> None:
        bad = {t for t in (types or {}).values() if t not in FIELD_TYPES}
        if bad:
            raise ValueError(f"unknown field types: {sorted(bad)}")
        if default_type not in FIELD_TYPES:
            raise ValueError(f"unknown default field type: {default_type!r}")
        self.types: Dict[str, str] = dict(types or {})
        self.default_type = default_type
        self.set_paths = set(set_paths or ())

    @classmethod
    def sniffed(cls, calls, set_paths: Optional[set] = None) -> "Normalizer":
        """Build a normalizer by inferring each field's type from a call sample."""
        set_paths = set(set_paths or ())
        observed: Dict[str, list] = {}
        for call in calls:
            for path, value in flatten(call, set_paths).items():
                bucket = observed.setdefault(path, [])
                if isinstance(value, MultiSet):
                    bucket.extend(value.values)
                else:
                    bucket.append(value)
        types = {path: sniff_type(path, vals) for path, vals in observed.items()}
        return cls(types=types, set_paths=set_paths)

    def type_of(self, path: str) -> str:
        return self.types.get(path, self.default_type)

    def normalize_value(self, path: str, value: Any) -> Any:
        field_type = self.type_of(path)
        if isinstance(value, MultiSet):
            normalized = [_NORMALIZERS[field_type](v) for v in value.values]
            return tuple(sorted(normalized, key=lambda v: (type(v).__name__, str(v))))
        return _NORMALIZERS[field_type](value)

    def normalize_call(self, call: Mapping[str, Any]) -> Dict[str, Any]:
        """Flatten and canonicalize a whole argument object."""
        flat = flatten(call, self.set_paths)
        return {path: self.normalize_value(path, value) for path, value in flat.items()}
