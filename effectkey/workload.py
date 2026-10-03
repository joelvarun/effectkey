"""Synthetic agent tool-call workload with hidden ground truth.

The generator keeps two parallel representations of every call:

``semantic``
    What the call means. One value per field, in canonical form.
``wire``
    What the agent actually emitted. A formatting variant of the semantic
    value, with keys in arbitrary order and optional fields added or dropped.

Ground truth effect identity is computed from ``semantic`` restricted to each
tool's hidden effect fields. The system under test only ever sees ``wire``. No
component of the dedup pipeline can see the semantic layer, so a normalizer
that fails to undo a formatting variant produces a real, counted duplicate
rather than a hidden pass.

Retries reformat effect fields but never change their meaning, which is what a
retry is. Noise fields may change meaning freely, because an agent rewriting a
call does reword the human-facing parts of it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SUCCESS = "success"
CLEAN_FAIL = "clean_fail"
AMBIGUOUS_FAIL = "ambiguous_fail"

EMAIL = "email"
MONEY = "money"
DATE = "date"
IDENT = "ident"
TEXT = "text"
NAME = "name"
BOOL = "bool"
TAGS = "tags"
CODE = "code"


@dataclass(frozen=True)
class FieldSpec:
    """One argument field, and whether the backend's effect depends on it."""

    path: str
    kind: str
    effect: bool
    optional: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fields: Tuple[FieldSpec, ...]

    @property
    def effect_fields(self) -> Tuple[str, ...]:
        return tuple(f.path for f in self.fields if f.effect)


@dataclass(frozen=True)
class Attempt:
    """One emitted tool call, and what the backend would do if it ran.

    The trace is fixed before any dedup strategy sees it, so every strategy is
    replayed against identical input. ``outcome`` is conditional: it describes
    what happens *if* this attempt is executed, which is what lets the harness
    count real landings rather than assuming them.

    Holding the attempt stream fixed is deliberately conservative. A dedup layer
    that returns a cached success would also cut the agent's retry loop short,
    so the measured benefit understates the real one.
    """

    attempt_id: str
    tool: str
    intent_id: str
    args: Mapping[str, Any]
    timestamp: float
    outcome: str
    is_retry: bool

    @property
    def landed_if_executed(self) -> bool:
        """Whether running this attempt reaches the backend and applies."""
        return self.outcome in (SUCCESS, AMBIGUOUS_FAIL)


@dataclass
class Workload:
    attempts: List[Attempt] = field(default_factory=list)
    effect_of_intent: Dict[str, str] = field(default_factory=dict)
    tools: Tuple[ToolSpec, ...] = ()

    def intents(self) -> Dict[str, List[Attempt]]:
        grouped: Dict[str, List[Attempt]] = {}
        for attempt in self.attempts:
            grouped.setdefault(attempt.intent_id, []).append(attempt)
        return grouped


TOOLS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="send_invoice",
        fields=(
            FieldSpec("customer.email", EMAIL, effect=True),
            FieldSpec("invoice_number", IDENT, effect=True),
            FieldSpec("amount", MONEY, effect=True),
            FieldSpec("customer.name", NAME, effect=False),
            FieldSpec("note", TEXT, effect=False),
            FieldSpec("due_date", DATE, effect=False),
            FieldSpec("locale", CODE, effect=False, optional=True),
            FieldSpec("send_copy", BOOL, effect=False, optional=True),
            FieldSpec("tags", TAGS, effect=False, optional=True),
        ),
    ),
    ToolSpec(
        name="post_message",
        fields=(
            FieldSpec("channel", IDENT, effect=True),
            FieldSpec("text", TEXT, effect=True),
            FieldSpec("username", NAME, effect=False),
            FieldSpec("unfurl_links", BOOL, effect=False, optional=True),
            FieldSpec("icon", CODE, effect=False, optional=True),
        ),
    ),
    ToolSpec(
        name="create_shipment",
        fields=(
            FieldSpec("order_id", IDENT, effect=True),
            FieldSpec("carrier", CODE, effect=True),
            FieldSpec("address.line1", TEXT, effect=True),
            FieldSpec("address.postal_code", CODE, effect=True),
            FieldSpec("notes", TEXT, effect=False),
            FieldSpec("requested_by", EMAIL, effect=False),
            FieldSpec("ship_date", DATE, effect=False, optional=True),
            FieldSpec("label_format", CODE, effect=False, optional=True),
        ),
    ),
    ToolSpec(
        name="charge_card",
        fields=(
            FieldSpec("card_token", IDENT, effect=True),
            FieldSpec("amount", MONEY, effect=True),
            FieldSpec("currency", CODE, effect=True),
            FieldSpec("description", TEXT, effect=False),
            FieldSpec("metadata.order", IDENT, effect=False, optional=True),
            FieldSpec("statement_suffix", CODE, effect=False, optional=True),
        ),
    ),
)


_FIRST = ("dana", "amir", "lena", "raj", "nora", "theo", "ivy", "marco", "sana", "kofi")
_LAST = ("shaw", "okoro", "vargas", "lindqvist", "haddad", "patel", "moreau", "chen")
_DOMAIN = ("corp.example", "acme-supply.example", "northwind.example", "vertex.example")
_CARRIERS = ("ups", "dhl", "fedex", "royalmail", "aramex")
_CURRENCY = ("usd", "eur", "gbp", "inr", "jpy")
_LOCALE = ("en-us", "en-gb", "de-de", "fr-fr")
_LABEL = ("pdf", "zpl", "png")
_ICON = ("robot", "bell", "chart", "wrench")
_CHANNELS = (
    "ops-alerts", "billing", "shipping-desk", "finance-eod", "support",
    "warehouse-a", "warehouse-b", "carrier-escalation", "treasury", "tax-desk",
    "returns", "procurement", "night-shift", "exec-digest", "audit-trail",
)
_STREET = ("main street", "oak avenue", "harbour road", "north suite", "apartment 4b")
_TAGPOOL = ("urgent", "q4", "renewal", "net30", "audited", "manual", "eu", "apac")

_NOTE_TEMPLATES = (
    "please find the attached statement for your records.",
    "payment terms are net thirty from the issue date.",
    "let us know if the purchase order number is missing.",
    "this supersedes the draft sent earlier this week.",
    "the finance team has approved the adjusted total.",
)

_MESSAGE_TEMPLATES = (
    "nightly reconciliation finished with zero open items.",
    "shipment backlog cleared for the western region.",
    "two invoices are awaiting manual approval.",
    "the billing export completed in under four minutes.",
    "carrier rates refreshed from the latest contract.",
    "the duty drawback filing is ready for review.",
    "three shipments missed the carrier cutoff window.",
    "the quarterly accrual journal posted successfully.",
    "inbound dock capacity drops below plan on friday.",
    "a customer credit memo needs a second approver.",
    "the tax engine returned a stale rate table.",
    "label stock at the west dock is running low.",
    "an order was released without a shipping method.",
    "the payment file cleared the bank without errors.",
    "freight invoices exceeded the audit tolerance.",
)

_ABBREV = {
    "street": "st",
    "avenue": "ave",
    "apartment": "apt",
    "suite": "ste",
    "north": "n",
    "road": "rd",
}


def _semantic_value(kind: str, rng: random.Random) -> Any:
    """Draw a canonical semantic value for a field kind."""
    if kind == EMAIL:
        return f"{rng.choice(_FIRST)}.{rng.choice(_LAST)}@{rng.choice(_DOMAIN)}"
    if kind == MONEY:
        cents = rng.choice((0, 0, 0, 50, 25, 99))
        return format(Decimal(rng.randrange(40, 9800)) + Decimal(cents) / 100, ".2f")
    if kind == DATE:
        return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if kind == IDENT:
        prefix = rng.choice(("INV", "ORD", "TOK", "CH"))
        return f"{prefix}-{rng.randrange(10000, 99999)}"
    if kind == TEXT:
        return rng.choice(_NOTE_TEMPLATES + _MESSAGE_TEMPLATES + _STREET)
    if kind == NAME:
        return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
    if kind == BOOL:
        return rng.random() < 0.5
    if kind == TAGS:
        return tuple(sorted(rng.sample(_TAGPOOL, rng.randint(1, 3))))
    if kind == CODE:
        pool = _CARRIERS + _CURRENCY + _LOCALE + _LABEL + _ICON + _CHANNELS
        return rng.choice(pool)
    raise ValueError(f"unknown field kind: {kind!r}")


def _semantic_for_path(spec: FieldSpec, rng: random.Random) -> Any:
    """Field-aware semantic draw, so codes stay inside their own vocabulary."""
    leaf = spec.path.rsplit(".", 1)[-1]
    if spec.kind == CODE:
        pool = {
            "carrier": _CARRIERS,
            "currency": _CURRENCY,
            "locale": _LOCALE,
            "label_format": _LABEL,
            "icon": _ICON,
            "statement_suffix": _CURRENCY,
            "postal_code": None,
        }.get(leaf, _CARRIERS + _CURRENCY)
        if leaf == "postal_code":
            return f"{rng.randrange(10000, 99999)}"
        return rng.choice(pool)
    if spec.kind == IDENT and leaf == "channel":
        return rng.choice(_CHANNELS)
    if spec.kind == TEXT and leaf in ("line1",):
        return f"{rng.randrange(1, 900)} {rng.choice(_STREET)}"
    if spec.kind == TEXT and leaf == "text":
        return rng.choice(_MESSAGE_TEMPLATES)
    return _semantic_value(spec.kind, rng)


def _render(kind: str, semantic: Any, rng: random.Random) -> Any:
    """Pick a formatting variant of a semantic value.

    Every variant here preserves meaning. These are the rewrites a model makes
    when it reissues a call it has already made once.
    """
    if kind == EMAIL:
        return rng.choice(
            (semantic, semantic.upper(), f"<{semantic}>", f"  {semantic} ",
             semantic.capitalize())
        )
    if kind == MONEY:
        amount = Decimal(semantic)
        integral = amount == amount.to_integral_value()
        choices = [semantic, f"${semantic}", f"{amount:,.2f}", f"${amount:,.2f}",
                   f" {semantic} "]
        if integral:
            whole = int(amount)
            choices += [str(whole), f"{whole:,}", whole, float(whole)]
        return rng.choice(choices)
    if kind == DATE:
        from datetime import datetime

        parsed = datetime.strptime(semantic, "%Y-%m-%d")
        return rng.choice(
            (semantic,
             parsed.strftime("%b %d, %Y").replace(" 0", " "),
             parsed.strftime("%B %d, %Y").replace(" 0", " "),
             parsed.strftime("%d/%m/%Y"),
             parsed.strftime("%Y/%m/%d"))
        )
    if kind == IDENT:
        return rng.choice((semantic, f" {semantic}", f"{semantic} ",
                           f"{semantic} ", semantic))
    if kind in (TEXT, NAME):
        text = str(semantic)
        return rng.choice(
            (text,
             text.replace(" ", "  "),
             text.capitalize(),
             text.replace("'", "’").replace('"', "“"),
             f" {text}\n")
        )
    if kind == BOOL:
        return rng.choice((semantic, str(semantic).lower(),
                           "yes" if semantic else "no"))
    if kind == TAGS:
        shuffled = list(semantic)
        rng.shuffle(shuffled)
        return shuffled
    if kind == CODE:
        return rng.choice((semantic, semantic.upper(), f" {semantic} "))
    raise ValueError(f"unknown field kind: {kind!r}")


def _abbreviate(text: str) -> str:
    """Domain-equivalent rewrite that no generic formatter can undo.

    The backend treats "123 Main Street" and "123 Main St" as one address, so
    the effect is the same. String canonicalization cannot know that. This is
    the honest residual: field-set inference fixes which fields matter, not
    value-level domain equivalence.
    """
    words = text.split()
    return " ".join(_ABBREV.get(w.lower(), w) for w in words)


def _nest(flat: Mapping[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Build the nested wire object, with keys emitted in arbitrary order.

    Key order is the cheapest rewrite a model makes and the one that defeats
    hashing a raw JSON string, so it is randomized on every attempt.
    """
    paths = list(flat)
    rng.shuffle(paths)
    out: Dict[str, Any] = {}
    for path in paths:
        parts = path.split(".")
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = flat[path]
    return out


@dataclass
class GeneratorConfig:
    """Rates controlling the shape of the generated trace."""

    retry_rate: float = 0.38
    max_retries: int = 2
    clean_fail_rate: float = 0.55
    noise_rewrite_rate: float = 0.45
    optional_churn_rate: float = 0.30
    abbrev_rate: float = 0.08
    near_miss_rate: float = 0.22
    resend_rate: float = 0.10
    resend_min_lag: int = 18
    retry_gap: float = 4.0
    intent_gap: float = 90.0


class WorkloadGenerator:
    """Generates a replayable agent tool-call trace with hidden ground truth."""

    def __init__(
        self,
        seed: int = 0,
        tools: Sequence[ToolSpec] = TOOLS,
        config: Optional[GeneratorConfig] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.tools = tuple(tools)
        self.config = config or GeneratorConfig()

    def _draw_semantic(self, tool: ToolSpec) -> Dict[str, Any]:
        rng = self.rng
        semantic: Dict[str, Any] = {}
        for spec in tool.fields:
            if spec.optional and rng.random() < 0.45:
                continue
            semantic[spec.path] = _semantic_for_path(spec, rng)
        return semantic

    def _near_miss(self, tool: ToolSpec, base: Mapping[str, Any]) -> Dict[str, Any]:
        """A genuinely different call that differs in exactly one effect field.

        This is the case that punishes similarity thresholds: almost every byte
        matches, and suppressing it loses a real side effect.
        """
        rng = self.rng
        semantic = dict(base)
        candidates = [f for f in tool.fields if f.effect and f.path in semantic]
        target = rng.choice(candidates)
        for _ in range(12):
            replacement = _semantic_for_path(target, rng)
            if replacement != semantic[target.path]:
                semantic[target.path] = replacement
                return semantic
        semantic[target.path] = f"{semantic[target.path]}-b"
        return semantic

    def _render_attempt(
        self,
        tool: ToolSpec,
        semantic: Mapping[str, Any],
        is_retry: bool,
    ) -> Dict[str, Any]:
        rng = self.rng
        cfg = self.config
        by_path = {spec.path: spec for spec in tool.fields}
        flat: Dict[str, Any] = {}

        for path, value in semantic.items():
            spec = by_path[path]
            if is_retry and spec.optional and rng.random() < cfg.optional_churn_rate:
                continue
            payload = value
            if is_retry and not spec.effect and rng.random() < cfg.noise_rewrite_rate:
                payload = _semantic_for_path(spec, rng)
            rendered = _render(spec.kind, payload, rng)
            if (
                is_retry
                and spec.effect
                and spec.kind in (TEXT, NAME)
                and rng.random() < cfg.abbrev_rate
            ):
                rendered = _abbreviate(str(rendered))
            flat[path] = rendered

        if is_retry:
            for spec in tool.fields:
                if (
                    spec.optional
                    and spec.path not in flat
                    and rng.random() < cfg.optional_churn_rate
                ):
                    flat[spec.path] = _render(
                        spec.kind, _semantic_for_path(spec, rng), rng
                    )

        return _nest(flat, rng)

    def generate(self, n_intents: int = 600) -> Workload:
        """Produce a time-ordered trace of ``n_intents`` distinct intents.

        Each intent should land exactly once. Retries of an intent carry the
        same meaning in reformatted clothes. Near-miss intents differ from a
        recent intent in exactly one effect field. Resends reuse the effect
        values of an intent far enough in the past that a dedup window has
        legitimately expired.
        """
        if n_intents <= 0:
            raise ValueError("n_intents must be positive")
        rng = self.rng
        cfg = self.config
        workload = Workload(tools=self.tools)
        history: Dict[str, List[Dict[str, Any]]] = {t.name: [] for t in self.tools}
        clock = 0.0

        for index in range(n_intents):
            tool = rng.choice(self.tools)
            past = history[tool.name]

            if past and rng.random() < cfg.near_miss_rate:
                semantic = self._near_miss(tool, past[-1])
            elif len(past) > cfg.resend_min_lag and rng.random() < cfg.resend_rate:
                source = past[rng.randrange(0, len(past) - cfg.resend_min_lag)]
                semantic = {
                    spec.path: (
                        source[spec.path]
                        if spec.effect and spec.path in source
                        else _semantic_for_path(spec, rng)
                    )
                    for spec in tool.fields
                    if spec.effect or spec.path in source
                }
            else:
                semantic = self._draw_semantic(tool)

            past.append(dict(semantic))
            intent_id = f"i{index:05d}"
            workload.effect_of_intent[intent_id] = repr(
                (tool.name, tuple(sorted(
                    (p, v) for p, v in semantic.items()
                    if p in set(tool.effect_fields)
                )))
            )

            retries = (
                rng.randint(1, cfg.max_retries)
                if rng.random() < cfg.retry_rate
                else 0
            )
            for step in range(retries + 1):
                last = step == retries
                outcome = (
                    SUCCESS
                    if last
                    else (
                        CLEAN_FAIL
                        if rng.random() < cfg.clean_fail_rate
                        else AMBIGUOUS_FAIL
                    )
                )
                workload.attempts.append(
                    Attempt(
                        attempt_id=f"{intent_id}a{step}",
                        tool=tool.name,
                        intent_id=intent_id,
                        args=self._render_attempt(tool, semantic, is_retry=step > 0),
                        timestamp=clock + step * cfg.retry_gap,
                        outcome=outcome,
                        is_retry=step > 0,
                    )
                )
            clock += cfg.intent_gap

        return workload


def labeled_log(
    workload: Workload,
    seed: int = 0,
    same_per_tool: int = 60,
    distinct_per_tool: int = 220,
    label_noise: float = 0.0,
):
    """Derive a pair-labeled training log from a trace.

    SAME pairs come from two attempts of one intent, which is what a backend
    conflict or a duplicate incident actually tells you. DISTINCT pairs come
    from intents with different effect signatures.

    Pairs drawn from two different intents that happen to share an effect
    signature (the resend case) are left unlabeled on purpose: in production
    nobody files a conflict for a resend that the dedup window has already let
    through, so pretending that label exists would be inventing evidence.
    """
    from .constraints import CallRecord, PairLabel

    rng = random.Random(seed)
    calls = [
        CallRecord(call_id=a.attempt_id, tool=a.tool, args=a.args)
        for a in workload.attempts
    ]

    per_tool_intents: Dict[str, Dict[str, List[Attempt]]] = {}
    for attempt in workload.attempts:
        per_tool_intents.setdefault(attempt.tool, {}).setdefault(
            attempt.intent_id, []
        ).append(attempt)

    labels: List[PairLabel] = []
    for tool, intents in per_tool_intents.items():
        multi = [v for v in intents.values() if len(v) > 1]
        rng.shuffle(multi)
        for group in multi[:same_per_tool]:
            left, right = rng.sample(group, 2)
            labels.append(PairLabel(left.attempt_id, right.attempt_id, True))

        ids = list(intents)
        picked = 0
        guard = 0
        while picked < distinct_per_tool and guard < distinct_per_tool * 40:
            guard += 1
            a_id, b_id = rng.sample(ids, 2)
            if workload.effect_of_intent[a_id] == workload.effect_of_intent[b_id]:
                continue
            left = rng.choice(intents[a_id])
            right = rng.choice(intents[b_id])
            labels.append(PairLabel(left.attempt_id, right.attempt_id, False))
            picked += 1

    if label_noise > 0:
        labels = [
            PairLabel(lbl.a, lbl.b, not lbl.same_effect)
            if rng.random() < label_noise
            else lbl
            for lbl in labels
        ]
    return calls, labels
