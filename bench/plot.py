"""Figure generation for the benchmark.

Two panels, one measure each, no second y-axis anywhere. Panel A compares
strategies on the two ways a dedup layer can be wrong. Panel B is the cost
curve: effect error against oracle queries spent.

Colors are categorical slots 1 and 2 of the reference palette, validated for
lightness band, chroma floor, CVD separation, normal-vision separation and
contrast against the chart surface before use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8980"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
GRID = "#e2e1dc"

BAR_PX = 20.0
ROUND_FRACTION = 0.004


def _rounded_bar(ax, y, width, height, color, x_span):
    """A bar with a rounded data-end and a square baseline end."""
    if width <= 0:
        ax.add_patch(
            Rectangle((0, y - height / 2), x_span * 0.0015, height,
                      facecolor=color, edgecolor="none", zorder=3)
        )
        return
    radius = min(x_span * ROUND_FRACTION, width / 2, height / 2)
    ax.add_patch(
        FancyBboxPatch(
            (0, y - height / 2),
            max(width - radius, radius * 0.01),
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            mutation_aspect=1.0,
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
    )
    ax.add_patch(
        Rectangle((0, y - height / 2), radius, height,
                  facecolor=color, edgecolor="none", zorder=4)
    )


def _style_axes(ax, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9.5, labelpad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _panel_a(ax, rows: Sequence[Dict], highlight: Sequence[str]) -> None:
    labels = [r["label"] for r in rows]
    dup = [r["duplicate_pct"] for r in rows]
    lost = [r["lost_pct"] for r in rows]
    span = max(max(dup), max(lost)) * 1.18 or 1.0

    slots = list(range(len(rows)))
    # Two bars per slot. Thickness is capped so the band keeps visible air and
    # the 2px surface gap between the pair survives at render size.
    figure_px = ax.figure.get_size_inches()[1] * ax.figure.dpi
    axis_px = figure_px * 0.72
    thickness = min(0.34, BAR_PX / max(axis_px / max(len(rows), 1), 1.0))
    gap = thickness * 0.16

    for index, (dup_value, lost_value) in enumerate(zip(dup, lost)):
        base = len(rows) - 1 - index
        _rounded_bar(ax, base + (thickness + gap) / 2, dup_value, thickness,
                     SERIES_1, span)
        _rounded_bar(ax, base - (thickness + gap) / 2, lost_value, thickness,
                     SERIES_2, span)

    for index, row in enumerate(rows):
        if row["label"] not in highlight:
            continue
        base = len(rows) - 1 - index
        for value, offset, color in (
            (row["duplicate_pct"], (thickness + gap) / 2, SERIES_1),
            (row["lost_pct"], -(thickness + gap) / 2, SERIES_2),
        ):
            ax.text(
                value + span * 0.022,
                base + offset,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color=INK,
                fontweight="semibold",
                zorder=5,
            )

    ax.set_yticks(slots)
    ax.set_yticklabels(list(reversed(labels)), fontsize=9.5, color=INK)
    for tick, label in zip(ax.get_yticklabels(), reversed(labels)):
        if label in highlight:
            tick.set_color(INK)
            tick.set_fontweight("semibold")
        else:
            tick.set_color(INK_SOFT)
    ax.set_xlim(0, span)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    _style_axes(ax, "percent of intents (lower is better)")
    ax.set_title(
        "A.  Wrong backend state, by keying strategy",
        loc="left", fontsize=11.5, color=INK, fontweight="semibold", pad=30,
    )
    ax.legend(
        handles=[
            Line2D([], [], color=SERIES_1, lw=7, solid_capstyle="round",
                   label="duplicate landings (effect fired twice)"),
            Line2D([], [], color=SERIES_2, lw=7, solid_capstyle="round",
                   label="lost intents (effect never fired)"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.085),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SOFT,
        handlelength=1.1,
        borderpad=0.0,
        columnspacing=1.6,
    )


def _panel_b(ax, curve: Dict[str, List]) -> None:
    series = (
        ("active selection", curve["active"], SERIES_1, "o"),
        ("random pairs", curve["random"], SERIES_2, "s"),
    )
    for name, points, color, marker in series:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=color, lw=2.0, zorder=3,
                solid_capstyle="round", label=name)
        ax.plot(xs, ys, linestyle="none", marker=marker, markersize=6.5,
                markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=2.0, zorder=4)

    for name, points, color, offset in (
        ("active selection", curve["active"], SERIES_1, (8, 10)),
        ("random pairs", curve["random"], SERIES_2, (-10, -30)),
    ):
        last_x, last_y = points[-1]
        ax.annotate(
            f"{last_y:.2f} percent" + chr(10) + f"at {last_x} queries",
            xy=(last_x, last_y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SOFT,
            ha="left" if offset[0] > 0 else "right",
            va="bottom" if offset[1] > 0 else "top",
            zorder=5,
        )

    ax.set_xscale("log")
    _style_axes(ax, "oracle queries spent on labeling (log scale)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("effect error, percent of intents", color=INK_SOFT,
                  fontsize=9.5, labelpad=8)
    top = max(y for points in curve.values() for _, y in points)
    ax.set_ylim(-top * 0.05, top * 1.12)
    ax.set_title(
        "B.  Cost of pinning the key down",
        loc="left", fontsize=11.5, color=INK, fontweight="semibold", pad=30,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.085), ncol=2,
              frameon=False, fontsize=9, labelcolor=INK_SOFT,
              handlelength=1.6, borderpad=0.0, columnspacing=1.6)


def render(
    rows: Sequence[Dict],
    curve: Dict[str, List],
    headline: Dict[str, float],
    out_path: Path,
) -> Path:
    """Write the two-panel comparison figure."""
    plottable = [r for r in rows if not r["label"].startswith("_")]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.4, 6.2), dpi=170,
        gridspec_kw={"width_ratios": [1.52, 1.0], "wspace": 0.30},
    )
    figure.patch.set_facecolor(SURFACE)

    _panel_a(left, plottable, highlight=("no dedup", "effectkey, audited"))
    _panel_b(right, curve)

    figure.suptitle(
        "effectkey: inferred idempotency keys for LLM agent tool calls",
        x=0.012, y=0.982, ha="left", fontsize=14, color=INK,
        fontweight="semibold",
    )
    figure.text(
        0.012, 0.936,
        f"Held-out trace of {headline['attempts']} attempts over "
        f"{headline['intents']} intents across 4 side-effectful tools. "
        f"Wrong backend state falls from {headline['before']:.2f} percent of "
        f"intents to {headline['after']:.2f} percent.",
        ha="left", fontsize=9.5, color=INK_SOFT,
    )

    figure.subplots_adjust(top=0.805, bottom=0.105, left=0.145, right=0.985)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, facecolor=SURFACE)
    plt.close(figure)
    return out_path
