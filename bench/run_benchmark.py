"""Run the full benchmark and write results plus the comparison figure.

    python bench/run_benchmark.py

Writes ``bench/results.json`` and ``docs/before_after.png``. Everything is
seeded, so a rerun reproduces the same numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import plot
from bench.pipeline import (
    N_INTENTS,
    build_bundle,
    field_set_accuracy,
    learning_curve,
    run_inference,
    score_all,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", type=int, default=N_INTENTS)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "before_after.png")
    parser.add_argument("--results", type=Path, default=ROOT / "bench" / "results.json")
    args = parser.parse_args()

    print(f"building workload ({args.intents} intents per split)", flush=True)
    bundle = build_bundle(args.intents)
    print(
        f"  train: {len(bundle.train.attempts)} attempts  "
        f"test: {len(bundle.test.attempts)} attempts  "
        f"tools: {len(bundle.truth)}",
        flush=True,
    )

    print("inferring effect keys", flush=True)
    inferred = run_inference(bundle)
    print(
        f"  active selection: {inferred['active_queries']} oracle queries\n"
        f"  verification:     {inferred['verify_queries']} oracle queries\n"
        f"  random baseline:  {inferred['random_queries']} oracle queries",
        flush=True,
    )

    accuracy = {
        name: field_set_accuracy(inferred[key], bundle.truth)
        for name, key in (
            ("random_labels", "specs_random"),
            ("active_labels", "specs_active"),
            ("active_greedy", "specs_greedy"),
            ("audited", "specs_verified"),
        )
    }
    for name, (exact, jaccard) in accuracy.items():
        print(
            f"  field set {name:14s} exact {exact}/{len(bundle.truth)}  "
            f"mean Jaccard {jaccard:.3f}",
            flush=True,
        )

    print("scoring strategies on the held-out trace", flush=True)
    rows = score_all(bundle, inferred)
    print("building learning curve", flush=True)
    curve = learning_curve(bundle)

    scored = [r for r in rows if not r["label"].startswith("_")]
    before = next(r for r in scored if r["label"] == "no dedup")
    after = next(r for r in scored if r["label"] == "effectkey, audited")

    print()
    header = f"{'strategy':30s}{'dup %':>9s}{'lost %':>9s}{'error %':>9s}"
    print(header)
    print("-" * len(header))
    for row in scored:
        print(
            f"{row['label']:30s}{row['duplicate_pct']:9.2f}"
            f"{row['lost_pct']:9.2f}{row['effect_error_pct']:9.2f}"
        )
    print("-" * len(header))
    reduction = (
        before["effect_error_pct"] / after["effect_error_pct"]
        if after["effect_error_pct"]
        else float("inf")
    )
    print(
        f"effect error {before['effect_error_pct']:.2f} -> "
        f"{after['effect_error_pct']:.2f} percent of intents "
        f"({reduction:.1f}x lower)"
    )

    payload = {
        "config": {
            "intents_per_split": args.intents,
            "train_attempts": len(bundle.train.attempts),
            "test_attempts": len(bundle.test.attempts),
            "tools": sorted(bundle.truth),
            "ttl_seconds": 900.0,
        },
        "oracle_queries": {
            "active_selection": inferred["active_queries"],
            "verification": inferred["verify_queries"],
            "total": inferred["total_queries"],
            "random_baseline": inferred["random_queries"],
        },
        "field_set_accuracy": {
            name: {"exact": exact, "of": len(bundle.truth), "mean_jaccard": round(j, 4)}
            for name, (exact, j) in accuracy.items()
        },
        "inferred_keys": {
            tool: {
                "key_fields": list(spec.key_fields),
                "ground_truth": sorted(bundle.truth[tool]),
                "match": set(spec.key_fields) == bundle.truth[tool],
                "forbidden": list(spec.forbidden),
                "solver_optimal": spec.optimal,
                "branch_and_bound_nodes": spec.nodes,
            }
            for tool, spec in sorted(inferred["specs_verified"].items())
        },
        "audit": {
            tool: [r.as_dict() for r in reports]
            for tool, reports in sorted(inferred["audit_trail"].items())
        },
        "results": rows,
        "learning_curve": curve,
        "headline": {
            "metric": "wrong backend state (duplicate or lost side effects), "
                      "percent of intents",
            "baseline_no_dedup": before["effect_error_pct"],
            "effectkey_audited": after["effect_error_pct"],
            "reduction_factor": round(reduction, 2),
        },
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.results.relative_to(ROOT)}")

    out = plot.render(
        rows,
        curve,
        {
            "attempts": len(bundle.test.attempts),
            "intents": args.intents,
            "before": before["effect_error_pct"],
            "after": after["effect_error_pct"],
        },
        args.out,
    )
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
