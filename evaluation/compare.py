"""
compare.py
==========
Compares evaluation reports from multiple models side by side.
Prints a comparison table and saves compare_report.json

Usage:
    python -m evaluation.compare \
        --reports results/base_eval.json results/sft_eval.json results/dpo_eval.json
"""

import json
import argparse
from pathlib import Path
from config import RESULTS_DIR


def load_report(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(report_paths: list[str]):
    reports = [load_report(p) for p in report_paths]
    model_names = [r["model"] for r in reports]

    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(f"Models: {' | '.join(model_names)}\n")

    # ── Readability metrics table ──
    metric_keys = [
        ("flesch_reading_ease", "Flesch reading ease  (↑ easier)"),
        ("flesch_kincaid_grade", "Grade level          (↓ simpler)"),
        ("word_count", "Avg word count"),
        ("jargon_density", "Jargon density       (↓ better)"),
        ("example_score", "Example signals      (↑ better)"),
        ("analogy_score", "Analogy signals      (↑ better)"),
    ]

    print("READABILITY METRICS")
    print(f"{'Metric':<35}", end="")
    for name in model_names:
        print(f"{name:>12}", end="")
    print()
    print("-" * (35 + 12 * len(model_names)))

    for key, label in metric_keys:
        print(f"{label:<35}", end="")
        for r in reports:
            val = r["summary"]["overall_metrics"].get(key, "n/a")
            print(f"{str(val):>12}", end="")
        print()

    # ── Judge scores table ──
    judge_keys = [
        ("clarity", "Clarity              (↑ better)"),
        ("beginner_friendliness", "Beginner friendly    (↑ better)"),
        ("use_of_examples", "Use of examples      (↑ better)"),
        ("jargon_handling", "Jargon handling      (↑ better)"),
        ("logical_flow", "Logical flow         (↑ better)"),
        ("overall", "OVERALL SCORE"),
    ]

    has_judge = all(r["summary"].get("overall_judge") for r in reports)

    if has_judge:
        print("\nLLM-AS-JUDGE SCORES (1-5)")
        print(f"{'Metric':<35}", end="")
        for name in model_names:
            print(f"{name:>12}", end="")
        print()
        print("-" * (35 + 12 * len(model_names)))

        for key, label in judge_keys:
            print(f"{label:<35}", end="")
            vals = []
            for r in reports:
                val = r["summary"]["overall_judge"].get(key, "n/a")
                vals.append(val)
                print(f"{str(val):>12}", end="")
            print()

        # Improvement over baseline
        if len(reports) > 1:
            print(f"\nIMPROVEMENT OVER BASELINE ({model_names[0]})")
            baseline_overall = reports[0]["summary"]["overall_judge"].get("overall", 0)
            for r in reports[1:]:
                model_overall = r["summary"]["overall_judge"].get("overall", 0)
                delta = round(model_overall - baseline_overall, 3)
                sign = "+" if delta > 0 else ""
                print(f"  {r['model']:20s} {sign}{delta} overall score")

    # ── Category breakdown ──
    print("\nOVERALL JUDGE SCORE BY CATEGORY")
    print(f"{'Category':<20}", end="")
    for name in model_names:
        print(f"{name:>12}", end="")
    print()
    print("-" * (20 + 12 * len(model_names)))

    all_categorys = set()
    for r in reports:
        all_categorys.update(r["summary"]["by_category"].keys())

    for category in sorted(all_categorys):
        print(f"{category:<20}", end="")
        for r in reports:
            val = (
                r["summary"]["by_category"]
                .get(category, {})
                .get("judge_overall", "n/a")
            )
            print(f"{str(val):>12}", end="")
        print()

    # ── Save comparison report ──
    comparison = {
        "models": model_names,
        "reports": {r["model"]: r["summary"] for r in reports},
    }
    output_path = RESULTS_DIR / "comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\n✓ Comparison saved → {output_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Paths to eval JSON files, e.g. results/base_eval.json results/sft_eval.json",
    )
    args = parser.parse_args()
    compare(args.reports)
