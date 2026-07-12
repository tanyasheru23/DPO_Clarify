"""
evaluate.py
===========
Runs metrics + judge on a generated responses file.
Saves full evaluation report to results/{model_name}_eval.json

Usage:
    python -m evaluation.evaluate --responses results/base_responses.json
    python -m evaluation.evaluate --responses results/sft_responses.json
    python -m evaluation.evaluate --responses results/dpo_responses.json

    # Skip judge (only compute textstat + heuristics, no API cost)
    python -m evaluation.evaluate --responses results/base_responses.json --no-judge
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from evaluation.metrics import compute_all_metrics
from evaluation.judge import judge_all
from config import RESULTS_DIR


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate(responses_path: str, skip_judge: bool = False) -> dict:
    """
    Load responses, compute metrics + judge scores, save report.
    """
    path = Path(responses_path)
    responses = json.loads(path.read_text(encoding="utf-8"))
    model_name = responses[0]["model"] if responses else "unknown"

    print(f"Evaluating {len(responses)} responses from model: {model_name}")

    # ── Step 1: compute objective metrics ──
    print("\nComputing readability + heuristic metrics...")
    for item in responses:
        item["metrics"] = compute_all_metrics(item["prompt"], item["response"])

    # ── Step 2: LLM-as-judge ──
    if not skip_judge:
        responses = judge_all(responses)
    else:
        print("Skipping judge (--no-judge flag set)")
        for item in responses:
            item["judge_scores"] = None

    # ── Step 3: compute summary stats ──
    summary = compute_summary(responses, skip_judge)

    # ── Step 4: build report ──
    report = {
        "model":     model_name,
        "n_samples": len(responses),
        "summary":   summary,
        "results":   responses,
    }

    # ── Step 5: save ──
    output_path = RESULTS_DIR / f"{model_name}_eval.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n✓ Saved evaluation report → {output_path}")

    # ── Step 6: print summary to terminal ──
    print_summary(summary, model_name)

    return report


def compute_summary(responses: list[dict], skip_judge: bool) -> dict:
    """Compute average scores across all responses, broken down by category."""

    # Overall averages — metrics
    metric_keys = [
        "flesch_reading_ease", "flesch_kincaid_grade", "word_count",
        "jargon_density", "example_score", "analogy_score",
        "definition_first", "has_steps",
        "bertscore_f1", "is_non_english", "non_english_ratio",
        "has_repeated_prompt", "formatting_error_rate"
    ]
    overall_metrics = {}
    for key in metric_keys:
        vals = [r["metrics"][key] for r in responses if key in r.get("metrics", {})]
        overall_metrics[key] = round(sum(vals) / len(vals), 3) if vals else 0

    # Overall averages — judge scores
    overall_judge = {}
    if not skip_judge:
        judge_keys = ["clarity", "beginner_friendliness", "use_of_examples",
                      "jargon_handling", "logical_flow", "overall"]
        for key in judge_keys:
            vals = [
                r["judge_scores"][key]
                for r in responses
                if r.get("judge_scores") and key in r["judge_scores"]
            ]
            overall_judge[key] = round(sum(vals) / len(vals), 3) if vals else 0

    # By category breakdown
    by_category = defaultdict(list)
    for r in responses:
        by_category[r["category"]].append(r)

    category_summary = {}
    for category, items in by_category.items():
        category_summary[category] = {}
        # avg flesch reading ease per category
        ease_vals = [i["metrics"]["flesch_reading_ease"] for i in items]
        category_summary[category]["flesch_reading_ease"] = round(
            sum(ease_vals) / len(ease_vals), 2
        )
        if not skip_judge:
            overall_vals = [
                i["judge_scores"]["overall"]
                for i in items
                if i.get("judge_scores")
            ]
            category_summary[category]["judge_overall"] = round(
                sum(overall_vals) / len(overall_vals), 2
            ) if overall_vals else 0

    return {
        "overall_metrics": overall_metrics,
        "overall_judge":overall_judge,
        "by_category":category_summary,
    }


def print_summary(summary: dict, model_name: str):
    """Print a clean summary table to terminal."""
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY — {model_name.upper()}")
    print(f"{'='*50}")

    print("\nReadability metrics:")
    m = summary["overall_metrics"]
    print(f"  Flesch reading ease:  {m.get('flesch_reading_ease')}  (higher = easier, 60-70 is ideal)")
    print(f"  Grade level:          {m.get('flesch_kincaid_grade')}")
    print(f"  Avg word count:       {m.get('word_count')}")
    print(f"  Jargon density:       {m.get('jargon_density')}")
    print(f"  Avg example signals:  {m.get('example_score')}")
    print(f"  Avg analogy signals:  {m.get('analogy_score')}")
    print(f"  BERTScore F1:         {m.get('bertscore_f1')}  (higher = more relevant to prompt)")
    print(f"  Non-English ratio:    {m.get('non_english_ratio')}  (0 = fully English)")
    print(f"  Formatting error rate:{m.get('formatting_error_rate')}  (0 = clean)")

    if summary.get("overall_judge"):
        j = summary["overall_judge"]
        print(f"\nLLM-as-judge scores (1-5):")
        print(f"  Clarity:              {j.get('clarity')}")
        print(f"  Beginner friendly:    {j.get('beginner_friendliness')}")
        print(f"  Use of examples:      {j.get('use_of_examples')}")
        print(f"  Jargon handling:      {j.get('jargon_handling')}")
        print(f"  Logical flow:         {j.get('logical_flow')}")
        print(f"  Overall:              {j.get('overall')}")

    if summary.get("by_category"):
        print(f"\nJudge overall by category:")
        for category, stats in summary["by_category"].items():
            score = stats.get("judge_overall", "n/a")
            print(f"  {category:15s} {score}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Path to responses JSON file (output of generate.py)"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge (no API cost, only textstat + heuristics)"
    )
    args = parser.parse_args()
    evaluate(args.responses, skip_judge=args.no_judge)