"""
run_experiment.py

Main entry point for the Truth Through Debate experiment.

Usage examples:
  python run_experiment.py --mode sample --n 10
  python run_experiment.py --mode fever --n 50
  python run_experiment.py --mode sample --system baseline
  python run_experiment.py --mode sample --n 20 --export results.csv
  python run_experiment.py --mode sample --n 5 --rounds 3 --verbose
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from data.sample_claims import load_sample
from evaluation.evaluator import evaluate_system
from utils.logger import console, log_results_table, log_summary
from rich.panel import Panel


def parse_args():
    p = argparse.ArgumentParser(
        description="Truth Through Debate — Multi-Agent Fact Checking"
    )
    p.add_argument(
        "--mode",
        choices=["sample", "fever"],
        default="sample",
        help="Data source: 'sample' (built-in) or 'fever' (HuggingFace FEVER dataset)",
    )
    p.add_argument(
        "--n", type=int, default=10,
        help="Number of claims to evaluate (default: 10)"
    )
    p.add_argument(
        "--system",
        choices=["baseline", "debate", "both"],
        default="both",
        help="Which system(s) to run (default: both)"
    )
    p.add_argument(
        "--rounds", type=int, default=2,
        help="Number of debate rounds (default: 2)"
    )
    p.add_argument(
        "--export", type=str, default=None,
        help="Path to export results CSV (optional)"
    )
    p.add_argument(
        "--export-json", type=str, default=None,
        help="Path to export full results JSON (optional)"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print full debate transcripts"
    )
    p.add_argument(
        "--no-reasoning-score", action="store_true",
        help="Skip LLM-as-judge reasoning quality scoring (faster)"
    )
    p.add_argument(
        "--no-hallucination", action="store_true",
        help="Skip hallucination detection (faster)"
    )
    return p.parse_args()


def load_claims(args) -> list[tuple[str, str]]:
    if args.mode == "sample":
        return load_sample(n=args.n)
    elif args.mode == "fever":
        from data.fever_loader import load_fever
        return load_fever(n=args.n)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def export_csv(
    baseline_results: list[dict],
    debate_results: list[dict],
    path: str,
):
    """Write side-by-side results to a CSV file."""
    # Merge on claim
    debate_map = {r["claim"]: r for r in debate_results}
    baseline_map = {r["claim"]: r for r in baseline_results}
    all_claims = set(debate_map) | set(baseline_map)

    rows = []
    for claim in all_claims:
        b = baseline_map.get(claim, {})
        d = debate_map.get(claim, {})
        rows.append({
            "claim": claim,
            "ground_truth": b.get("ground_truth") or d.get("ground_truth"),
            "baseline_verdict": b.get("verdict", ""),
            "baseline_confidence": b.get("confidence", ""),
            "baseline_correct": b.get("correct", ""),
            "baseline_reasoning_quality": b.get("reasoning_quality", ""),
            "baseline_hallucination": b.get("hallucination_detected", ""),
            "debate_verdict": d.get("verdict", ""),
            "debate_confidence": d.get("confidence", ""),
            "debate_correct": d.get("correct", ""),
            "debate_reasoning_quality": d.get("reasoning_quality", ""),
            "debate_hallucination": d.get("hallucination_detected", ""),
            "debate_explanation": d.get("explanation", ""),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"[green]✅ CSV exported to {path}[/]")


def main():
    args = parse_args()

    console.print(
        Panel(
            "[bold]Truth Through Debate[/]\nMulti-Agent LLM Framework for Fact Checking",
            border_style="blue",
        )
    )

    # Load claims
    claims = load_claims(args)
    if not claims:
        console.print("[red]No claims loaded. Exiting.[/]")
        sys.exit(1)

    console.print(
        f"\n[bold]Running:[/] {args.system} system | "
        f"{len(claims)} claims | {args.rounds} debate round(s)\n"
    )

    # Run evaluation
    output = evaluate_system(
        claims=claims,
        system=args.system,
        num_rounds=args.rounds,
        score_reasoning=not args.no_reasoning_score,
        score_hallucination=not args.no_hallucination,
        verbose=args.verbose,
    )

    baseline_results = output["baseline_results"]
    debate_results = output["debate_results"]
    baseline_metrics = output["baseline_metrics"]
    debate_metrics = output["debate_metrics"]

    # Print per-claim table (if both systems ran)
    if baseline_results and debate_results:
        merged = []
        bmap = {r["claim"]: r for r in baseline_results}
        dmap = {r["claim"]: r for r in debate_results}
        for claim, _ in claims:
            b = bmap.get(claim, {})
            d = dmap.get(claim, {})
            merged.append({
                "claim": claim,
                "ground_truth": b.get("ground_truth") or d.get("ground_truth", "?"),
                "baseline_verdict": b.get("verdict", "?"),
                "baseline_correct": b.get("correct", False),
                "baseline_confidence": b.get("confidence", 0),
                "debate_verdict": d.get("verdict", "?"),
                "debate_correct": d.get("correct", False),
                "debate_confidence": d.get("confidence", 0),
            })
        log_results_table(merged)

    # Print summary
    if baseline_metrics and debate_metrics:
        log_summary(baseline_metrics, debate_metrics)
    elif baseline_metrics:
        console.print("\n[bold]Baseline Metrics:[/]")
        console.print(baseline_metrics)
    elif debate_metrics:
        console.print("\n[bold]Debate Metrics:[/]")
        console.print(debate_metrics)

    # Export
    if args.export and baseline_results and debate_results:
        export_csv(baseline_results, debate_results, args.export)

    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(
                {
                    "baseline_results": baseline_results,
                    "debate_results": debate_results,
                    "baseline_metrics": baseline_metrics,
                    "debate_metrics": debate_metrics,
                },
                f,
                indent=2,
            )
        console.print(f"[green]✅ JSON exported to {args.export_json}[/]")


if __name__ == "__main__":
    main()
