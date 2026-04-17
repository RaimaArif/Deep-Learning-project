"""
experiments/baseline.py

Standalone script to run ONLY the single-LLM baseline experiment.

Usage:
    python -m experiments.baseline --n 20
    python -m experiments.baseline --n 50 --mode fever --export baseline_results.csv
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from data.sample_claims import load_sample
from evaluation.evaluator import run_baseline
from evaluation.metrics import (
    reasoning_quality_score,
    hallucination_detected,
    compute_all_metrics,
)
from utils.logger import console
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Baseline single-LLM experiment")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mode", choices=["sample", "fever"], default="sample")
    parser.add_argument("--export", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "sample":
        claims = load_sample(n=args.n)
    else:
        from data.fever_loader import load_fever
        claims = load_fever(n=args.n)

    console.print(f"\n[bold yellow]Running Baseline on {len(claims)} claims...[/]\n")

    results = []
    for claim, gt in tqdm(claims, desc="Baseline"):
        r = run_baseline(claim, gt)
        r["reasoning_quality"] = reasoning_quality_score(
            claim, r["evidence"], r["explanation"]
        )
        r["hallucination_detected"] = hallucination_detected(
            claim, r["evidence"], r["explanation"]
        )
        results.append(r)

    metrics = compute_all_metrics(results)
    console.print("\n[bold]Baseline Metrics:[/]")
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.export:
        df = pd.DataFrame(results)
        df.to_csv(args.export, index=False)
        console.print(f"[green]Exported to {args.export}[/]")


if __name__ == "__main__":
    main()
