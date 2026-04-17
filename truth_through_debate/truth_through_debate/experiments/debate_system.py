"""
experiments/debate_system.py

Standalone script to run ONLY the multi-agent debate system experiment.

Usage:
    python -m experiments.debate_system --n 10
    python -m experiments.debate_system --n 30 --rounds 3 --verbose
    python -m experiments.debate_system --n 50 --mode fever --export debate_results.csv
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from data.sample_claims import load_sample
from evaluation.evaluator import run_debate_system
from evaluation.metrics import (
    reasoning_quality_score,
    hallucination_detected,
    compute_all_metrics,
)
from utils.logger import console, log_claim, log_evidence, log_debate_round, log_verdict
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Multi-agent debate system experiment")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--mode", choices=["sample", "fever"], default="sample")
    parser.add_argument("--export", type=str, default=None)
    parser.add_argument("--verbose", action="store_true",
                        help="Print full debate transcripts for each claim")
    args = parser.parse_args()

    if args.mode == "sample":
        claims = load_sample(n=args.n)
    else:
        from data.fever_loader import load_fever
        claims = load_fever(n=args.n)

    console.print(
        f"\n[bold green]Running Debate System on {len(claims)} claims "
        f"({args.rounds} rounds each)...[/]\n"
    )

    results = []
    for claim, gt in tqdm(claims, desc="Debate"):
        if args.verbose:
            log_claim(claim)

        r = run_debate_system(claim, gt, num_rounds=args.rounds)

        if args.verbose:
            log_evidence(r["evidence"])
            for rd in r.get("debate_rounds", []):
                log_debate_round(rd["round"], "Debater A (Pro-True)", rd["argument_a"])
                log_debate_round(rd["round"], "Debater B (Pro-False)", rd["argument_b"])
            log_verdict(r["verdict"], r["confidence"], r["explanation"])

        r["reasoning_quality"] = reasoning_quality_score(
            claim, r["evidence"], r["explanation"]
        )
        r["hallucination_detected"] = hallucination_detected(
            claim, r["evidence"], r["explanation"]
        )
        results.append(r)

    metrics = compute_all_metrics(results)
    console.print("\n[bold]Debate System Metrics:[/]")
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.export:
        # Flatten debate_rounds for CSV
        export_results = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "debate_rounds"}
            row["num_rounds"] = len(r.get("debate_rounds", []))
            export_results.append(row)
        df = pd.DataFrame(export_results)
        df.to_csv(args.export, index=False)
        console.print(f"[green]Exported to {args.export}[/]")


if __name__ == "__main__":
    main()
