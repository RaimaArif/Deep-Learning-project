"""
demo.py

Run a single claim through both systems side-by-side with full verbose output.
Great for debugging and demoing to your professor.

Usage:
    python demo.py
    python demo.py --claim "The Great Wall of China is visible from space."
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.evaluator import run_baseline, run_debate_system
from evaluation.metrics import reasoning_quality_score
from utils.logger import console, log_claim, log_evidence, log_debate_round, log_verdict
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table


DEFAULT_CLAIM = "Napoleon Bonaparte was unusually short for his time."
DEFAULT_GT = "REFUTES"


def main():
    parser = argparse.ArgumentParser(description="Single-claim demo")
    parser.add_argument("--claim", type=str, default=DEFAULT_CLAIM)
    parser.add_argument("--gt", type=str, default=DEFAULT_GT,
                        help="Ground truth: SUPPORTS or REFUTES")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold]Truth Through Debate — Single Claim Demo[/]",
            border_style="blue",
        )
    )

    # ── Baseline ────────────────────────────────────────────────────────
    console.rule("[yellow]BASELINE: Single LLM[/]")
    b = run_baseline(args.claim, args.gt)
    console.print(f"[bold]Claim:[/] {args.claim}")
    console.print(f"[bold]Verdict:[/] {b['verdict']}  |  Confidence: {b['confidence']:.2f}")
    console.print(f"[bold]Explanation:[/] {b['explanation']}")
    b_rq = reasoning_quality_score(args.claim, b["evidence"], b["explanation"])
    console.print(f"[bold]Reasoning Quality:[/] {b_rq}/5")
    correct_str = "[green]✅ CORRECT[/]" if b["correct"] else "[red]❌ WRONG[/]"
    console.print(f"[bold]Result:[/] {correct_str}\n")

    # ── Debate System ────────────────────────────────────────────────────
    console.rule("[green]DEBATE SYSTEM: Multi-Agent[/]")
    log_claim(args.claim)

    d = run_debate_system(args.claim, args.gt, num_rounds=args.rounds)

    log_evidence(d["evidence"])
    for rd in d.get("debate_rounds", []):
        log_debate_round(rd["round"], "Debater A (Pro-True)", rd["argument_a"])
        log_debate_round(rd["round"], "Debater B (Pro-False)", rd["argument_b"])

    log_verdict(d["verdict"], d["confidence"], d["explanation"])
    d_rq = reasoning_quality_score(args.claim, d["evidence"], d["explanation"])
    console.print(f"[bold]Reasoning Quality:[/] {d_rq}/5")
    correct_str = "[green]✅ CORRECT[/]" if d["correct"] else "[red]❌ WRONG[/]"
    console.print(f"[bold]Result:[/] {correct_str}\n")

    # ── Side-by-side comparison ─────────────────────────────────────────
    console.rule("[bold blue]COMPARISON[/]")
    table = Table(show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="center", style="yellow")
    table.add_column("Debate System", justify="center", style="green")

    table.add_row("Verdict", b["verdict"], d["verdict"])
    table.add_row("Ground Truth", args.gt, args.gt)
    table.add_row("Correct?",
                  "✅" if b["correct"] else "❌",
                  "✅" if d["correct"] else "❌")
    table.add_row("Confidence", f"{b['confidence']:.2f}", f"{d['confidence']:.2f}")
    table.add_row("Reasoning Quality", f"{b_rq}/5", f"{d_rq}/5")
    console.print(table)


if __name__ == "__main__":
    main()
