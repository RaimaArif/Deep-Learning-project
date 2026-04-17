"""
utils/logger.py
Structured logging using Rich for pretty terminal output.
"""
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def log_claim(claim: str):
    console.print(Panel(f"[bold cyan]CLAIM:[/] {claim}", border_style="cyan"))


def log_evidence(snippets: list[str]):
    console.print("[bold yellow]📄 Evidence Retrieved:[/]")
    for i, s in enumerate(snippets, 1):
        console.print(f"  [{i}] {s}")


def log_debate_round(round_num: int, agent: str, argument: str):
    color = "green" if "A" in agent else "red"
    console.print(
        Panel(
            argument,
            title=f"[bold {color}]Round {round_num} — {agent}[/]",
            border_style=color,
        )
    )


def log_verdict(verdict: str, confidence: float, explanation: str):
    color = "green" if verdict == "TRUE" else "red"
    console.print(
        Panel(
            f"[bold]Verdict:[/] [{color}]{verdict}[/]\n"
            f"[bold]Confidence:[/] {confidence:.2f}\n\n"
            f"[bold]Explanation:[/] {explanation}",
            title="⚖️  Judge Decision",
            border_style="magenta",
        )
    )


def log_results_table(results: list[dict]):
    from rich.table import Table

    table = Table(title="Experiment Results", show_lines=True)
    table.add_column("Claim", style="cyan", max_width=40)
    table.add_column("Ground Truth", justify="center")
    table.add_column("Baseline", justify="center")
    table.add_column("Debate", justify="center")
    table.add_column("B-Conf", justify="center")
    table.add_column("D-Conf", justify="center")

    for r in results:
        gt = r.get("ground_truth", "?")
        b_correct = "✅" if r.get("baseline_correct") else "❌"
        d_correct = "✅" if r.get("debate_correct") else "❌"
        table.add_row(
            r["claim"][:40],
            gt,
            f"{r.get('baseline_verdict','?')} {b_correct}",
            f"{r.get('debate_verdict','?')} {d_correct}",
            f"{r.get('baseline_confidence', 0):.2f}",
            f"{r.get('debate_confidence', 0):.2f}",
        )

    console.print(table)


def log_summary(baseline_metrics: dict, debate_metrics: dict):
    from rich.table import Table

    table = Table(title="📊 Summary Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="center", style="yellow")
    table.add_column("Debate System", justify="center", style="green")
    table.add_column("Δ", justify="center")

    metrics = [
        ("Accuracy", "accuracy", ".3f"),
        ("Avg Reasoning Quality", "avg_reasoning_quality", ".2f"),
        ("ECE (Calibration ↓)", "ece", ".4f"),
        ("Avg Confidence", "avg_confidence", ".3f"),
    ]

    for label, key, fmt in metrics:
        b_val = baseline_metrics.get(key, 0)
        d_val = debate_metrics.get(key, 0)
        delta = d_val - b_val
        sign = "+" if delta >= 0 else ""
        # For ECE, lower is better — flip color
        if key == "ece":
            delta_str = f"[{'green' if delta < 0 else 'red'}]{sign}{delta:{fmt}}[/]"
        else:
            delta_str = f"[{'green' if delta > 0 else 'red'}]{sign}{delta:{fmt}}[/]"
        table.add_row(label, f"{b_val:{fmt}}", f"{d_val:{fmt}}", delta_str)

    console.print(table)
