"""
evaluation/evaluator.py
Full batch evaluation: runs debate + baseline on a list of claims,
computes all metrics, applies Platt calibration, saves results.
"""
from __future__ import annotations
import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from truth_through_debate.config import Config
from truth_through_debate.pipeline import DebatePipeline
from truth_through_debate.schema import DebateResult, EvalMetrics
from truth_through_debate.calibration.platt import CalibrationPipeline, compute_ece

log = logging.getLogger(__name__)
console = Console()


def compute_metrics(results: list[DebateResult]) -> EvalMetrics:
    n = len(results)
    if n == 0:
        raise ValueError("Empty results list")

    debate_correct   = [r for r in results if r.correct is True]
    baseline_correct = [r for r in results if r.baseline_correct is True]

    accuracy          = len(debate_correct) / n
    baseline_accuracy = len(baseline_correct) / n

    avg_rs          = sum(r.reasoning_score for r in results) / n
    baseline_avg_rs = sum(r.baseline_reasoning_score for r in results
                          if hasattr(r, "baseline_reasoning_score")) / n \
                      if any(hasattr(r, "baseline_reasoning_score") for r in results) else 0.0

    # ECE using calibrated confidence
    cal_confs  = [r.calibrated_confidence for r in results if r.correct is not None]
    raw_confs  = [r.confidence for r in results if r.correct is not None]
    corrects   = [r.correct for r in results if r.correct is not None]
    b_confs    = [r.baseline_confidence for r in results
                  if r.baseline_confidence is not None and r.baseline_correct is not None]
    b_corrects = [r.baseline_correct for r in results
                  if r.baseline_confidence is not None and r.baseline_correct is not None]

    ece          = compute_ece(cal_confs, corrects) if cal_confs else 0.0
    baseline_ece = compute_ece(b_confs, b_corrects) if b_confs else 0.0

    thr = Config().hallucination_conf_threshold
    halluc_rate = sum(
        1 for r in results if r.correct is False and r.confidence >= thr
    ) / max(n, 1)
    baseline_halluc = sum(
        1 for r in results
        if r.baseline_correct is False and (r.baseline_confidence or 0) >= thr
    ) / max(n, 1)

    avg_conf = sum(r.calibrated_confidence for r in results) / n

    return EvalMetrics(
        n=n, accuracy=accuracy, baseline_accuracy=baseline_accuracy,
        avg_reasoning_score=avg_rs, baseline_avg_reasoning=baseline_avg_rs,
        ece=ece, baseline_ece=baseline_ece,
        hallucination_rate=halluc_rate, baseline_hallucination_rate=baseline_halluc,
        avg_confidence=avg_conf,
    )


def print_summary(metrics: EvalMetrics) -> None:
    table = Table(title="Evaluation Results", show_header=True,
                  header_style="bold cyan", border_style="dim")
    table.add_column("Metric", style="bold", min_width=30)
    table.add_column("Baseline", justify="right", min_width=12)
    table.add_column("Debate System", justify="right", min_width=14, style="green")
    table.add_column("Delta", justify="right", min_width=10)

    for row in metrics.summary_rows():
        metric, base, debate, delta = row
        delta_style = "green" if delta.startswith("+") else ("red" if delta.startswith("-") else None)
        delta_cell = f"[{delta_style}]{delta}[/{delta_style}]" if delta_style else delta
        table.add_row(metric, base, debate, delta_cell)
    console.print(table)


def save_results(results: list[DebateResult], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSONL
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

    # CSV
    if results:
        fieldnames = list(results[0].to_dict().keys())
        with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(r.to_dict() for r in results)

    console.print(f"[green]Results saved → {out}[/]")


async def run_evaluation(
    claims: list[tuple[str, str | None]],
    cfg: Config | None = None,
    output_dir: str = "outputs",
    plot_calibration: bool = True,
    progress_callback: Callable | None = None,
) -> tuple[list[DebateResult], EvalMetrics]:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    claims : list of (claim_text, ground_truth_label)
    cfg    : Config object (defaults to Config.from_env())
    """
    cfg = cfg or Config.from_env()
    results: list[DebateResult] = []

    async with DebatePipeline(cfg) as pipeline:
        with Progress(
            SpinnerColumn(), TextColumn("[bold]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(), console=console, transient=False,
        ) as progress:
            task = progress.add_task("Evaluating claims...", total=len(claims))

            for i, (claim, gt) in enumerate(claims):
                result = await pipeline.run(claim, gt)
                results.append(result)
                progress.advance(task)
                correct_str = "✓" if result.correct else "✗" if result.correct is False else "?"
                console.print(f"[dim]{i+1}/{len(claims)}[/] {correct_str} "
                               f"[bold]{result.verdict}[/] conf={result.confidence:.2f}  "
                               f"rs={result.reasoning_score}/5  {claim[:60]}")
                if progress_callback:
                    progress_callback(i + 1, len(claims), result)
                if i < len(claims) - 1:
                    await asyncio.sleep(cfg.sleep_between_claims)

    # Calibration
    cal = CalibrationPipeline()
    cal.fit(results)
    results = cal.transform(results)
    if plot_calibration:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cal.plot_reliability_diagram(str(out / "reliability_diagram.png"))

    metrics = compute_metrics(results)
    print_summary(metrics)
    save_results(results, output_dir)
    return results, metrics
