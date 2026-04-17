"""
cli.py  — ttd command-line interface
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ttd",
        description="Truth Through Debate — Multi-Agent Fact-Checker",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ttd debate "claim" [--rounds N] [--no-baseline] [--verbose]
    p_debate = sub.add_parser("debate", help="Evaluate a single claim")
    p_debate.add_argument("claim", type=str)
    p_debate.add_argument("--rounds", type=int, default=2)
    p_debate.add_argument("--no-baseline", action="store_true")
    p_debate.add_argument("--verbose", "-v", action="store_true")
    p_debate.add_argument("--truth", type=str, default=None, help="Ground truth label")

    # ttd eval --source builtin|fever --n 50 --output outputs/
    p_eval = sub.add_parser("eval", help="Batch evaluation")
    p_eval.add_argument("--source", choices=["builtin","fever"], default="builtin")
    p_eval.add_argument("--fever-path", default="data/fever_dev.jsonl")
    p_eval.add_argument("--n", type=int, default=20)
    p_eval.add_argument("--rounds", type=int, default=2)
    p_eval.add_argument("--output", default="outputs")
    p_eval.add_argument("--no-calibration", action="store_true")

    # ttd ui [--port 7860] [--share]
    p_ui = sub.add_parser("ui", help="Launch Gradio web UI")
    p_ui.add_argument("--port", type=int, default=7860)
    p_ui.add_argument("--share", action="store_true")

    # ttd build-index --corpus data/corpus.jsonl
    p_idx = sub.add_parser("build-index", help="Build FAISS index from JSONL corpus")
    p_idx.add_argument("--corpus", required=True)
    p_idx.add_argument("--index-out", default="data/faiss.index")
    p_idx.add_argument("--corpus-out", default="data/faiss_corpus.pkl")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.cmd == "debate":
        from truth_through_debate.config import Config
        from truth_through_debate.pipeline import DebatePipeline
        from rich.console import Console
        from rich.panel import Panel

        async def _run():
            cfg = Config.from_env()
            cfg.num_rounds = args.rounds
            async with DebatePipeline(cfg) as p:
                result = await p.run(args.claim, args.truth,
                                     run_baseline=not args.no_baseline,
                                     verbose=args.verbose)
            c = Console()
            c.print(Panel(f"[bold]Verdict:[/] {result.verdict}\n"
                          f"[bold]Confidence:[/] {result.confidence:.2f}\n"
                          f"[bold]Reasoning:[/] {result.reasoning_score}/5\n"
                          f"[bold]Correct:[/] {result.correct}\n\n"
                          f"{result.explanation}",
                          title="🎯 Truth Through Debate", border_style="cyan"))
            if result.baseline_verdict:
                c.print(f"[dim]Baseline: {result.baseline_verdict} "
                        f"conf={result.baseline_confidence:.2f}[/]")

        asyncio.run(_run())

    elif args.cmd == "eval":
        from truth_through_debate.config import Config
        from truth_through_debate.evaluation.evaluator import run_evaluation
        from truth_through_debate.evaluation.fever_loader import load_builtin, load_fever

        async def _eval():
            cfg = Config.from_env()
            cfg.num_rounds = args.rounds
            claims = (load_builtin(args.n) if args.source == "builtin"
                      else load_fever(args.fever_path, args.n))
            await run_evaluation(claims, cfg, output_dir=args.output,
                                 plot_calibration=not args.no_calibration)

        asyncio.run(_eval())

    elif args.cmd == "ui":
        from truth_through_debate.ui.app import launch
        launch(share=args.share, port=args.port)

    elif args.cmd == "build-index":
        from truth_through_debate.retrieval.hybrid_retriever import build_faiss_index
        build_faiss_index(args.corpus, args.index_out, args.corpus_out)


if __name__ == "__main__":
    main()
