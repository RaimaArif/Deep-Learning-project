"""
ui/app.py
Gradio web UI — single-claim live demo + batch evaluation tab.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os

log = logging.getLogger(__name__)

_THEME_CSS = """
.verdict-true  { color: #2DC653 !important; font-weight: 600; }
.verdict-false { color: #E63946 !important; font-weight: 600; }
.verdict-nei   { color: #F4A261 !important; font-weight: 600; }
.evidence-box  { font-size: 0.85em; background: var(--background-fill-secondary);
                 border-radius: 8px; padding: 10px; max-height: 250px; overflow-y: auto; }
.round-box     { border-left: 3px solid #7F77DD; padding-left: 10px; margin-bottom: 8px; }
"""


def _fmt_evidence(evidence: list) -> str:
    if not evidence:
        return "No evidence retrieved."
    lines = []
    for i, ev in enumerate(evidence, 1):
        lines.append(f"**[{i}]** `{ev.source}` — {ev.title}\n{ev.text[:300]}")
    return "\n\n---\n".join(lines)


def _fmt_rounds(rounds: list) -> str:
    parts = []
    for rd in rounds:
        parts.append(f"### Round {rd.round_num}\n\n"
                     f"**🟢 Debater A (TRUE):**\n{rd.argument_a}\n\n"
                     f"**🔴 Debater B (FALSE):**\n{rd.argument_b}\n\n"
                     f"**👿 Devil's Advocate:**\n{rd.devils_argument}")
    return "\n\n---\n\n".join(parts)


def build_app():
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("gradio not installed. pip install gradio")

    from truth_through_debate.config import Config
    from truth_through_debate.pipeline import DebatePipeline
    from truth_through_debate.evaluation.fever_loader import load_builtin, load_fever
    from truth_through_debate.evaluation.evaluator import run_evaluation, compute_metrics, print_summary

    async def run_single(claim: str, rounds: int, use_baseline: bool) -> tuple:
        """Run pipeline and return UI outputs."""
        if not claim.strip():
            return ("—", 0.0, 0.0, "No claim provided.", "—", "—")
        try:
            cfg = Config.from_env()
            cfg.num_rounds = rounds
            async with DebatePipeline(cfg) as p:
                result = await p.run(claim, run_baseline=use_baseline)

            ev_text = _fmt_evidence(result.evidence)
            debate_text = _fmt_rounds(result.rounds)

            verdict_display = f"**{result.verdict}** (raw conf: {result.confidence:.2f})"
            baseline_display = (
                f"{result.baseline_verdict} (conf: {result.baseline_confidence:.2f})"
                if result.baseline_verdict else "—"
            )
            return (
                verdict_display,
                result.confidence,
                result.reasoning_score / 5,
                result.explanation,
                ev_text,
                debate_text,
                baseline_display,
            )
        except Exception as e:
            log.exception(e)
            return (f"Error: {e}", 0.0, 0.0, "", "", "", "—")

    def run_single_sync(claim, rounds, use_baseline):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_single(claim, rounds, use_baseline))
        finally:
            loop.close()

    async def run_batch_fn(source: str, fever_path: str, n: int, rounds: int) -> str:
        if source == "Built-in 50 claims":
            claims = load_builtin(n=int(n))
        else:
            claims = load_fever(fever_path, n=int(n))
        cfg = Config.from_env()
        cfg.num_rounds = rounds
        results, metrics = await run_evaluation(claims, cfg, output_dir="outputs/batch",
                                                plot_calibration=True)
        rows = metrics.summary_rows()
        lines = ["| Metric | Baseline | Debate | Delta |",
                 "|--------|----------|--------|-------|"]
        for m, b, d, delta in rows:
            lines.append(f"| {m} | {b} | {d} | {delta} |")
        return "\n".join(lines)

    def run_batch_sync(source, fever_path, n, rounds):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_batch_fn(source, fever_path, n, rounds))
        finally:
            loop.close()

    with gr.Blocks(title="Truth Through Debate v2", css=_THEME_CSS) as demo:
        gr.Markdown("# 🎯 Truth Through Debate v2\n"
                    "Multi-agent LLM debate framework for fact-checking. "
                    "Three debaters + judge + scorer + Platt calibration.")

        with gr.Tabs():
            # ── Tab 1: Single claim ──────────────────────────────────────────
            with gr.Tab("🔍 Single Claim"):
                with gr.Row():
                    with gr.Column(scale=3):
                        claim_input = gr.Textbox(
                            label="Claim to evaluate",
                            placeholder="e.g. The Great Wall of China is visible from space.",
                            lines=2,
                        )
                        with gr.Row():
                            rounds_slider = gr.Slider(1, 3, value=2, step=1,
                                                      label="Debate rounds")
                            baseline_check = gr.Checkbox(value=True, label="Run baseline")
                        run_btn = gr.Button("▶ Run Debate", variant="primary")

                    with gr.Column(scale=1):
                        verdict_out = gr.Markdown("**Verdict:** —")
                        conf_out    = gr.Label(label="Debate confidence", num_top_classes=0)
                        rs_out      = gr.Label(label="Reasoning score (0–1)", num_top_classes=0)
                        baseline_out = gr.Textbox(label="Baseline", interactive=False)

                explanation_out = gr.Textbox(label="Judge explanation", lines=3, interactive=False)

                with gr.Accordion("📋 Evidence retrieved", open=False):
                    evidence_out = gr.Markdown()

                with gr.Accordion("💬 Debate transcript", open=False):
                    debate_out = gr.Markdown()

                run_btn.click(
                    fn=run_single_sync,
                    inputs=[claim_input, rounds_slider, baseline_check],
                    outputs=[verdict_out, conf_out, rs_out, explanation_out,
                             evidence_out, debate_out, baseline_out],
                )

            # ── Tab 2: Batch evaluation ──────────────────────────────────────
            with gr.Tab("📊 Batch Evaluation"):
                gr.Markdown("Run a full evaluation on multiple claims and get metrics.")
                with gr.Row():
                    source_dd   = gr.Dropdown(["Built-in 50 claims", "FEVER file"],
                                              value="Built-in 50 claims", label="Data source")
                    fever_path  = gr.Textbox(label="FEVER .jsonl path", value="data/fever_dev.jsonl")
                    n_claims    = gr.Slider(5, 200, value=20, step=5, label="Number of claims")
                    rounds_b    = gr.Slider(1, 3, value=2, step=1, label="Debate rounds")
                run_batch_btn = gr.Button("▶ Run Batch Evaluation", variant="primary")
                batch_results = gr.Markdown(label="Results")

                run_batch_btn.click(
                    fn=run_batch_sync,
                    inputs=[source_dd, fever_path, n_claims, rounds_b],
                    outputs=[batch_results],
                )

            # ── Tab 3: Config ─────────────────────────────────────────────────
            with gr.Tab("⚙ Config"):
                gr.Markdown("""
### Model Assignments (edit via environment variables or Config)

| Role | Model |
|------|-------|
| Baseline | `llama-3.3-70b-versatile` |
| Debater A (TRUE) | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Debater B (FALSE) | `llama-3.1-8b-instant` |
| Devil's Advocate | `qwen/qwen3-32b` |
| Judge | `qwen/qwen3-32b` |
| Scorer | `moonshotai/kimi-k2-instruct` |

All models are **free on [Groq](https://console.groq.com)**. 
Set `GROQ_API_KEY` and optionally `TAVILY_API_KEY` in your environment.
""")

    return demo


def launch(share: bool = False, port: int = 7860) -> None:
    app = build_app()
    app.launch(share=share, server_port=port, show_error=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    launch()
