"""
pipeline.py
Async debate pipeline — orchestrates retrieval + debate rounds + judge + scorer.
Debater A and Debater B run concurrently within each round.
"""
from __future__ import annotations
import asyncio
import logging
import time

from truth_through_debate.config import Config
from truth_through_debate.schema import DebateResult, DebateRound, Evidence
from truth_through_debate.retrieval.hybrid_retriever import HybridRetriever
from truth_through_debate.agents.llm_client import GroqClient
from truth_through_debate.agents.debate_agents import (
    debater_a, debater_b, devils_advocate,
    judge_agent, scorer_agent, baseline_agent,
)

log = logging.getLogger(__name__)


class DebatePipeline:
    """
    Main entry point.

    Usage
    -----
    async with DebatePipeline(cfg) as p:
        result = await p.run("The Eiffel Tower is in Berlin.", ground_truth="FALSE")
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config.from_env()
        self.retriever = HybridRetriever(self.cfg)

    async def __aenter__(self) -> "DebatePipeline":
        self._client = GroqClient(self.cfg)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        await self._client.__aexit__(*args)

    async def run(
        self,
        claim: str,
        ground_truth: str | None = None,
        run_baseline: bool = True,
        verbose: bool = False,
    ) -> DebateResult:
        t0 = time.monotonic()

        # ── 1. Retrieve evidence (all backends in parallel) ───────────────────
        if verbose:
            log.info("[Pipeline] Retrieving evidence for: %r", claim)
        evidence: list[Evidence] = await self.retriever.retrieve(claim)
        if verbose:
            log.info("[Pipeline] Got %d evidence snippets.", len(evidence))

        # ── 2. Baseline (runs concurrently with first debate round) ───────────
        baseline_task = None
        if run_baseline:
            baseline_task = asyncio.create_task(
                baseline_agent(claim, evidence, self._client, self.cfg)
            )

        # ── 3. Debate rounds ──────────────────────────────────────────────────
        rounds_history: list[dict] = []
        debate_rounds: list[DebateRound] = []
        prev_a = prev_b = prev_d = ""

        for r in range(1, self.cfg.num_rounds + 1):
            if verbose:
                log.info("[Pipeline] Round %d/%d ...", r, self.cfg.num_rounds)

            # Debater A and B run in parallel
            (arg_a, t_a), (arg_b, t_b) = await asyncio.gather(
                debater_a(claim, evidence, prev_b, prev_d, r, self._client, self.cfg),
                debater_b(claim, evidence, prev_a, prev_d, r, self._client, self.cfg),
            )

            # Devil's Advocate runs after both debaters finish
            arg_d, t_d = await devils_advocate(claim, evidence, arg_a, arg_b,
                                               self._client, self.cfg)

            round_dict = {"round": r, "argument_a": arg_a, "argument_b": arg_b,
                          "devils_argument": arg_d}
            rounds_history.append(round_dict)
            debate_rounds.append(DebateRound(
                round_num=r, argument_a=arg_a, argument_b=arg_b,
                devils_argument=arg_d, elapsed_a=t_a, elapsed_b=t_b, elapsed_d=t_d,
            ))
            prev_a, prev_b, prev_d = arg_a, arg_b, arg_d

        # ── 4. Judge + Scorer (parallel) ──────────────────────────────────────
        judge_coro  = judge_agent(claim, evidence, rounds_history, self._client, self.cfg)
        # Scorer needs judge verdict, so run judge first if needed — but we can
        # prefetch by running scorer on intermediate state; simpler: sequential.
        judge_data  = await judge_coro
        scorer_data = await scorer_agent(
            claim, evidence, rounds_history,
            judge_data["verdict"], judge_data["explanation"],
            self._client, self.cfg,
        )

        # ── 5. Baseline result ────────────────────────────────────────────────
        baseline_data = None
        if baseline_task:
            baseline_data = await baseline_task

        # ── 6. Assemble result ────────────────────────────────────────────────
        verdict: str = judge_data["verdict"]
        correct = None
        if ground_truth is not None:
            gt = ground_truth.upper().replace("SUPPORTS","TRUE").replace("REFUTES","FALSE")
            correct = (verdict == gt)

        b_correct = None
        if baseline_data and ground_truth:
            gt = ground_truth.upper().replace("SUPPORTS","TRUE").replace("REFUTES","FALSE")
            b_correct = (baseline_data["verdict"] == gt)

        result = DebateResult(
            claim=claim,
            ground_truth=ground_truth,
            evidence=evidence,
            rounds=debate_rounds,
            verdict=verdict,
            explanation=judge_data["explanation"],
            confidence=judge_data["confidence"],
            calibrated_confidence=judge_data["confidence"],  # updated by calibrator
            reasoning_score=scorer_data["reasoning_score"],
            hallucination_flag=scorer_data["hallucination_flag"],
            correct=correct,
            elapsed_total=round(time.monotonic() - t0, 2),
            baseline_verdict=baseline_data["verdict"] if baseline_data else None,
            baseline_confidence=baseline_data["confidence"] if baseline_data else None,
            baseline_correct=b_correct,
            models_used={
                "debater_a": self.cfg.debater_a_model,
                "debater_b": self.cfg.debater_b_model,
                "devils":    self.cfg.devils_model,
                "judge":     self.cfg.judge_model,
                "scorer":    self.cfg.scorer_model,
                "baseline":  self.cfg.baseline_model,
            },
        )

        if verbose:
            log.info("[Pipeline] %s  conf=%.2f  correct=%s  elapsed=%.1fs",
                     verdict, result.confidence, correct, result.elapsed_total)
        return result

    async def run_batch(
        self,
        claims: list[tuple[str, str | None]],  # (claim, ground_truth)
        sleep_between: float | None = None,
        progress_callback=None,
    ) -> list[DebateResult]:
        """
        Run multiple claims sequentially (Groq free tier requires this).
        sleep_between: seconds to pause between claims (default: cfg.sleep_between_claims)
        progress_callback: callable(done, total, result) — for UI progress bars
        """
        gap = sleep_between if sleep_between is not None else self.cfg.sleep_between_claims
        results = []
        for i, (claim, gt) in enumerate(claims):
            r = await self.run(claim, gt)
            results.append(r)
            if progress_callback:
                progress_callback(i + 1, len(claims), r)
            if i < len(claims) - 1 and gap > 0:
                await asyncio.sleep(gap)
        return results
