"""
agents/debate_agents.py
All five agent roles, each as an async function.
Agents are stateless — they receive context, call the LLM, return text/dict.
"""
from __future__ import annotations
import logging
import time

from truth_through_debate.agents.llm_client import GroqClient, parse_json_response
from truth_through_debate.config import Config
from truth_through_debate.schema import Evidence, Verdict

log = logging.getLogger(__name__)


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _fmt_evidence(evidence: list[Evidence]) -> str:
    if not evidence:
        return "No external evidence retrieved. Use general knowledge carefully."
    lines = []
    for i, ev in enumerate(evidence, 1):
        lines.append(f"[{i}] ({ev.source}) {ev.title}: {ev.text[:350]}")
    return "\n".join(lines)


def _fmt_rounds_history(rounds_history: list[dict]) -> str:
    if not rounds_history:
        return ""
    parts = []
    for rh in rounds_history:
        parts.append(f"=== Round {rh['round']} ===")
        parts.append(f"Debater A (TRUE):\n{rh['argument_a']}")
        parts.append(f"Debater B (FALSE):\n{rh['argument_b']}")
        if rh.get("devils_argument"):
            parts.append(f"Devil's Advocate:\n{rh['devils_argument']}")
    return "\n\n".join(parts)


# ── Debater A (argues TRUE) ───────────────────────────────────────────────────

_SYS_A = """You are an expert fact-checker and debater. Your FIXED position is that the claim is TRUE.
Rules:
- Cite evidence by number e.g. [1], [2]. Only cite what is actually provided.
- Directly rebut your opponent's previous argument when given.
- Be logical, precise, and confident. Do not hedge.
- STRICT limit: {cap} words. Prioritise quality over length."""

async def debater_a(
    claim: str,
    evidence: list[Evidence],
    opponent_arg: str,
    devils_arg: str,
    round_num: int,
    client: GroqClient,
    cfg: Config,
) -> tuple[str, float]:
    t0 = time.monotonic()
    ev = _fmt_evidence(evidence)
    rebuttal = ""
    if opponent_arg:
        rebuttal = f"\nOpponent's argument (claiming FALSE):\n{opponent_arg}"
    if devils_arg:
        rebuttal += f"\nDevil's Advocate challenge:\n{devils_arg}"
    user = (f"CLAIM: {claim}\n\nEVIDENCE:\n{ev}{rebuttal}\n\n"
            f"Round {round_num}: Argue that this claim is TRUE. "
            f"Be concise (≤{cfg.word_cap_per_turn} words).")
    result = await client.complete(cfg.debater_a_model, _SYS_A.format(cap=cfg.word_cap_per_turn),
                                   user, cfg.max_tokens_debate, cfg.temperature_debate)
    return result, time.monotonic() - t0


# ── Debater B (argues FALSE) ──────────────────────────────────────────────────

_SYS_B = """You are an expert fact-checker and debater. Your FIXED position is that the claim is FALSE.
Rules:
- Cite evidence by number e.g. [1], [2]. Only cite what is actually provided.
- Directly rebut your opponent's previous argument when given.
- Be logical, precise, and confident. Do not hedge.
- STRICT limit: {cap} words. Prioritise quality over length."""

async def debater_b(
    claim: str,
    evidence: list[Evidence],
    opponent_arg: str,
    devils_arg: str,
    round_num: int,
    client: GroqClient,
    cfg: Config,
) -> tuple[str, float]:
    t0 = time.monotonic()
    ev = _fmt_evidence(evidence)
    rebuttal = ""
    if opponent_arg:
        rebuttal = f"\nOpponent's argument (claiming TRUE):\n{opponent_arg}"
    if devils_arg:
        rebuttal += f"\nDevil's Advocate challenge:\n{devils_arg}"
    user = (f"CLAIM: {claim}\n\nEVIDENCE:\n{ev}{rebuttal}\n\n"
            f"Round {round_num}: Argue that this claim is FALSE. "
            f"Be concise (≤{cfg.word_cap_per_turn} words).")
    result = await client.complete(cfg.debater_b_model, _SYS_B.format(cap=cfg.word_cap_per_turn),
                                   user, cfg.max_tokens_debate, cfg.temperature_debate)
    return result, time.monotonic() - t0


# ── Devil's Advocate ──────────────────────────────────────────────────────────

_SYS_D = """You are a Devil's Advocate. Your job is to challenge BOTH arguments presented to you.
- Identify the weakest assumption in each argument.
- Point out any evidence that is being misinterpreted or ignored.
- Do NOT state a verdict — just challenge the reasoning.
- Strict limit: 120 words."""

async def devils_advocate(
    claim: str,
    evidence: list[Evidence],
    arg_a: str,
    arg_b: str,
    client: GroqClient,
    cfg: Config,
) -> tuple[str, float]:
    t0 = time.monotonic()
    ev = _fmt_evidence(evidence)
    user = (f"CLAIM: {claim}\n\nEVIDENCE:\n{ev}\n\n"
            f"Argument for TRUE:\n{arg_a}\n\n"
            f"Argument for FALSE:\n{arg_b}\n\n"
            "Challenge both arguments. Identify flaws. Do not give a verdict.")
    result = await client.complete(cfg.devils_model, _SYS_D, user, 256, 0.7)
    return result, time.monotonic() - t0


# ── Judge ─────────────────────────────────────────────────────────────────────

_SYS_JUDGE = """You are a neutral, impartial fact-checking judge.
You will receive a claim, retrieved evidence, and a full debate transcript.

Render a final verdict. Output ONLY valid JSON — no preamble, no markdown fences:
{
  "verdict": "TRUE" | "FALSE" | "NOT ENOUGH INFO",
  "explanation": "<2-4 sentences of reasoning>",
  "confidence": <float 0.0–1.0>
}

Base your verdict on evidence quality, not debater eloquence.
If evidence is absent or contradictory, output NOT ENOUGH INFO with confidence 0.4–0.6."""

async def judge_agent(
    claim: str,
    evidence: list[Evidence],
    rounds_history: list[dict],
    client: GroqClient,
    cfg: Config,
) -> dict:
    ev = _fmt_evidence(evidence)
    transcript = _fmt_rounds_history(rounds_history)
    user = (f"CLAIM: {claim}\n\nEVIDENCE:\n{ev}\n\n"
            f"DEBATE TRANSCRIPT:\n{transcript}\n\nRender your verdict now.")
    raw = await client.complete(cfg.judge_model, _SYS_JUDGE, user,
                                cfg.max_tokens_judge, cfg.temperature_judge)
    try:
        data = parse_json_response(raw)
        verdict = str(data.get("verdict","NOT ENOUGH INFO")).upper()
        if verdict not in {"TRUE","FALSE","NOT ENOUGH INFO"}:
            verdict = "NOT ENOUGH INFO"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        return {"verdict": verdict, "explanation": str(data.get("explanation","")),
                "confidence": confidence}
    except Exception as e:
        log.warning("[Judge] JSON parse error: %s | raw: %s", e, raw[:200])
        return {"verdict": "NOT ENOUGH INFO", "explanation": raw[:300], "confidence": 0.5}


# ── Scorer ────────────────────────────────────────────────────────────────────

_SYS_SCORER = """You are an expert evaluator of argument quality.
Score the debate system's reasoning (NOT the verdict correctness) on a 1–5 rubric:
5 = Excellent: evidence directly cited, no logical gaps, clear rebuttals
4 = Good: mostly evidence-based, minor gaps
3 = Adequate: some evidence, some reliance on general knowledge
2 = Weak: little evidence cited, significant logical gaps
1 = Poor: no evidence cited, speculative or contradictory

Also flag hallucination: true if any factual claim is made without evidence support.

Output ONLY valid JSON:
{"reasoning_score": <int 1-5>, "hallucination_flag": <bool>, "rationale": "<1-2 sentences>"}"""

async def scorer_agent(
    claim: str,
    evidence: list[Evidence],
    rounds_history: list[dict],
    verdict: str,
    explanation: str,
    client: GroqClient,
    cfg: Config,
) -> dict:
    ev = _fmt_evidence(evidence)
    transcript = _fmt_rounds_history(rounds_history)
    user = (f"CLAIM: {claim}\n\nEVIDENCE:\n{ev}\n\n"
            f"DEBATE:\n{transcript}\n\n"
            f"FINAL VERDICT: {verdict}\nEXPLANATION: {explanation}\n\n"
            "Score the reasoning quality and check for hallucination.")
    raw = await client.complete(cfg.scorer_model, _SYS_SCORER, user,
                                cfg.max_tokens_scorer, 0.2)
    try:
        data = parse_json_response(raw)
        score = max(1, min(5, int(data.get("reasoning_score", 3))))
        halluc = bool(data.get("hallucination_flag", False))
        return {"reasoning_score": score, "hallucination_flag": halluc,
                "rationale": str(data.get("rationale",""))}
    except Exception as e:
        log.warning("[Scorer] Parse error: %s", e)
        return {"reasoning_score": 3, "hallucination_flag": False, "rationale": ""}


# ── Baseline (single-pass) ────────────────────────────────────────────────────

_SYS_BASELINE = """You are a fact-checking assistant.
Given a claim and evidence, output ONLY valid JSON — no markdown fences:
{
  "verdict": "TRUE" | "FALSE" | "NOT ENOUGH INFO",
  "explanation": "<2-3 sentences>",
  "confidence": <float 0.0–1.0>,
  "reasoning_score": <int 1-5>
}
Base your verdict on the evidence provided. Self-assess reasoning quality."""

async def baseline_agent(
    claim: str,
    evidence: list[Evidence],
    client: GroqClient,
    cfg: Config,
) -> dict:
    ev = _fmt_evidence(evidence)
    user = f"CLAIM: {claim}\n\nEVIDENCE:\n{ev}\n\nProvide your verdict."
    raw = await client.complete(cfg.baseline_model, _SYS_BASELINE, user, 400, 0.2)
    try:
        data = parse_json_response(raw)
        verdict = str(data.get("verdict","NOT ENOUGH INFO")).upper()
        if verdict not in {"TRUE","FALSE","NOT ENOUGH INFO"}:
            verdict = "NOT ENOUGH INFO"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        score = max(1, min(5, int(data.get("reasoning_score", 3))))
        return {"verdict": verdict, "explanation": str(data.get("explanation","")),
                "confidence": confidence, "reasoning_score": score}
    except Exception as e:
        log.warning("[Baseline] Parse error: %s", e)
        return {"verdict":"NOT ENOUGH INFO","explanation":raw[:200],"confidence":0.5,"reasoning_score":1}
