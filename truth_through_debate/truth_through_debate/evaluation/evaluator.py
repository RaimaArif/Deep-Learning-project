"""
evaluation/evaluator.py
Runs baseline or debate system over a list of claims with rate-limit-safe pacing.
"""
import re
import json
import time
from tqdm import tqdm
from debate.engine import run_debate, DebateResult
from evaluation.metrics import (
    reasoning_quality_score,
    hallucination_detected,
    compute_all_metrics,
)
from utils.llm_client import call_llm, BASELINE_MODEL

BASELINE_SYSTEM = """You are a fact-checking assistant.
Given a claim, determine if it is TRUE or FALSE.

Output ONLY valid JSON:
{
  "verdict": "TRUE" or "FALSE",
  "confidence": <float 0.0-1.0>,
  "explanation": "<1-3 sentence explanation>"
}"""


def _normalise_label(label: str) -> str:
    label = label.upper().strip()
    if label in ("SUPPORTS", "TRUE"):
        return "TRUE"
    elif label in ("REFUTES", "FALSE"):
        return "FALSE"
    return "NOT ENOUGH INFO"


def run_baseline(claim: str, ground_truth: str) -> dict:
    """Single-LLM baseline — Llama 3.3 70B, no retrieval, no debate."""
    prompt = f'Is the following claim TRUE or FALSE?\n\nClaim: "{claim}"'
    raw = call_llm(
        prompt,
        system=BASELINE_SYSTEM,
        model=BASELINE_MODEL,
        max_tokens=300,
        temperature=0.3,
    )
    raw_clean = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(raw_clean)
    except json.JSONDecodeError:
        verdict = "TRUE" if "TRUE" in raw.upper() else "FALSE"
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
        data = {
            "verdict": verdict,
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "explanation": raw[:200],
        }

    data["verdict"] = "TRUE" if str(data.get("verdict", "")).upper() == "TRUE" else "FALSE"
    data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    gt_norm = _normalise_label(ground_truth)

    return {
        "claim": claim,
        "ground_truth": gt_norm,
        "verdict": data["verdict"],
        "confidence": data["confidence"],
        "explanation": data.get("explanation", ""),
        "evidence": [],
        "correct": data["verdict"] == gt_norm,
        "system": "baseline",
        "model": BASELINE_MODEL,
    }


def run_debate_system(claim: str, ground_truth: str, num_rounds: int = 2) -> dict:
    """
    Full debate pipeline:
      Retriever  — Llama 4 Scout
      Debater A  — Llama 4 Scout  (pro-true)
      Debater B  — Llama 3.1 8B   (pro-false)
      Judge      — Qwen3 32B
    """
    result: DebateResult = run_debate(
        claim, ground_truth, num_rounds=num_rounds, verbose=False
    )
    return {
        "claim": claim,
        "ground_truth": result.ground_truth,
        "verdict": result.verdict,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "evidence": result.evidence,
        "correct": result.correct,
        "reasoning_quality_self": result.reasoning_quality_self,
        "debate_rounds": result.debate_rounds,
        "system": "debate",
    }


def evaluate_system(
    claims: list[tuple[str, str]],
    system: str = "both",
    num_rounds: int = 2,
    score_reasoning: bool = True,
    score_hallucination: bool = True,
    sleep_between: int = 8,
    verbose: bool = False,
) -> dict:
    """
    Run evaluation over a list of (claim, ground_truth) pairs.

    Args:
        claims:           List of (claim, ground_truth) tuples.
        system:           "baseline", "debate", or "both".
        num_rounds:       Debate rounds per claim.
        score_reasoning:  Run LLM-as-judge reasoning quality scoring.
        score_hallucination: Run hallucination detection.
        sleep_between:    Seconds to sleep between claims (rate limit safety).
        verbose:          Print progress details.
    """
    baseline_results = []
    debate_results = []

    for i, (claim, gt) in enumerate(tqdm(claims, desc="Evaluating", unit="claim")):
        if system in ("baseline", "both"):
            b = run_baseline(claim, gt)
            if score_reasoning:
                b["reasoning_quality"] = reasoning_quality_score(
                    claim, b["evidence"], b["explanation"]
                )
            if score_hallucination:
                b["hallucination_detected"] = hallucination_detected(
                    claim, b["evidence"], b["explanation"]
                )
            baseline_results.append(b)

        if system in ("debate", "both"):
            d = run_debate_system(claim, gt, num_rounds=num_rounds)
            if score_reasoning:
                d["reasoning_quality"] = reasoning_quality_score(
                    claim, d["evidence"], d["explanation"]
                )
            if score_hallucination:
                d["hallucination_detected"] = hallucination_detected(
                    claim, d["evidence"], d["explanation"]
                )
            debate_results.append(d)

        # Pause between claims to stay within free tier rate limits
        if i < len(claims) - 1:
            time.sleep(sleep_between)

    return {
        "baseline_results": baseline_results,
        "debate_results": debate_results,
        "baseline_metrics": compute_all_metrics(baseline_results) if baseline_results else {},
        "debate_metrics": compute_all_metrics(debate_results) if debate_results else {},
    }
