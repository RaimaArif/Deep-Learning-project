"""
evaluation/metrics.py
Evaluation metrics for the fact-checking system.

Metrics:
1. Accuracy              — % verdicts matching ground truth
2. Reasoning Quality     — 1-5 rubric via LLM-as-judge (uses SCORER_MODEL)
3. ECE                   — Expected Calibration Error (confidence calibration)
4. Hallucination Proxy   — fraction of explanations with unsupported claims
"""
import re
import numpy as np
from utils.llm_client import call_llm, SCORER_MODEL


# ── 1. Accuracy ──────────────────────────────────────────────────────────────

def accuracy(results: list[dict]) -> float:
    valid = [r for r in results if r.get("ground_truth") in ("TRUE", "FALSE")]
    if not valid:
        return 0.0
    correct = sum(1 for r in valid if r.get("verdict") == r.get("ground_truth"))
    return correct / len(valid)


# ── 2. Reasoning Quality ─────────────────────────────────────────────────────

REASONING_JUDGE_SYSTEM = """You are an expert evaluator of logical reasoning quality.
Score the following fact-checking explanation on a scale of 1-5.

Rubric:
5 = Exemplary: strong evidence citations, airtight logic, no unsupported leaps
4 = Good: mostly evidence-backed, minor logical gaps
3 = Adequate: some evidence use, noticeable reasoning gaps
2 = Weak: minimal evidence, significant unsupported claims
1 = Poor: no evidence, pure assertion or hallucination

Respond with ONLY a single integer (1, 2, 3, 4, or 5). Nothing else."""


def reasoning_quality_score(
    claim: str,
    evidence: list[str],
    explanation: str,
) -> int:
    """Score reasoning quality 1-5 using SCORER_MODEL (Kimi K2)."""
    evidence_str = "\n".join(f"- {s}" for s in evidence)
    prompt = (
        f"Claim: {claim}\n\n"
        f"Available evidence:\n{evidence_str}\n\n"
        f"Explanation to evaluate:\n{explanation}\n\n"
        "Score (1-5):"
    )
    raw = call_llm(
        prompt,
        system=REASONING_JUDGE_SYSTEM,
        model=SCORER_MODEL,
        max_tokens=5,
        temperature=0.0,
    )
    match = re.search(r"[1-5]", raw)
    return int(match.group()) if match else 3


def avg_reasoning_quality(results: list[dict]) -> float:
    scores = [r.get("reasoning_quality", 3) for r in results]
    return float(np.mean(scores)) if scores else 0.0


# ── 3. Expected Calibration Error ────────────────────────────────────────────

def expected_calibration_error(results: list[dict], n_bins: int = 10) -> float:
    """Lower ECE = better calibration between confidence and accuracy."""
    valid = [r for r in results if r.get("ground_truth") in ("TRUE", "FALSE")]
    if not valid:
        return 0.0

    confidences = np.array([r["confidence"] for r in valid])
    correct = np.array([1 if r.get("correct") else 0 for r in valid])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(valid)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


# ── 4. Hallucination Proxy ───────────────────────────────────────────────────

HALLUCINATION_SYSTEM = """You are a fact-checking auditor.
Given a claim, evidence snippets, and an explanation, determine if the explanation
makes any specific factual assertions that are NOT supported by the provided evidence.

Respond with ONLY "YES" (unsupported claims present) or "NO" (all claims grounded)."""


def hallucination_detected(
    claim: str,
    evidence: list[str],
    explanation: str,
) -> bool:
    """Return True if explanation contains unsupported factual claims."""
    evidence_str = "\n".join(f"- {s}" for s in evidence)
    prompt = (
        f"Claim: {claim}\n\n"
        f"Evidence:\n{evidence_str}\n\n"
        f"Explanation:\n{explanation}\n\n"
        "Does the explanation contain unsupported factual assertions? (YES/NO):"
    )
    raw = call_llm(
        prompt,
        system=HALLUCINATION_SYSTEM,
        model=SCORER_MODEL,
        max_tokens=5,
        temperature=0.0,
    )
    return "YES" in raw.upper()


def hallucination_rate(results: list[dict]) -> float:
    flagged = sum(1 for r in results if r.get("hallucination_detected", False))
    return flagged / len(results) if results else 0.0


# ── Aggregate ─────────────────────────────────────────────────────────────────

def compute_all_metrics(results: list[dict]) -> dict:
    return {
        "accuracy": accuracy(results),
        "avg_reasoning_quality": avg_reasoning_quality(results),
        "ece": expected_calibration_error(results),
        "avg_confidence": float(np.mean([r["confidence"] for r in results])) if results else 0.0,
        "hallucination_rate": hallucination_rate(results),
        "n": len(results),
    }
