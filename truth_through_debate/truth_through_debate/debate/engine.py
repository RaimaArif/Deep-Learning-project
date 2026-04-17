"""
debate/engine.py

Debate Engine — orchestrates the full pipeline:
  retrieve → debate rounds → judge → verdict

Returns a structured DebateResult dataclass for downstream evaluation.
"""
from dataclasses import dataclass, field
from agents.retriever import retrieve
from agents.debater import argue
from agents.judge import judge
from utils.logger import log_claim, log_evidence, log_debate_round, log_verdict


@dataclass
class DebateResult:
    claim: str
    ground_truth: str                   # "TRUE" / "FALSE" / "NOT ENOUGH INFO"
    evidence: list[str]
    debate_rounds: list[dict]           # [{round, argument_a, argument_b}]
    verdict: str                        # "TRUE" / "FALSE"
    confidence: float
    explanation: str
    reasoning_quality_self: int        # 1–5, self-rated by judge
    correct: bool = False


def run_debate(
    claim: str,
    ground_truth: str,
    num_rounds: int = 2,
    verbose: bool = True,
) -> DebateResult:
    """
    Run the full multi-agent debate pipeline for a single claim.

    Args:
        claim:        The factual claim to evaluate.
        ground_truth: Label from dataset ("SUPPORTS"/"REFUTES"/"NOT ENOUGH INFO").
        num_rounds:   Number of debate rounds (2–3 recommended).
        verbose:      Print rich-formatted output.

    Returns:
        DebateResult with all intermediate and final outputs.
    """
    # Normalise FEVER labels to TRUE/FALSE
    gt_normalised = _normalise_label(ground_truth)

    if verbose:
        log_claim(claim)

    # Step 1: Retrieve evidence
    evidence = retrieve(claim)
    if verbose:
        log_evidence(evidence)

    # Step 2: Debate rounds
    rounds = []
    prev_a = ""
    prev_b = ""

    for r in range(1, num_rounds + 1):
        arg_a = argue("A", claim, evidence, opponent_argument=prev_b, round_num=r)
        arg_b = argue("B", claim, evidence, opponent_argument=prev_a, round_num=r)

        if verbose:
            log_debate_round(r, "Debater A (Pro-True)", arg_a)
            log_debate_round(r, "Debater B (Pro-False)", arg_b)

        rounds.append({"round": r, "argument_a": arg_a, "argument_b": arg_b})
        prev_a = arg_a
        prev_b = arg_b

    # Step 3: Judge
    verdict_data = judge(claim, evidence, rounds)

    if verbose:
        log_verdict(
            verdict_data["verdict"],
            verdict_data["confidence"],
            verdict_data["explanation"],
        )

    correct = (verdict_data["verdict"] == gt_normalised)

    return DebateResult(
        claim=claim,
        ground_truth=gt_normalised,
        evidence=evidence,
        debate_rounds=rounds,
        verdict=verdict_data["verdict"],
        confidence=verdict_data["confidence"],
        explanation=verdict_data["explanation"],
        reasoning_quality_self=verdict_data["reasoning_quality_self"],
        correct=correct,
    )


def _normalise_label(label: str) -> str:
    """Map FEVER labels to TRUE/FALSE."""
    label = label.upper().strip()
    if label in ("SUPPORTS", "TRUE"):
        return "TRUE"
    elif label in ("REFUTES", "FALSE"):
        return "FALSE"
    else:
        return "NOT ENOUGH INFO"
