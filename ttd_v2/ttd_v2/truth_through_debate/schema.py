"""
schema.py
Shared dataclasses / typed dicts for the entire pipeline.
All data flowing between agents uses these types.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


Verdict = Literal["TRUE", "FALSE", "NOT ENOUGH INFO"]
Source  = Literal["wikipedia", "tavily", "bm25_fever", "faiss"]


@dataclass
class Evidence:
    text: str
    source: Source
    score: float = 1.0
    url: str = ""
    title: str = ""

    def __str__(self) -> str:
        return f"[{self.source}] {self.title}: {self.text[:200]}"


@dataclass
class DebateRound:
    round_num: int
    argument_a: str          # Debater A (TRUE)
    argument_b: str          # Debater B (FALSE)
    devils_argument: str     # Devil's Advocate
    elapsed_a: float = 0.0
    elapsed_b: float = 0.0
    elapsed_d: float = 0.0


@dataclass
class DebateResult:
    claim: str
    ground_truth: str | None
    evidence: list[Evidence]
    rounds: list[DebateRound]
    verdict: Verdict
    explanation: str
    confidence: float           # raw judge confidence 0–1
    calibrated_confidence: float = 0.0  # after Platt scaling
    reasoning_score: int = 0    # 1–5 from Scorer agent
    hallucination_flag: bool = False
    correct: bool | None = None
    elapsed_total: float = 0.0
    baseline_verdict: Verdict | None = None
    baseline_confidence: float | None = None
    baseline_correct: bool | None = None
    models_used: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "ground_truth": self.ground_truth,
            "verdict": self.verdict,
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "confidence": round(self.confidence, 4),
            "reasoning_score": self.reasoning_score,
            "hallucination_flag": self.hallucination_flag,
            "correct": self.correct,
            "explanation": self.explanation,
            "elapsed_total": round(self.elapsed_total, 2),
            "num_evidence": len(self.evidence),
            "num_rounds": len(self.rounds),
            "baseline_verdict": self.baseline_verdict,
            "baseline_confidence": self.baseline_confidence,
            "baseline_correct": self.baseline_correct,
        }


@dataclass
class EvalMetrics:
    n: int
    accuracy: float
    baseline_accuracy: float
    avg_reasoning_score: float
    baseline_avg_reasoning: float
    ece: float                      # Expected Calibration Error (debate)
    baseline_ece: float
    hallucination_rate: float
    baseline_hallucination_rate: float
    avg_confidence: float
    platt_slope: float = 0.0
    platt_intercept: float = 0.0

    def delta(self, field: str) -> float:
        return getattr(self, field) - getattr(self, f"baseline_{field}")

    def summary_rows(self) -> list[tuple[str, str, str, str]]:
        """Returns (metric, baseline, debate, delta) tuples for display."""
        rows = [
            ("Accuracy",            f"{self.baseline_accuracy:.3f}",       f"{self.accuracy:.3f}",            f"{self.accuracy - self.baseline_accuracy:+.3f}"),
            ("Avg reasoning (1–5)", f"{self.baseline_avg_reasoning:.2f}",  f"{self.avg_reasoning_score:.2f}", f"{self.avg_reasoning_score - self.baseline_avg_reasoning:+.2f}"),
            ("ECE ↓",               f"{self.baseline_ece:.4f}",            f"{self.ece:.4f}",                 f"{self.ece - self.baseline_ece:+.4f}"),
            ("Hallucination rate ↓",f"{self.baseline_hallucination_rate:.3f}", f"{self.hallucination_rate:.3f}", f"{self.hallucination_rate - self.baseline_hallucination_rate:+.3f}"),
            ("Avg confidence",      f"{self.baseline_accuracy:.3f}",       f"{self.avg_confidence:.3f}",      "—"),
        ]
        return rows
