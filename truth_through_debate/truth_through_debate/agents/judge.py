"""
agents/judge.py
Judge Agent — reads the full debate transcript and renders a final verdict.
Model: Qwen3 32B (JUDGE_MODEL) — Alibaba, different family from all debaters.

Output schema (JSON):
{
  "verdict":                "TRUE" | "FALSE",
  "confidence":             float in [0, 1],
  "explanation":            str,
  "reasoning_quality_self": int (1-5)
}
"""
import json
import re
from utils.llm_client import call_llm, JUDGE_MODEL

JUDGE_SYSTEM = """You are a neutral, impartial fact-checking judge.

You will receive:
1. A claim to evaluate.
2. Evidence snippets retrieved from a knowledge base.
3. A multi-round debate transcript between two debaters.

Your task:
- Determine whether the claim is TRUE or FALSE.
- Base your verdict primarily on the EVIDENCE, not on debater rhetoric.
- Identify which arguments were logically sound and evidence-backed.
- Assign a confidence score between 0.0 (completely uncertain) and 1.0 (certain).

Output ONLY valid JSON matching this schema exactly:
{
  "verdict": "TRUE" or "FALSE",
  "confidence": <float 0.0-1.0>,
  "explanation": "<2-4 sentence explanation citing evidence>",
  "reasoning_quality_self": <integer 1-5>
}

Where reasoning_quality_self rates your own reasoning quality:
5 = Strong evidence, clear logic, high certainty
4 = Good evidence, minor gaps
3 = Mixed evidence, moderate certainty
2 = Weak evidence, significant uncertainty
1 = Insufficient evidence, guessing

Do not output anything outside the JSON object."""


def _format_transcript(rounds: list[dict]) -> str:
    lines = []
    for r in rounds:
        lines.append(f"--- Round {r['round']} ---")
        lines.append(f"Debater A (Pro-True  / Llama 4 Scout): {r['argument_a']}")
        lines.append(f"Debater B (Pro-False / Llama 3.1 8B):  {r['argument_b']}")
    return "\n\n".join(lines)


def judge(
    claim: str,
    evidence: list[str],
    debate_rounds: list[dict],
) -> dict:
    """Produce a final verdict after reading the full debate."""
    evidence_str = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(evidence))
    transcript = _format_transcript(debate_rounds)

    prompt = (
        f"Claim: {claim}\n\n"
        f"Evidence:\n{evidence_str}\n\n"
        f"Debate Transcript:\n{transcript}\n\n"
        "Render your final verdict as JSON."
    )

    raw = call_llm(
        prompt,
        system=JUDGE_SYSTEM,
        model=JUDGE_MODEL,
        max_tokens=400,
        temperature=0.1,
    )
    return _parse_verdict(raw)


def _parse_verdict(raw: str) -> dict:
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        verdict = "TRUE" if "TRUE" in raw.upper() else "FALSE"
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
        exp_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', raw)
        data = {
            "verdict": verdict,
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "explanation": exp_match.group(1) if exp_match else raw[:200],
            "reasoning_quality_self": 2,
        }

    data["verdict"] = "TRUE" if str(data.get("verdict", "")).upper() == "TRUE" else "FALSE"
    data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    data["reasoning_quality_self"] = int(data.get("reasoning_quality_self", 3))
    return data
