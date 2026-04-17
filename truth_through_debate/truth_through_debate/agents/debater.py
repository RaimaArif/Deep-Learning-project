"""
agents/debater.py
Two debaters with opposing stances, each using a different model.

  Debater A (pro-true)  — Llama 4 Scout 17B (meta-llama/llama-4-scout-17b-16e-instruct)
  Debater B (pro-false) — Llama 3.1 8B      (llama-3.1-8b-instant)

Different model sizes create genuine diversity — a larger model arguing TRUE
vs a smaller model arguing FALSE tests whether debate quality depends on model size.
"""
from utils.llm_client import call_llm, DEBATER_A_MODEL, DEBATER_B_MODEL

DEBATER_A_SYSTEM = """You are Debater A, an expert fact-checker.
Your fixed position: the claim is TRUE.

Your job:
- Build the strongest possible case that the claim is TRUE.
- Use the provided evidence snippets. Cite them by number [1], [2], etc.
- If the opposing argument was given, rebut its weakest points specifically.
- Be concise: 3-5 sentences max. No fluff.
- Do NOT change your position. Your role is adversarial."""

DEBATER_B_SYSTEM = """You are Debater B, an expert fact-checker.
Your fixed position: the claim is FALSE.

Your job:
- Build the strongest possible case that the claim is FALSE.
- Use the provided evidence snippets. Cite them by number [1], [2], etc.
- If the opposing argument was given, rebut its weakest points specifically.
- Be concise: 3-5 sentences max. No fluff.
- Do NOT change your position. Your role is adversarial."""


def _format_evidence(snippets: list[str]) -> str:
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))


def argue(
    side: str,
    claim: str,
    evidence: list[str],
    opponent_argument: str = "",
    round_num: int = 1,
) -> str:
    """
    Generate one round of argumentation.

    Args:
        side:               "A" = pro-true  (Llama 4 Scout)
                            "B" = pro-false (Llama 3.1 8B)
        claim:              The factual claim.
        evidence:           Retrieved evidence snippets.
        opponent_argument:  Previous argument from the opposing debater.
        round_num:          Current round number.

    Returns:
        The debater's argument as a string.
    """
    if side == "A":
        system = DEBATER_A_SYSTEM
        model  = DEBATER_A_MODEL
        stance = "TRUE"
    else:
        system = DEBATER_B_SYSTEM
        model  = DEBATER_B_MODEL
        stance = "FALSE"

    evidence_str = _format_evidence(evidence)
    rebuttal_block = ""
    if opponent_argument and round_num > 1:
        rebuttal_block = (
            f"\n\nOpposing argument to rebut:\n\"\"\"\n{opponent_argument}\n\"\"\""
        )

    prompt = (
        f"Claim: {claim}\n\n"
        f"Evidence:\n{evidence_str}"
        f"{rebuttal_block}\n\n"
        f"Round {round_num}: Argue that the claim is {stance}."
    )

    return call_llm(prompt, system=system, model=model, max_tokens=300, temperature=0.5)
