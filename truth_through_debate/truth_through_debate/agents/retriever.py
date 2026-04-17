"""
agents/retriever.py
Retriever Agent — fetches 3-5 evidence snippets for a given claim.
Model: Llama 4 Scout (DEBATER_A_MODEL) — fast and efficient for retrieval.
"""
import re
from utils.llm_client import call_llm, DEBATER_A_MODEL

RETRIEVER_SYSTEM = """You are a fact-checking research assistant.
Given a claim, retrieve 3-5 concise factual statements from your knowledge
that are DIRECTLY relevant to evaluating the claim.

Rules:
- Each snippet must be a standalone factual sentence.
- Do NOT state whether the claim is true or false.
- Number each snippet: 1. ... 2. ... etc.
- Keep each snippet under 40 words.
- Prefer specific facts: dates, numbers, names, events.
- If you are uncertain about a fact, omit it."""


def retrieve(claim: str, n: int = 5) -> list[str]:
    """Return a list of evidence snippets relevant to the claim."""
    prompt = (
        f"Claim: {claim}\n\n"
        f"Retrieve {n} concise factual evidence snippets relevant to this claim."
    )
    raw = call_llm(
        prompt,
        system=RETRIEVER_SYSTEM,
        model=DEBATER_A_MODEL,
        max_tokens=400,
        temperature=0.1,
    )
    return _parse_snippets(raw)


def _parse_snippets(raw: str) -> list[str]:
    lines = raw.strip().splitlines()
    snippets = []
    for line in lines:
        line = line.strip()
        match = re.match(r"^\d+[\.\)]\s+(.+)", line)
        if match:
            snippets.append(match.group(1).strip())
    if not snippets:
        snippets = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return snippets[:5]
