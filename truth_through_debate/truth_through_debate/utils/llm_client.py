"""
utils/llm_client.py
Groq SDK wrapper — 5 unique free models, one per role.
Get your free API key at: https://console.groq.com

Model assignments:
  BASELINE_MODEL  — Llama 3.3 70B      (Meta, strongest — hard baseline to beat)
  DEBATER_A_MODEL — Llama 4 Scout 17B  (Meta, newer architecture, argues TRUE)
  DEBATER_B_MODEL — Llama 3.1 8B       (Meta, small & fast, argues FALSE)
  JUDGE_MODEL     — Qwen3 32B          (Alibaba, different family, neutral arbiter)
  SCORER_MODEL    — Kimi K2            (Moonshot AI, independent evaluator)
"""
import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None

# ── 5 unique models — all free on Groq ───────────────────────────────────────
BASELINE_MODEL  = "llama-3.3-70b-versatile"
DEBATER_A_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEBATER_B_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL     = "qwen/qwen3-32b"
SCORER_MODEL    = "moonshotai/kimi-k2-instruct"
# ─────────────────────────────────────────────────────────────────────────────


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to Colab Secrets or your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


def call_llm(
    prompt: str,
    system: str = "",
    model: str = BASELINE_MODEL,
    max_tokens: int = 800,
    temperature: float = 0.3,
    retries: int = 3,
    backoff: float = 5.0,
    **kwargs,
) -> str:
    """
    Call the Groq API and return the text response.

    Args:
        prompt:      User-turn message.
        system:      System prompt for role/persona.
        model:       Groq model string. Use the constants above.
        max_tokens:  Max output tokens.
        temperature: Sampling temperature (lower = more deterministic).
        retries:     Number of retry attempts on transient errors.
        backoff:     Exponential backoff base in seconds.

    Returns:
        Assistant response as a plain string.
    """
    client = get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                wait = backoff ** (attempt + 1)
                print(f"[Groq:{model}] Rate limited. Waiting {wait:.1f}s...")
                time.sleep(wait)
            elif attempt == retries - 1:
                raise
            else:
                print(f"[Groq:{model}] Error: {e}. Retrying...")
                time.sleep(backoff)

    raise RuntimeError(f"Groq API call failed after all retries. Model: {model}")
