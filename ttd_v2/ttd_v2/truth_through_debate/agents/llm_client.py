"""
agents/llm_client.py
Async Groq API wrapper with linear back-off retry.
Supports both async (aio) and sync usage.
"""
from __future__ import annotations
import asyncio
import time
import json
import re
import logging
from typing import Any

import aiohttp

from truth_through_debate.config import Config

log = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient:
    """Async Groq client, one instance per pipeline run."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "GroqClient":
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.cfg.groq_api_key}",
                     "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=120),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Send a chat completion and return the assistant text."""
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        for attempt in range(self.cfg.groq_retry_attempts):
            try:
                async with self._session.post(_GROQ_URL, json=payload) as resp:
                    if resp.status == 429 or resp.status == 503:
                        wait = self.cfg.groq_base_wait_s * (attempt + 1)
                        log.warning("[Groq:%s] HTTP %d — waiting %ds (attempt %d)",
                                    model, resp.status, wait, attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except aiohttp.ClientResponseError as e:
                if attempt == self.cfg.groq_retry_attempts - 1:
                    raise
                wait = self.cfg.groq_base_wait_s * (attempt + 1)
                log.warning("[Groq:%s] Error %s — waiting %ds", model, e, wait)
                await asyncio.sleep(wait)
        raise RuntimeError(f"Groq call failed after {self.cfg.groq_retry_attempts} attempts")

    def complete_sync(self, model: str, system: str, user: str,
                      max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Sync wrapper (used by Gradio callbacks)."""
        return asyncio.get_event_loop().run_until_complete(
            self.complete(model, system, user, max_tokens, temperature)
        )


def parse_json_response(raw: str) -> dict:
    # Strip <think>...</think> or unclosed <think>... (Qwen3 chain-of-thought)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)
