"""
config.py
All configurable parameters in one place.
Override via environment variables or pass a Config object.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── API keys ──────────────────────────────────────────────────────────────
    groq_api_key: str        = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    tavily_api_key: str      = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # ── Model assignments ─────────────────────────────────────────────────────
    baseline_model: str      = "llama-3.3-70b-versatile"
    debater_a_model: str     = "meta-llama/llama-4-scout-17b-16e-instruct"
    debater_b_model: str     = "llama-3.1-8b-instant"
    devils_model: str        = "qwen/qwen3-32b"
    judge_model: str         = "qwen/qwen3-32b"
    scorer_model: str        = "moonshotai/kimi-k2-instruct"
    embedder_model: str      = "all-MiniLM-L6-v2"  # for FAISS

    # ── Debate hyperparameters ─────────────────────────────────────────────────
    num_rounds: int          = 2          # 2 or 3
    max_tokens_debate: int   = 350        # per debater turn
    max_tokens_judge: int    = 512
    max_tokens_scorer: int   = 256
    temperature_debate: float = 0.65
    temperature_judge: float  = 0.1
    word_cap_per_turn: int   = 200        # soft cap in prompt

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_per_source: int    = 3          # evidence per retriever
    top_k_final: int         = 8          # merged evidence fed to debate
    use_wikipedia: bool      = True
    use_tavily: bool         = True       # requires TAVILY_API_KEY
    use_bm25: bool           = True       # requires FEVER corpus
    use_faiss: bool          = True       # requires corpus + embedder

    fever_jsonl_path: str    = "data/fever_dev.jsonl"
    faiss_index_path: str    = "data/faiss.index"
    faiss_corpus_path: str   = "data/faiss_corpus.pkl"

    # ── Rate-limit / retry ────────────────────────────────────────────────────
    groq_retry_attempts: int  = 6
    groq_base_wait_s: int     = 20        # linear: 20, 40, 60, 80, 100, 120
    sleep_between_claims: int = 8         # seconds between claims in batch

    # ── Calibration ───────────────────────────────────────────────────────────
    calibration_bins: int     = 10
    platt_fit_min_samples: int = 20

    # ── Evaluation ────────────────────────────────────────────────────────────
    hallucination_conf_threshold: float = 0.7

    # ── UI ─────────────────────────────────────────────────────────────────────
    gradio_share: bool        = False
    gradio_port: int          = 7860

    def validate(self) -> None:
        if not self.groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set.\n"
                "  export GROQ_API_KEY=gsk_..."
            )
        if self.use_tavily and not self.tavily_api_key:
            import warnings
            warnings.warn("TAVILY_API_KEY not set — Tavily retrieval disabled.", stacklevel=2)
            self.use_tavily = False

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        cfg.validate()
        return cfg
