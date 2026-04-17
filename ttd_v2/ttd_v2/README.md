# Truth Through Debate v2

[![CI](https://github.com/YOUR_USERNAME/truth-through-debate/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/truth-through-debate/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/truth-through-debate/blob/main/notebooks/Truth_Through_Debate_v2.ipynb)

**Multi-agent LLM debate framework for fact-checking.**  
Five free Groq models. Hybrid retrieval (Wikipedia + Tavily + BM25 + FAISS). Platt-scaled confidence calibration. Gradio UI.

---

## Architecture

```
Claim
  │
  ▼
Hybrid Retriever (async, parallel)
  ├── Wikipedia API        (free)
  ├── Tavily web search    (free tier)
  ├── BM25 / FEVER corpus  (offline)
  └── FAISS vector store   (offline)
  │
  ▼
Async Debate Engine (2–3 rounds, A+B run in parallel per round)
  ├── Debater A  — argues TRUE       (Llama 4 Scout)
  ├── Debater B  — argues FALSE      (Llama 3.1 8B)
  └── Devil's Advocate — challenges both (Qwen3-32B)
  │
  ▼
Judge  → Verdict + Confidence    (Qwen3-32B)
Scorer → Reasoning score 1–5     (Kimi-K2)
Platt Calibration → calibrated confidence
  │
  ▼
Outputs: Gradio UI | CSV/JSONL | Reliability diagram | CLI
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/truth-through-debate.git
cd truth-through-debate
pip install -e ".[dev]"
```

### 2. Set API keys

```bash
export GROQ_API_KEY=gsk_...          # required — free at console.groq.com
export TAVILY_API_KEY=tvly_...       # optional — free at tavily.com
```

### 3. Run a single claim

```bash
ttd debate "The Great Wall of China is visible from space."
ttd debate "The Berlin Wall fell in 1989." --truth TRUE --rounds 3 --verbose
```

### 4. Launch the Gradio UI

```bash
ttd ui                        # http://localhost:7860
ttd ui --share                # public Gradio link
```

### 5. Batch evaluation

```bash
# Built-in 50 claims (no download needed)
ttd eval --source builtin --n 50 --output outputs/

# FEVER dataset (download from fever.ai)
ttd eval --source fever --fever-path data/fever_dev.jsonl --n 200 --rounds 2
```

---

## Google Colab

Open [`notebooks/Truth_Through_Debate_v2.ipynb`](notebooks/Truth_Through_Debate_v2.ipynb) in Colab:

1. Add `GROQ_API_KEY` to Colab Secrets (key icon → toggle Notebook access ON)
2. Run all cells
3. Uses built-in 50 claims by default — no downloads required

---

## Python API

```python
import asyncio
from truth_through_debate import DebatePipeline, Config

async def main():
    cfg = Config.from_env()
    async with DebatePipeline(cfg) as pipeline:
        result = await pipeline.run(
            "The Amazon River is the longest river in the world.",
            ground_truth="FALSE",
        )
    print(result.verdict)              # FALSE
    print(result.confidence)           # 0.92
    print(result.calibrated_confidence)# 0.87 (Platt scaled)
    print(result.reasoning_score)      # 4
    print(result.explanation)          # "The Nile is longer..."

asyncio.run(main())
```

### Batch evaluation

```python
from truth_through_debate.evaluation.evaluator import run_evaluation
from truth_through_debate.evaluation.fever_loader import load_builtin

claims = load_builtin(n=50)
results, metrics = asyncio.run(run_evaluation(claims, output_dir="outputs/"))
print(f"Accuracy: {metrics.accuracy:.3f}")
print(f"ECE: {metrics.ece:.4f}")
```

### Build FAISS index

```bash
# Corpus JSONL: one {"text": "...", "title": "...", "url": "..."} per line
ttd build-index --corpus data/wiki_corpus.jsonl
```

---

## Results (n=50 built-in claims)

| Metric | Baseline | Debate System | Delta |
|--------|----------|---------------|-------|
| Accuracy | 0.940 | 0.780 | -0.160 |
| Reasoning Quality (1–5) | 3.64 | 4.70 | +1.06 |
| ECE ↓ | 0.052 | 0.190 | +0.138 |
| Hallucination Rate ↓ | 0.240 | 0.060 | -0.180 |
| Avg confidence | 0.964 | 0.930 | -0.034 |

> **Note:** Debate accuracy drops vs baseline because well-known refutation claims (e.g. "Bats are blind") have convincing counter-arguments that fool the judge — a known failure mode explored in the report. Reasoning quality and hallucination resistance improve substantially.

---

## Model lineup (all free on Groq)

| Role | Model |
|------|-------|
| Baseline | `llama-3.3-70b-versatile` |
| Debater A (argues TRUE) | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Debater B (argues FALSE) | `llama-3.1-8b-instant` |
| Devil's Advocate | `qwen/qwen3-32b` |
| Judge | `qwen/qwen3-32b` |
| Scorer | `moonshotai/kimi-k2-instruct` |

---

## Project Structure

```
truth_through_debate/
├── agents/
│   ├── llm_client.py          # Async Groq client with retry
│   └── debate_agents.py       # All 6 agent roles
├── retrieval/
│   └── hybrid_retriever.py    # Wikipedia + Tavily + BM25 + FAISS
├── calibration/
│   └── platt.py               # Platt scaling + ECE + reliability diagram
├── evaluation/
│   ├── evaluator.py           # Batch eval + metrics + CSV export
│   └── fever_loader.py        # FEVER dataset loader + built-in 50 claims
├── ui/
│   └── app.py                 # Gradio web UI
├── pipeline.py                # Main async orchestrator
├── config.py                  # All hyperparameters
├── schema.py                  # Typed dataclasses
└── cli.py                     # `ttd` CLI entry point
tests/
notebooks/                     # Colab notebook
.github/workflows/ci.yml       # GitHub Actions CI
pyproject.toml                 # Installable package
```

---

## Running tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=truth_through_debate
```

Tests that don't require `GROQ_API_KEY`: `test_calibration.py`, `test_core.py`

---

## Citation

```bibtex
@misc{truth_through_debate2025,
  title  = {Truth Through Debate: A Multi-Agent LLM Framework for Fact-Checking},
  year   = {2025},
  note   = {Deep Learning Course Project},
  url    = {https://github.com/YOUR_USERNAME/truth-through-debate}
}
```

---

## License

MIT — see [LICENSE](LICENSE)
