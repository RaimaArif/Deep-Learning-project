# Truth Through Debate: A Multi-Agent LLM Framework for Fact Checking

## Overview
This project evaluates whether **multi-agent debate** improves factual accuracy and reasoning quality over a single-LLM baseline, using the **FEVER dataset** for fact verification.

## Architecture
```
User Claim
    ↓
Retriever Agent  →  Top 3–5 evidence snippets
    ↓
Debater A (Pro-True)   Debater B (Pro-False)
    ↓           ↓
    Debate Rounds (2–3 turns)
    ↓
Judge Agent  →  Verdict (TRUE/FALSE) + Confidence + Explanation
```

## Project Structure
```
truth_through_debate/
├── agents/
│   ├── retriever.py       # Evidence retrieval agent
│   ├── debater.py         # Debater A (pro-true) & Debater B (pro-false)
│   └── judge.py           # Judge agent for final verdict
├── debate/
│   ├── engine.py          # Debate loop orchestrator
│   └── round.py           # Single debate round logic
├── evaluation/
│   ├── metrics.py         # Accuracy, reasoning quality, confidence calibration
│   └── evaluator.py       # Run experiments and compare systems
├── data/
│   ├── fever_loader.py    # FEVER dataset loader & preprocessor
│   └── sample_claims.py   # Small built-in sample for quick testing
├── experiments/
│   ├── baseline.py        # Single-LLM baseline experiment
│   └── debate_system.py   # Full debate system experiment
├── utils/
│   ├── llm_client.py      # Anthropic API wrapper
│   └── logger.py          # Structured logging
├── run_experiment.py      # Main entry point
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # Free key from console.groq.com
```

## Quick Start
```bash
# Run on built-in sample claims (no FEVER download needed)
python run_experiment.py --mode sample --n 10

# Run on FEVER dataset
python run_experiment.py --mode fever --n 50

# Run only baseline
python run_experiment.py --mode sample --system baseline

# Run only debate system
python run_experiment.py --mode sample --system debate

# Export results to CSV
python run_experiment.py --mode sample --n 20 --export results.csv
```

## Metrics
| Metric | Description |
|--------|-------------|
| Accuracy | % of verdicts matching ground truth |
| Reasoning Quality | 1–5 rubric: evidence use + logical consistency |
| Confidence Calibration | ECE score: does high confidence → correct? |
| Hallucination Proxy | Unsupported claims flagged in reasoning |

## Grading Checklist
- [x] Retriever Agent with evidence grounding
- [x] Debater A (Pro-True) + Debater B (Pro-False)
- [x] 2–3 round debate loop with rebuttals
- [x] Judge Agent with verdict + confidence + explanation
- [x] Single-LLM baseline
- [x] FEVER dataset integration
- [x] Accuracy, Reasoning Quality, Confidence Calibration metrics
- [x] Side-by-side comparison table
