"""
Truth Through Debate v2
=======================
Multi-agent LLM debate framework for fact-checking.

Quick start
-----------
>>> from truth_through_debate import DebatePipeline
>>> pipeline = DebatePipeline()
>>> result = await pipeline.run("The Great Wall is visible from space.")
>>> print(result.verdict, result.confidence)
"""

from truth_through_debate.pipeline import DebatePipeline
from truth_through_debate.config import Config
from truth_through_debate.schema import DebateResult, EvalMetrics

__version__ = "2.0.0"
__all__ = ["DebatePipeline", "Config", "DebateResult", "EvalMetrics"]
