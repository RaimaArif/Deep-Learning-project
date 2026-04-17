"""
tests/test_core.py
Tests for schema, fever_loader, config — no API key needed.
"""
import pytest
from truth_through_debate.schema import DebateResult, EvalMetrics, Evidence
from truth_through_debate.evaluation.fever_loader import load_builtin, BUILTIN_50
from truth_through_debate.config import Config


class TestSchema:
    def test_debate_result_to_dict(self):
        r = DebateResult(
            claim="test", ground_truth="TRUE", evidence=[], rounds=[],
            verdict="TRUE", explanation="ok", confidence=0.9,
        )
        d = r.to_dict()
        assert d["verdict"] == "TRUE"
        assert 0.0 <= d["confidence"] <= 1.0

    def test_evidence_str(self):
        ev = Evidence(text="Hello world", source="wikipedia", title="Test")
        assert "wikipedia" in str(ev)

    def test_eval_metrics_delta(self):
        m = EvalMetrics(
            n=50, accuracy=0.78, baseline_accuracy=0.94,
            avg_reasoning_score=4.7, baseline_avg_reasoning=3.64,
            ece=0.19, baseline_ece=0.05,
            hallucination_rate=0.06, baseline_hallucination_rate=0.24,
            avg_confidence=0.93,
        )
        rows = m.summary_rows()
        assert len(rows) >= 4
        # accuracy row should show negative delta
        acc_row = rows[0]
        assert "-" in acc_row[3]


class TestFeverLoader:
    def test_builtin_count(self):
        data = load_builtin(n=50)
        assert len(data) == 50

    def test_builtin_balanced(self):
        data = load_builtin(n=50)
        true_count  = sum(1 for _, l in data if l == "TRUE")
        false_count = sum(1 for _, l in data if l == "FALSE")
        assert true_count == 25 and false_count == 25

    def test_builtin_small_n(self):
        data = load_builtin(n=10)
        assert len(data) == 10

    def test_labels_valid(self):
        for _, label in BUILTIN_50:
            assert label in {"TRUE", "FALSE", "NOT ENOUGH INFO"}


class TestConfig:
    def test_default_models(self):
        cfg = Config(groq_api_key="test_key")
        assert "llama" in cfg.baseline_model
        assert cfg.num_rounds == 2

    def test_validate_missing_key(self):
        cfg = Config(groq_api_key="")
        with pytest.raises(EnvironmentError):
            cfg.validate()

    def test_validate_tavily_warning(self):
        import warnings
        cfg = Config(groq_api_key="test", tavily_api_key="", use_tavily=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg.validate()
            assert any("TAVILY" in str(warning.message) for warning in w)
        assert cfg.use_tavily is False
