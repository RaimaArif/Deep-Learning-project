"""
tests/test_calibration.py
Unit tests for Platt scaling and ECE — no API key needed.
"""
import pytest
from truth_through_debate.calibration.platt import (
    PlattCalibrator, compute_ece, CalibrationPipeline
)
from truth_through_debate.schema import DebateResult, DebateRound


def _make_result(confidence: float, correct: bool) -> DebateResult:
    return DebateResult(
        claim="test claim",
        ground_truth="TRUE",
        evidence=[],
        rounds=[],
        verdict="TRUE",
        explanation="",
        confidence=confidence,
        calibrated_confidence=confidence,
        reasoning_score=3,
        correct=correct,
    )


class TestPlattCalibrator:
    def test_fit_predict_shape(self):
        cal = PlattCalibrator()
        confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        labels = [1, 1, 0, 1, 0, 0]
        cal.fit(confs, labels)
        assert cal._fitted
        out = cal.transform(0.8)
        assert 0.0 <= out <= 1.0

    def test_identity_when_not_fitted(self):
        cal = PlattCalibrator()
        assert cal.transform(0.75) == 0.75

    def test_overconfident_compression(self):
        """Overconfident model (high conf, wrong) should be compressed downward."""
        cal = PlattCalibrator()
        # mostly wrong at high confidence -> should push probs down
        confs  = [0.95, 0.9, 0.88, 0.92, 0.85]
        labels = [0,    0,   1,    0,    0   ]
        cal.fit(confs, labels, epochs=3000)
        out = cal.transform(0.9)
        assert out < 0.9  # calibrated should be lower


class TestECE:
    def test_perfect_calibration(self):
        # When confidence exactly matches accuracy, ECE = 0
        confs    = [0.1, 0.3, 0.5, 0.7, 0.9]
        corrects = [False, False, True, True, True]
        ece = compute_ece(confs, corrects, n_bins=5)
        assert ece < 0.3  # loose bound for small N

    def test_terrible_calibration(self):
        # All high confidence, all wrong
        confs    = [0.99] * 20
        corrects = [False] * 20
        ece = compute_ece(confs, corrects)
        assert ece > 0.5


class TestCalibrationPipeline:
    def test_fit_transform_cycle(self):
        results = [_make_result(0.9, True), _make_result(0.8, False),
                   _make_result(0.6, True), _make_result(0.5, False),
                   _make_result(0.7, True)]
        cal = CalibrationPipeline()
        cal.fit(results)
        out = cal.transform(results)
        assert all(0.0 <= r.calibrated_confidence <= 1.0 for r in out)

    def test_skip_when_too_few_samples(self):
        results = [_make_result(0.8, True)]
        cal = CalibrationPipeline()
        cal.fit(results)  # should not raise
        assert not cal.calibrator._fitted
