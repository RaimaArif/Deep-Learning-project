"""
calibration/platt.py
Platt scaling for confidence calibration + ECE computation + reliability diagram.

Platt scaling fits a logistic regression on (raw_confidence → correct) pairs,
transforming overconfident model outputs into well-calibrated probabilities.

Usage
-----
from truth_through_debate.calibration.platt import CalibrationPipeline

cal = CalibrationPipeline()
cal.fit(results)                         # fit on completed eval results
new_results = cal.transform(results)    # writes calibrated_confidence
cal.plot_reliability_diagram("reliability.png")
print(cal.ece_before, cal.ece_after)
"""
from __future__ import annotations
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class PlattCalibrator:
    """
    Logistic regression: P_cal = sigmoid(a * logit(confidence) + b)
    Fit with gradient descent (no sklearn dependency needed).
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0
        self._fitted: bool = False

    @staticmethod
    def _logit(p: float, eps: float = 1e-7) -> float:
        p = max(eps, min(1 - eps, p))
        return math.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def fit(self, confidences: list[float], labels: list[int],
            lr: float = 0.01, epochs: int = 2000) -> None:
        """
        confidences: raw model confidences [0,1]
        labels: 1 = correct, 0 = wrong
        """
        if len(confidences) < 2:
            log.warning("[Platt] Not enough samples to fit (%d). Using identity.", len(confidences))
            return

        X = [self._logit(c) for c in confidences]
        y = labels
        a, b = self.a, self.b

        for _ in range(epochs):
            da = db = 0.0
            for xi, yi in zip(X, y):
                p = self._sigmoid(a * xi + b)
                err = p - yi
                da += err * xi
                db += err
            a -= lr * da / len(X)
            b -= lr * db / len(X)

        self.a, self.b = a, b
        self._fitted = True
        log.info("[Platt] Fitted: a=%.4f  b=%.4f", a, b)

    def transform(self, confidence: float) -> float:
        if not self._fitted:
            return confidence
        return self._sigmoid(self.a * self._logit(confidence) + self.b)


def compute_ece(confidences: list[float], corrects: list[bool],
                n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins: dict[int, list] = defaultdict(list)
    for conf, correct in zip(confidences, corrects):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, int(correct)))

    ece = 0.0
    n = len(confidences)
    for items in bins.values():
        if not items:
            continue
        avg_conf = sum(x[0] for x in items) / len(items)
        avg_acc  = sum(x[1] for x in items) / len(items)
        ece += (len(items) / n) * abs(avg_conf - avg_acc)
    return round(ece, 5)


class CalibrationPipeline:
    """End-to-end calibration: fit Platt, compute ECE, plot reliability."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self.calibrator = PlattCalibrator()
        self.ece_before: float = 0.0
        self.ece_after: float = 0.0
        self._raw_confs: list[float] = []
        self._cal_confs: list[float] = []
        self._corrects: list[bool] = []

    def fit(self, results) -> "CalibrationPipeline":
        """
        Fit on DebateResult objects (must have .correct and .confidence).
        """
        from truth_through_debate.schema import DebateResult
        confs, labels = [], []
        for r in results:
            if r.correct is None:
                continue
            confs.append(r.confidence)
            labels.append(int(r.correct))

        if len(confs) < 4:
            log.warning("[Cal] Too few labelled results (%d). Calibration skipped.", len(confs))
            return self

        self._raw_confs = confs
        self._corrects  = [bool(l) for l in labels]
        self.ece_before = compute_ece(confs, self._corrects, self.n_bins)
        self.calibrator.fit(confs, labels)
        self._cal_confs = [self.calibrator.transform(c) for c in confs]
        self.ece_after  = compute_ece(self._cal_confs, self._corrects, self.n_bins)
        log.info("[Cal] ECE before=%.4f  after=%.4f", self.ece_before, self.ece_after)
        return self

    def transform(self, results) -> list:
        """Write calibrated_confidence back into each DebateResult."""
        for r in results:
            r.calibrated_confidence = self.calibrator.transform(r.confidence)
        return results

    def plot_reliability_diagram(self, save_path: str | None = None) -> None:
        """
        Reliability diagram: fraction correct vs mean confidence per bin.
        Saves PNG if save_path given, else shows interactively.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            log.warning("[Cal] matplotlib not installed. Skipping plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Reliability Diagrams — Debate System", fontsize=13, fontweight="bold")

        def _plot(ax, confs, corrects, title, ece):
            bins = defaultdict(list)
            for c, ok in zip(confs, corrects):
                b = min(int(c * self.n_bins), self.n_bins - 1)
                bins[b].append((c, int(ok)))

            xs, ys, ws = [], [], []
            for b in range(self.n_bins):
                if bins[b]:
                    xs.append(sum(x[0] for x in bins[b]) / len(bins[b]))
                    ys.append(sum(x[1] for x in bins[b]) / len(bins[b]))
                    ws.append(len(bins[b]))

            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
            sc = ax.scatter(xs, ys, c=ws, cmap="Blues", s=80, edgecolors="steelblue",
                            zorder=3, label="Bins")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Mean confidence"); ax.set_ylabel("Fraction correct")
            ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=11)
            ax.legend(fontsize=9)
            plt.colorbar(sc, ax=ax, label="Bin size")

        if self._raw_confs:
            _plot(axes[0], self._raw_confs, self._corrects, "Before calibration", self.ece_before)
            _plot(axes[1], self._cal_confs,  self._corrects, "After Platt scaling", self.ece_after)
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("[Cal] Reliability diagram saved → %s", save_path)
        else:
            plt.show()
        plt.close()
