"""Classification calibration and selective prediction metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def calibration_metrics(
    correct: list[bool],
    confidence: list[float],
    *,
    bins: int = 15,
) -> dict[str, Any]:
    if len(correct) != len(confidence) or not correct:
        raise ValueError("correct and confidence must be non-empty and aligned")
    y = np.asarray(correct, dtype=float)
    p = np.clip(np.asarray(confidence, dtype=float), 1e-12, 1 - 1e-12)
    order = np.argsort(p)
    groups = np.array_split(order, min(bins, len(order)))
    reliability: list[dict[str, float | int]] = []
    ece = 0.0
    for group in groups:
        accuracy = float(np.mean(y[group]))
        mean_confidence = float(np.mean(p[group]))
        weight = len(group) / len(y)
        ece += weight * abs(accuracy - mean_confidence)
        reliability.append(
            {
                "n": len(group),
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    ranked = np.argsort(-p)
    cumulative_errors = np.cumsum(1 - y[ranked])
    coverage = np.arange(1, len(y) + 1) / len(y)
    risk = cumulative_errors / np.arange(1, len(y) + 1)
    idx_80 = max(0, math.ceil(0.8 * len(y)) - 1)
    return {
        "adaptive_ece": float(ece),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "reliability": reliability,
        "risk_coverage": [
            {"coverage": float(c), "risk": float(r)} for c, r in zip(coverage, risk, strict=True)
        ],
        "aurc": float(np.trapezoid(risk, coverage)),
        "accuracy_at_80_coverage": float(1 - risk[idx_80]),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(correct)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if any(correct) else None,
    }


def calibrate_threshold(labels: list[bool], scores: list[float]) -> dict[str, float]:
    if len(labels) != len(scores) or len(set(labels)) < 2:
        raise ValueError("Calibration requires aligned positive and negative examples")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.argmax(f1[:-1]))
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "threshold": float(thresholds[index]),
        "dev_f1": float(f1[index]),
    }
