"""Offline issue-ranking metrics for full-pool detection experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class DetectionMetrics:
    auprc: float
    auroc: float


def evaluate_detection(
    status: np.ndarray,
    score: np.ndarray,
) -> DetectionMetrics:
    """Evaluate issue-ranking scores on one offline subset."""
    status = np.asarray(status, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if status.shape != score.shape or not status.size:
        raise ValueError("status and scores must be non-empty and aligned")
    if not np.all(np.isin(status, (0, 1))):
        raise ValueError("issue status must be binary")
    if not np.all(np.isfinite(score)):
        raise ValueError("issue scores must be finite")

    if len(np.unique(status)) < 2:
        return DetectionMetrics(auprc=float("nan"), auroc=float("nan"))
    return DetectionMetrics(
        auprc=float(average_precision_score(status, score)),
        auroc=float(roc_auc_score(status, score)),
    )
