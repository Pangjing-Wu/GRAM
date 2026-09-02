"""Base mislabel scores shared with the AMADE benchmark.

All functions return a score whose larger value means that the assigned label
is less trustworthy. The policies add AUM-B-style stratified verification and
status calibration on top of every score.
"""

from __future__ import annotations

import numpy as np

from ..utils.neighbors import faiss_neighbors


SCORE_CALIBRATION_METHODS = (
    "aum_b",
    "el2n_b",
    "knn_label_disagreement_b",
    "moderate_b",
)

_RAW_METHOD = {
    "aum_b": "aum",
    "el2n_b": "el2n",
    "knn_label_disagreement_b": "knn_label_disagreement",
    "moderate_b": "moderate",
}


def _features(artifacts, feature_override: np.ndarray | None) -> np.ndarray:
    values = np.asarray(
        artifacts.embedding if feature_override is None else feature_override,
        dtype=np.float64,
    )
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("score baseline embeddings must be finite N x D values")
    return values


def _el2n(probability: np.ndarray, labels: np.ndarray) -> np.ndarray:
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[labels]
    return np.linalg.norm(probability - one_hot, axis=1)


def _knn_label_disagreement(
    features: np.ndarray, labels: np.ndarray, neighbors: int = 10
) -> np.ndarray:
    indices, _ = faiss_neighbors(
        features, min(neighbors, len(features) - 1), normalize=False
    )
    return np.mean(labels[indices] != labels[:, None], axis=1)


def _moderate_risk(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distance = np.empty(len(labels), dtype=np.float64)
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        prototype = np.median(features[ids], axis=0)
        distance[ids] = np.linalg.norm(features[ids] - prototype, axis=1)
    # Moderate selects points close to the global median distance.  AMADE
    # reverses that representativeness score when using it as mislabel risk.
    return np.abs(distance - np.median(distance))


def base_mislabel_score(
    method: str,
    artifacts,
    labels: np.ndarray,
    feature_override: np.ndarray | None = None,
) -> np.ndarray:
    if method not in SCORE_CALIBRATION_METHODS:
        raise ValueError(f"unknown score-calibration method: {method}")
    raw_method = _RAW_METHOD[method]
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.asarray(artifacts.probability, dtype=np.float64)
    if probability.ndim != 2 or len(probability) != len(labels):
        raise ValueError("probabilities and labels must be aligned")
    if raw_method == "aum":
        score = -np.asarray(artifacts.aum, dtype=np.float64)
    elif raw_method == "el2n":
        score = _el2n(probability, labels)
    elif raw_method == "knn_label_disagreement":
        score = _knn_label_disagreement(
            _features(artifacts, feature_override), labels
        )
    elif raw_method == "moderate":
        score = _moderate_risk(_features(artifacts, feature_override), labels)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if score.shape != labels.shape or not np.all(np.isfinite(score)):
        raise RuntimeError(f"{method} did not produce one finite score per sample")
    return score
