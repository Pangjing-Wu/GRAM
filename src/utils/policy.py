"""Shared policy result types and selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass
class QueryResult:
    selected: np.ndarray
    score: np.ndarray
    posterior_variance: np.ndarray | None = None
    pseudo_selected: np.ndarray | None = None
    repair_value: np.ndarray | None = None
    informative_value: np.ndarray | None = None


@dataclass
class MethodPrediction:
    score: np.ndarray
    predicted_issue: np.ndarray
    decision_rule: str


def entropy(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability), 1.0e-12, 1.0)
    return -(values * np.log(values)).sum(axis=1)


def query_candidates(
    queried: np.ndarray, candidate_mask: np.ndarray | None
) -> np.ndarray:
    queried = np.asarray(queried, dtype=bool).reshape(-1)
    if candidate_mask is None:
        eligible = np.ones(len(queried), dtype=bool)
    else:
        eligible = np.asarray(candidate_mask, dtype=bool).reshape(-1)
        if eligible.shape != queried.shape:
            raise ValueError("candidate mask and queried mask must be aligned")
    return np.flatnonzero(eligible & ~queried)


def top_scores(
    scores: np.ndarray, candidates: np.ndarray, count: int, seed: int
) -> np.ndarray:
    tie = np.random.default_rng(seed).random(len(candidates))
    order = np.lexsort((tie, -np.asarray(scores)[candidates]))
    return candidates[order[:count]]


def percentile_mislabel_score(score: np.ndarray) -> np.ndarray:
    """Map an arbitrary native ranking score to a finite value in [0, 1]."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if score.size == 0 or not np.all(np.isfinite(score)):
        raise ValueError("global mislabel scores must be non-empty and finite")
    if np.all(score == score[0]):
        return np.full(len(score), 0.5, dtype=np.float64)
    return rankdata(score, method="average") / (len(score) + 1.0)
