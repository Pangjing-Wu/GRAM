from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VerificationBatch:
    positions: np.ndarray
    true_labels: np.ndarray
    mislabel_status: np.ndarray


class LabelVerificationOracle:
    def __init__(
        self,
        observed_labels: np.ndarray,
        true_labels: np.ndarray,
    ):
        observed = np.asarray(observed_labels, dtype=np.int64).reshape(-1)
        truth = np.asarray(true_labels, dtype=np.int64).reshape(-1)
        if observed.shape != truth.shape or observed.size == 0:
            raise ValueError("observed and true labels must be non-empty and aligned")
        self._true_labels = truth.copy()
        self._mislabel_mask = observed != truth
        self._verified = np.zeros(len(truth), dtype=bool)

    def verify(self, positions: np.ndarray) -> VerificationBatch:
        ids = np.asarray(positions, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("a verification batch cannot be empty")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("a verification batch contains duplicate positions")
        if np.any(ids < 0) or np.any(ids >= len(self._true_labels)):
            raise IndexError("verification position is outside the training set")
        if np.any(self._verified[ids]):
            raise ValueError("a sample cannot be verified more than once")
        self._verified[ids] = True
        return VerificationBatch(
            positions=ids.copy(),
            true_labels=self._true_labels[ids].copy(),
            mislabel_status=self._mislabel_mask[ids].astype(np.int64),
        )

    def ground_truth_mask(self) -> np.ndarray:
        return self._mislabel_mask.copy()
