"""Method registry and shared factory for experiment protocols."""

from __future__ import annotations

import numpy as np

from .baselines.policy import (
    BASELINE_METHODS,
    CORRECTION_METHODS as BASELINE_CORRECTION_METHODS,
    STATUS_ONLY_METHODS as BASELINE_STATUS_ONLY_METHODS,
    BaselinePolicy,
    method_feedback_mode as baseline_feedback_mode,
    method_update_schedule as baseline_update_schedule,
    should_train_acquisition_model as should_train_baseline_model,
)
from .gram import GRAM_METHOD, GramPolicy


METHODS = (*BASELINE_METHODS, GRAM_METHOD)
STATUS_ONLY_METHODS = {*BASELINE_STATUS_ONLY_METHODS, GRAM_METHOD}
CORRECTION_METHODS = set(BASELINE_CORRECTION_METHODS)


def create_method_policy(
    method: str,
    noisy_labels: np.ndarray,
    *,
    modality: str,
    seed: int,
    native_features: np.ndarray | None = None,
):
    if method == GRAM_METHOD:
        return GramPolicy(
            noisy_labels,
            modality=modality,
            seed=seed,
            native_features=native_features,
        )
    if method in BASELINE_METHODS:
        return BaselinePolicy(
            method,
            noisy_labels,
            modality=modality,
            seed=seed,
            native_features=native_features,
        )
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def should_train_acquisition_model(method: str, checkpoint_index: int) -> bool:
    if method == GRAM_METHOD:
        return checkpoint_index == 0
    return should_train_baseline_model(method, checkpoint_index)


def method_feedback_mode(method: str) -> str:
    if method == GRAM_METHOD:
        return "mislabel_status"
    return baseline_feedback_mode(method)


def method_update_schedule(method: str) -> str:
    if method == GRAM_METHOD:
        return "task_model_once_status_inference_each_budget_checkpoint"
    return baseline_update_schedule(method)
