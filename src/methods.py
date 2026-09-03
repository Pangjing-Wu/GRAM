"""Method registry and shared factory for experiment protocols."""

from __future__ import annotations

import numpy as np

from .baselines.policy import (
    BASELINE_METHODS,
    CORRECTION_METHODS as BASELINE_CORRECTION_METHODS,
    STATUS_ONLY_METHODS as BASELINE_STATUS_ONLY_METHODS,
    WARM_START_METHODS as BASELINE_WARM_START_METHODS,
    BaselinePolicy,
    method_feedback_mode as baseline_feedback_mode,
    method_update_schedule as baseline_update_schedule,
    should_train_acquisition_model as should_train_baseline_model,
)
from config.methods import resolve_gram_variant

from .gram import GRAM_METHOD, GramPolicy


METHODS = (*BASELINE_METHODS, GRAM_METHOD)
STATUS_ONLY_METHODS = {*BASELINE_STATUS_ONLY_METHODS, GRAM_METHOD}
CORRECTION_METHODS = set(BASELINE_CORRECTION_METHODS)
WARM_START_METHODS = set(BASELINE_WARM_START_METHODS)


def create_method_policy(
    method: str,
    noisy_labels: np.ndarray,
    *,
    modality: str,
    seed: int,
    native_features: np.ndarray | None = None,
    gram_variant: str | None = None,
):
    if method == GRAM_METHOD:
        variant = resolve_gram_variant(method, gram_variant)
        return GramPolicy(
            noisy_labels,
            modality=modality,
            seed=seed,
            native_features=native_features,
            variant=variant,
        )
    if method in BASELINE_METHODS:
        resolve_gram_variant(method, gram_variant)
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


def method_update_schedule(method: str, *, warm_start: bool = False) -> str:
    if method == GRAM_METHOD:
        return (
            "task_model_once_kernel_weight_and_status_inference_"
            "each_budget_checkpoint"
        )
    return baseline_update_schedule(method, warm_start=warm_start)
