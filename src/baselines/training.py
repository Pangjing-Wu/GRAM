"""Training routines used only by baseline methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.methods import ACTIVE_LABEL_CORRECTION

from ..utils.training import _fit_model, _infer


@dataclass
class SelectorArtifacts:
    probability: np.ndarray
    embedding: np.ndarray
    contribution: np.ndarray
    train_seconds: float


def train_robust_selector(
    data,
    labels: np.ndarray,
    *,
    config: dict,
    seed: int,
    cache_dir: str | Path,
) -> SelectorArtifacts:
    (
        model,
        evaluation_train_set,
        _,
        _,
        _,
        _,
        contribution,
        device,
        train_seconds,
        _,
    ) = _fit_model(
        data,
        labels,
        config=config,
        seed=seed,
        cache_dir=cache_dir,
        contribution_config=ACTIVE_LABEL_CORRECTION,
    )
    probability, embedding, _ = _infer(
        model,
        evaluation_train_set,
        num_classes=data.num_classes,
        embedding_dim=model.embedding_dim,
        batch_size=config["batch_size"],
        workers=config["num_workers"],
        device=device,
    )
    return SelectorArtifacts(
        probability=probability,
        embedding=embedding,
        contribution=contribution,
        train_seconds=train_seconds,
    )
