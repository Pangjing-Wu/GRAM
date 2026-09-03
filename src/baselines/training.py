"""Training routines used only by baseline methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from config.methods import ACTIVE_LABEL_CORRECTION

from ..utils.training import _fit_model, _infer


@dataclass
class SelectorArtifacts:
    probability: np.ndarray
    embedding: np.ndarray
    contribution: np.ndarray
    train_seconds: float
    model_state: dict[str, torch.Tensor]
    training_epochs: int
    warm_started: bool
    epoch_offset: int


def train_robust_selector(
    data,
    labels: np.ndarray,
    *,
    config: dict,
    seed: int,
    cache_dir: str | Path,
    initial_model_state: Mapping[str, torch.Tensor] | None = None,
    training_epochs: int | None = None,
    learning_rate: float | None = None,
    epoch_offset: int = 0,
    initial_contribution: np.ndarray | None = None,
) -> SelectorArtifacts:
    (
        model,
        evaluation_train_set,
        _,
        _,
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
        initial_model_state=initial_model_state,
        training_epochs=training_epochs,
        learning_rate=learning_rate,
        epoch_offset=epoch_offset,
        initial_contribution=initial_contribution,
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
        model_state={
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        training_epochs=(
            int(config["epochs"])
            if training_epochs is None
            else int(training_epochs)
        ),
        warm_started=initial_model_state is not None,
        epoch_offset=int(epoch_offset),
    )
