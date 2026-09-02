from __future__ import annotations

from copy import deepcopy

from .datasets import DATASETS
from .models import MODELS


def load_config(dataset: str) -> tuple[dict, dict]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    return deepcopy(DATASETS[dataset]), deepcopy(MODELS[dataset])

