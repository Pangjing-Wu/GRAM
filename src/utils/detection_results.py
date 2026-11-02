"""Persist the per-budget detection state needed for downstream training."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


DETECTION_SNAPSHOTS_FILE = "downstream_snapshots.npz"
SNAPSHOT_SCHEMA_VERSION = 1
ARRAY_FIELDS = ("score", "predicted_issue", "queried", "repaired_labels")
SCALAR_FIELDS = (
    "checkpoint",
    "configured_budget_fraction",
    "decision_rule",
    "prediction_seconds",
)


def save_detection_snapshots(
    output: Path,
    snapshots: list[dict],
    noisy_labels: np.ndarray,
    clean_labels: np.ndarray,
) -> None:
    """Publish this last, after the detection CSVs and metadata are complete."""
    arrays = {
        name: np.asarray([snapshot[name] for snapshot in snapshots])
        for name in (*ARRAY_FIELDS, *SCALAR_FIELDS)
    }
    with tempfile.NamedTemporaryFile(
        dir=output, suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            schema_version=SNAPSHOT_SCHEMA_VERSION,
            noisy_labels=noisy_labels,
            clean_labels=clean_labels,
            **arrays,
        )
        os.replace(temporary, output / DETECTION_SNAPSHOTS_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def load_detection_snapshots(
    output: Path,
    *,
    budgets: tuple[float, ...],
    noisy_labels: np.ndarray,
    clean_labels: np.ndarray,
    target_counts: np.ndarray,
) -> list[dict]:
    """Reject partial, incompatible, or misaligned results before training."""
    with np.load(output / DETECTION_SNAPSHOTS_FILE, allow_pickle=False) as stored:
        if int(stored["schema_version"]) != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported detection snapshot schema")
        if not np.array_equal(stored["noisy_labels"], noisy_labels) or not np.array_equal(
            stored["clean_labels"], clean_labels
        ):
            raise ValueError("detection labels do not match the current dataset/noise")
        if not np.array_equal(stored["configured_budget_fraction"], budgets):
            raise ValueError("detection budget checkpoints do not match")
        if not np.array_equal(stored["checkpoint"], np.arange(1, len(budgets) + 1)):
            raise ValueError("detection checkpoint indices do not match")
        arrays = {name: stored[name] for name in (*ARRAY_FIELDS, *SCALAR_FIELDS)}

    shape = (len(budgets), len(noisy_labels))
    if any(arrays[name].shape != shape for name in ARRAY_FIELDS):
        raise ValueError("detection snapshot arrays have inconsistent shapes")
    if any(arrays[name].shape != (len(budgets),) for name in SCALAR_FIELDS):
        raise ValueError("detection snapshot metadata has inconsistent shapes")
    if not np.all(np.isfinite(arrays["score"])):
        raise ValueError("detection scores must be finite")
    if not np.all(np.isfinite(arrays["prediction_seconds"])) or np.any(
        arrays["prediction_seconds"] < 0
    ):
        raise ValueError("detection prediction timings must be finite and non-negative")
    for name in ("queried", "predicted_issue"):
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"detection {name} mask must be boolean")
    queried = arrays["queried"]
    if not np.array_equal(queried.sum(axis=1), target_counts):
        raise ValueError("detection query counts do not match the budget schedule")
    if np.any(queried[:-1] & ~queried[1:]):
        raise ValueError("detection queries must be cumulative")
    expected_labels = np.where(queried, clean_labels, noisy_labels)
    if not np.array_equal(arrays["repaired_labels"], expected_labels):
        raise ValueError("detection repaired labels do not match verified labels")
    if any(not rule for rule in arrays["decision_rule"]):
        raise ValueError("detection binary decision rules are missing")
    return [
        {name: values[index] for name, values in arrays.items()}
        for index in range(len(budgets))
    ]
