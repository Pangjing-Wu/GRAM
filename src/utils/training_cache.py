"""Persistent, process-safe cache for shared task-model training artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .training import TrainingArtifacts


CACHE_SCHEMA_VERSION = 1
_ARRAY_FIELDS = (
    "probability",
    "embedding",
    "early_loss",
    "final_loss",
    "aum",
    "forgetting",
    "forgetting_count",
    "learned",
    "margin_trajectory",
    "gradient_trajectory",
)


@dataclass(frozen=True)
class SharedTrainingArtifact:
    artifacts: TrainingArtifacts
    cache_hit: bool
    fingerprint: str
    directory: Path
    source_train_seconds: float


def _training_implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).with_name("training.py"),
        Path(__file__).with_name("models.py"),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def training_artifact_fingerprint(
    *,
    dataset: str,
    noise_type: str,
    rho: float,
    seed: int,
    dataset_config: dict,
    model_config: dict,
    noisy_labels: np.ndarray,
) -> tuple[str, dict]:
    labels = np.asarray(noisy_labels, dtype=np.int64).reshape(-1)
    label_digest = hashlib.sha256(labels.tobytes()).hexdigest()
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "training_implementation_sha256": _training_implementation_digest(),
        "dataset": dataset,
        "noise_type": noise_type,
        "rho": float(rho),
        "seed": int(seed),
        "dataset_config": dataset_config,
        "model_config": model_config,
        "noisy_label_count": int(len(labels)),
        "noisy_labels_sha256": label_digest,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _atomic_save_npz(path: Path, artifacts: TrainingArtifacts) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **{name: getattr(artifacts, name) for name in _ARRAY_FIELDS})
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, state: dict[str, torch.Tensor]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".pt", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_artifacts(
    arrays_path: Path,
    model_path: Path,
    manifest: dict,
    *,
    load_model_state: bool,
) -> TrainingArtifacts:
    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name], copy=True) for name in _ARRAY_FIELDS}
    model_state = (
        torch.load(model_path, map_location="cpu", weights_only=True)
        if load_model_state
        else None
    )
    return TrainingArtifacts(
        **arrays,
        test_accuracy=float("nan"),
        test_macro_f1=float("nan"),
        train_seconds=float(manifest["train_seconds"]),
        model_state=model_state,
        training_epochs=int(manifest["training_epochs"]),
        warm_started=False,
        epoch_offset=0,
    )


def load_or_create_shared_training_artifact(
    *,
    cache_root: str | Path,
    dataset: str,
    noise_type: str,
    rho: float,
    seed: int,
    dataset_config: dict,
    model_config: dict,
    noisy_labels: np.ndarray,
    load_model_state: bool,
    trainer: Callable[[], TrainingArtifacts],
) -> SharedTrainingArtifact:
    """Load one scenario's base training record or create it under a file lock."""
    fingerprint, payload = training_artifact_fingerprint(
        dataset=dataset,
        noise_type=noise_type,
        rho=rho,
        seed=seed,
        dataset_config=dataset_config,
        model_config=model_config,
        noisy_labels=noisy_labels,
    )
    directory = (
        Path(cache_root)
        / dataset
        / noise_type
        / f"rho{rho:g}"
        / f"seed{seed}"
        / fingerprint[:16]
    )
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path = directory / "artifacts.npz"
    model_path = directory / "model_state.pt"
    manifest_path = directory / "manifest.json"
    lock_path = directory / ".lock"

    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        complete = (
            arrays_path.is_file()
            and model_path.is_file()
            and manifest_path.is_file()
        )
        if complete:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            if manifest.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"shared training artifact fingerprint mismatch: {directory}"
                )
            artifacts = _load_artifacts(
                arrays_path,
                model_path,
                manifest,
                load_model_state=load_model_state,
            )
            return SharedTrainingArtifact(
                artifacts=artifacts,
                cache_hit=True,
                fingerprint=fingerprint,
                directory=directory,
                source_train_seconds=float(manifest["train_seconds"]),
            )

        artifacts = trainer()
        if artifacts.model_state is None:
            raise RuntimeError(
                "shared training artifact trainer did not capture model state"
            )
        _atomic_save_npz(arrays_path, artifacts)
        _atomic_torch_save(model_path, artifacts.model_state)
        manifest = {
            **payload,
            "fingerprint": fingerprint,
            "train_seconds": float(artifacts.train_seconds),
            "training_epochs": int(artifacts.training_epochs),
            "array_fields": list(_ARRAY_FIELDS),
        }
        _atomic_save_json(manifest_path, manifest)
        if not load_model_state:
            artifacts = replace(artifacts, model_state=None)
        return SharedTrainingArtifact(
            artifacts=artifacts,
            cache_hit=False,
            fingerprint=fingerprint,
            directory=directory,
            source_train_seconds=float(manifest["train_seconds"]),
        )
