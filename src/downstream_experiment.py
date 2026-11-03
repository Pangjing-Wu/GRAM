"""Checkpoint-wise repair and automatic filtering with fresh downstream models."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from config import load_config
from config.methods import resolve_gram_variant
from .detection_experiment import DetectionScenario
from .downstream_detection import ensure_detection_results
from .methods import METHODS
from .utils.budget import validate_budget_checkpoints
from .utils.data import load_data
from scripts.best_k import get_best_k
from .utils.noise import inject_noise
from .utils.run_logging import ensure_output_available, run_logger
from .utils.training import train_downstream_model


@dataclass(frozen=True)
class DownstreamScenario:
    noise_type: str
    rho: float
    budgets: tuple[float, ...]
    seed: int

    def validate(self) -> None:
        if self.noise_type not in {"symmetric", "pairflip", "instance"}:
            raise ValueError("noise_type must be symmetric, pairflip, or instance")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must be in (0, 1)")
        validate_budget_checkpoints(self.budgets)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def _atomic_write(path: Path, write) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write(path, write)


def _write_config(path: Path, config: dict) -> None:
    _atomic_write(path, lambda stream: json.dump(
        config, stream, indent=2, sort_keys=True, allow_nan=False,
    ))


def otsu_threshold(scores: np.ndarray) -> float:
    """Exact one-dimensional Otsu split; no histogram bins or tuned parameters.

    Fit on the full pool, including verified samples. Constant scores provide no
    separation, so return that value and retain all samples under the strict > rule.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Otsu requires a non-empty finite score vector")
    values = np.sort(values)
    if values[0] == values[-1]:
        return float(values[0])
    # Positive affine scaling leaves the optimum unchanged and avoids overflow.
    scale = max(abs(values[0]), abs(values[-1]))
    scaled = values / scale
    scaled -= scaled[0]
    cut = np.flatnonzero(values[:-1] < values[1:])
    cumulative = np.cumsum(scaled)
    left_count = cut + 1
    objective = (
        cumulative[cut] ** 2 / left_count
        + (cumulative[-1] - cumulative[cut]) ** 2 / (len(values) - left_count)
    )
    index = cut[int(np.argmax(objective))]
    threshold = float(values[index] / 2.0 + values[index + 1] / 2.0)
    # Adjacent floating-point numbers may have no representable midpoint.
    return float(values[index]) if threshold >= values[index + 1] else threshold


def downstream_output_dir(root, dataset, method, scenario, gram_variant=None) -> Path:
    output = Path(root) / "downstream" / dataset / scenario.noise_type / f"rho{scenario.rho:g}" / method
    variant = resolve_gram_variant(method, gram_variant)
    if variant is not None:
        output /= variant
    return output / f"seed{scenario.seed}"


_output_dir = downstream_output_dir


def downstream_reference_output_dir(root, dataset, reference, seed, *, noise_type=None, rho=None) -> Path:
    output = Path(root) / "downstream" / dataset
    if reference == "clean":
        if noise_type is not None or rho is not None:
            raise ValueError("the clean reference has no noise_type or rho")
    elif reference == "noisy":
        if noise_type not in {"symmetric", "pairflip", "instance"} or rho not in {0.2, 0.4}:
            raise ValueError("the noisy reference requires a noise_type and rho=0.2 or 0.4")
        output = output / noise_type / f"rho{rho:g}"
    else:
        raise ValueError("reference must be noisy or clean")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return output / reference / f"seed{seed}"


def _result_row(data, labels, retained, queried, noise_mask, *, setting,
                checkpoint, budget, threshold, model_config, seed, data_root):
    result = train_downstream_model(
        data, labels, np.flatnonzero(retained), config=model_config,
        seed=seed, cache_dir=data_root / "huggingface",
    )
    deleted = ~retained
    deleted_count = int(deleted.sum())
    deleted_wrong = int((deleted & noise_mask).sum())
    unverified = ~queried
    remaining = int((labels[retained] != data.clean_train_y[retained]).sum())
    query_count = int(queried.sum())
    found = int((queried & noise_mask).sum())
    is_clean_reference = setting == "clean"
    return {
        "setting": setting,
        "checkpoint": checkpoint,
        "configured_budget_fraction": budget,
        "verification_budget": None if is_clean_reference else query_count,
        "verification_fraction": None if is_clean_reference else float(queried.mean()),
        "queried_mislabels_found": None if is_clean_reference else found,
        "query_precision": found / query_count if query_count else None,
        "otsu_threshold": threshold,
        "deleted_count": deleted_count,
        "deleted_fraction_unverified": deleted_count / int(unverified.sum()) if unverified.any() else 0.0,
        "deleted_mislabels": deleted_wrong,
        "deleted_clean_samples": deleted_count - deleted_wrong,
        "delete_precision": deleted_wrong / deleted_count if deleted_count else None,
        "delete_recall": deleted_wrong / int((noise_mask & unverified).sum())
        if (noise_mask & unverified).any() else None,
        "retained_count": int(retained.sum()),
        "retained_fraction": float(retained.mean()),
        "remaining_mislabels": remaining,
        "remaining_noise_rate": remaining / int(retained.sum()),
        "diagnostic_retained_label_accuracy": 1.0 - remaining / int(retained.sum()),
        "downstream_test_metric": data.test_metric,
        "downstream_test_performance": result.test_accuracy
        if data.test_metric == "accuracy" else result.test_macro_f1,
        "downstream_test_accuracy": result.test_accuracy,
        "downstream_test_macro_f1": result.test_macro_f1,
        "downstream_train_seconds": result.train_seconds,
    }


def run_downstream_reference(
    *, dataset: str, reference: str, seed: int,
    data_root: str | Path, results_root: str | Path,
    noise_type: str | None = None, rho: float | None = None,
    overwrite: bool = False,
) -> Path:
    """Train one shared unmodified-data reference without detection results."""
    output = downstream_reference_output_dir(
        results_root, dataset, reference, seed, noise_type=noise_type, rho=rho,
    )
    ensure_output_available(output, overwrite)
    started = time.perf_counter()
    data_root = Path(data_root)
    dataset_config, model_config = load_config(dataset)
    data = load_data(
        dataset, root=data_root, seed=seed,
        dataset_config=dataset_config, model_config=model_config,
    )
    labels = data.clean_train_y
    noise_mask = np.zeros(data.sample_count, dtype=bool)
    scenario = {"seed": seed}
    if reference == "noisy":
        labels, noise_mask = inject_noise(
            data.clean_train_y, data.noise_x, noise_type=noise_type,
            rho=rho, num_classes=data.num_classes, seed=seed,
        )
        labels = np.asarray(labels, dtype=np.int64)
        if not np.array_equal(noise_mask, labels != data.clean_train_y):
            raise RuntimeError("noise mask and labels disagree")
        scenario.update(noise_type=noise_type, rho=rho)
    downstream_seed = int(np.random.SeedSequence([seed, 65_537]).generate_state(1)[0])
    config = {
        "protocol": "downstream_reference", "status": "running",
        "dataset": dataset, "method": None, "gram_variant": None,
        "reference": reference, "scenario": scenario,
        "evaluated_checkpoints": [], "settings": [reference], "expected_rows": 1,
        "dataset_config": dataset_config, "model_config": model_config,
        "downstream_initialization": "fresh_method_independent_seed",
        "downstream_seed": downstream_seed,
        "downstream_training_exposure_per_epoch": data.sample_count,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_config(output / "config.json", config)
    row = _result_row(
        data, labels, np.ones(data.sample_count, dtype=bool),
        np.zeros(data.sample_count, dtype=bool), noise_mask,
        setting=reference, checkpoint=0, budget=None if reference == "clean" else 0.0,
        threshold=None, model_config=model_config, seed=downstream_seed, data_root=data_root,
    )
    _write_csv(output / "downstream_metrics.csv", [row])
    config.update(status="complete", elapsed_seconds=time.perf_counter() - started)
    _write_config(output / "config.json", config)
    run_logger.info("downstream_reference setting=%s performance=%.6f",
                    reference, row["downstream_test_performance"])
    return output


def run_downstream_experiment(
    *, dataset: str, method: str, scenario: DownstreamScenario,
    data_root: str | Path, results_root: str | Path,
    gram_variant: str | None = None, beta: float | None = None,
    checkpoints: tuple[float, ...] | None = None,
    overwrite: bool = False,
) -> Path:
    """Evaluate each saved checkpoint without selecting or verifying samples."""
    scenario.validate()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if beta is not None and (method != "ours" or gram_variant is not None):
        raise ValueError("beta requires method=ours and no explicit gram_variant")
    if method == "ours" and gram_variant is None:
        choice = get_best_k(dataset, scenario.noise_type, scenario.rho,
                            1.0 if beta is None else beta, results_root=results_root)
        if scenario.seed not in {int(seed) for seed in choice["seeds"].split(",")}:
            raise ValueError(f"selected {choice['gram_variant']} lacks completed seed{scenario.seed}")
        gram_variant = choice["gram_variant"]
    variant = resolve_gram_variant(method, gram_variant)
    selected = validate_budget_checkpoints(scenario.budgets if checkpoints is None else checkpoints)
    if not set(selected).issubset(scenario.budgets):
        raise ValueError("requested checkpoints must be present in the saved detection schedule")
    data_root, results_root = Path(data_root), Path(results_root)
    output = downstream_output_dir(results_root, dataset, method, scenario, variant)
    ensure_output_available(output, overwrite)
    started = time.perf_counter()
    dataset_config, model_config = load_config(dataset)
    data = load_data(
        dataset, root=data_root, seed=scenario.seed,
        dataset_config=dataset_config, model_config=model_config,
    )
    noisy_labels, noise_mask = inject_noise(
        data.clean_train_y, data.noise_x, noise_type=scenario.noise_type,
        rho=scenario.rho, num_classes=data.num_classes, seed=scenario.seed,
    )
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    if not np.array_equal(noise_mask, noisy_labels != data.clean_train_y):
        raise RuntimeError("noise mask and labels disagree")
    # Recover and validate the complete score trajectory before any test evaluation.
    detection = ensure_detection_results(
        dataset=dataset, method=method,
        scenario=DetectionScenario(scenario.noise_type, scenario.rho, scenario.budgets, scenario.seed),
        data_root=data_root, results_root=results_root, gram_variant=variant,
        dataset_config=dataset_config, model_config=model_config,
        noisy_labels=noisy_labels, clean_labels=data.clean_train_y, data=data,
    )
    downstream_seed = int(np.random.SeedSequence([scenario.seed, 65_537]).generate_state(1)[0])
    config = {
        "protocol": "downstream_repair_otsu", "status": "running",
        "dataset": dataset, "method": method, "gram_variant": variant,
        "scenario": asdict(scenario), "evaluated_checkpoints": list(selected),
        "dataset_config": dataset_config, "model_config": model_config,
        "method_config": detection.config["method_config"],
        "detection_results_dir": str(detection.directory),
        "score_source": detection.score_source,
        "score_replay_train_seconds": detection.replay_train_seconds,
        "filtering": {"rule": "exact_otsu", "fit_pool": "full_training_set", "delete_if": "unverified_and_score_gt_threshold"},
        "settings": ["repair_only", "repair_otsu"],
        "expected_rows": 2 * len(selected),
        "downstream_initialization": "fresh_method_independent_seed",
        "downstream_seed": downstream_seed,
        "downstream_training_exposure_per_epoch": data.sample_count,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_config(output / "config.json", config)
    rows = []
    full = np.ones(data.sample_count, dtype=bool)

    def train_and_save(labels, retained, queried, **metadata):
        row = _result_row(
            data, labels, retained, queried, noise_mask, model_config=model_config,
            seed=downstream_seed, data_root=data_root, **metadata,
        )
        rows.append(row)
        _write_csv(output / "downstream_metrics.csv", rows)
        run_logger.info(
            "downstream_result setting=%s checkpoint=%d budget=%s retained=%d performance=%.6f",
            row["setting"], row["checkpoint"], row["configured_budget_fraction"],
            row["retained_count"], row["downstream_test_performance"],
        )

    for snapshot in detection.snapshots:
        budget = float(snapshot["configured_budget_fraction"])
        if budget not in selected:
            continue
        threshold = otsu_threshold(snapshot["score"])
        queried, labels = snapshot["queried"], snapshot["repaired_labels"]
        for setting, retained in (
            ("repair_only", full),
            ("repair_otsu", queried | (snapshot["score"] <= threshold)),
        ):
            train_and_save(
                labels, retained, queried, setting=setting, checkpoint=int(snapshot["checkpoint"]),
                budget=budget, threshold=threshold if setting == "repair_otsu" else None,
            )
    config.update(status="complete", elapsed_seconds=time.perf_counter() - started)
    _write_config(output / "config.json", config)
    # Remove only artifacts produced by the former downstream protocol on explicit overwrite.
    if overwrite:
        for name in ("summary.json", "run_args.json", "run.log", "queries.csv", "kernel_weights.csv",
                     "pseudo_labels.csv", "extrapolation_predictions.csv"):
            (output / name).unlink(missing_ok=True)
    return output
