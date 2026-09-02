from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from config import load_config
from config.methods import METHOD_CONFIGS

from .detection import DetectionMetrics, LabelVerificationOracle, evaluate_detection
from .baselines.training import train_robust_selector
from .methods import (
    CORRECTION_METHODS,
    METHODS,
    create_method_policy,
    method_feedback_mode,
    method_update_schedule,
    should_train_acquisition_model,
)
from .utils.budget import budget_schedule, budget_tag, validate_budget_checkpoints
from .utils.data import load_data
from .utils.noise import inject_noise
from .utils.run_logging import run_logger
from .utils.training import train_task_model


@dataclass(frozen=True)
class DetectionScenario:
    noise_type: str
    rho: float
    budgets: tuple[float, ...]
    seed: int

    def validate(self) -> None:
        if self.noise_type not in {"symmetric", "pairflip", "instance"}:
            raise ValueError("noise_type must be symmetric, pairflip, or instance")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must be in (0, 1)")
        budgets = validate_budget_checkpoints(self.budgets)
        if budgets[-1] >= 1.0:
            raise ValueError("maximum budget must leave a non-empty unverified set")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def detection_output_dir(
    root: Path, dataset: str, method: str, scenario: DetectionScenario
) -> Path:
    return (
        root
        / "detection"
        / dataset
        / scenario.noise_type
        / f"rho{scenario.rho:g}_B{budget_tag(tuple(scenario.budgets))}"
        / method
        / f"seed{scenario.seed}"
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _metric_fields(
    prefix: str, metrics: DetectionMetrics | None = None
) -> dict[str, float | int | None]:
    names = ("auprc", "auroc")
    return {
        f"{prefix}_{name}": None if metrics is None else getattr(metrics, name)
        for name in names
    }


def _normalized_curve_area(
    values: np.ndarray, budgets: np.ndarray, max_budget: float
) -> float:
    if not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.trapz(values, budgets) / max_budget)


def run_detection_experiment(
    *,
    dataset: str,
    method: str,
    scenario: DetectionScenario,
    data_root: str | Path,
    results_root: str | Path,
) -> Path:
    scenario.validate()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    dataset_config, model_config = load_config(dataset)
    data_root, results_root = Path(data_root), Path(results_root)
    data = load_data(
        dataset,
        root=data_root,
        seed=scenario.seed,
        dataset_config=dataset_config,
        model_config=model_config,
    )
    noisy_labels, injected_mask = inject_noise(
        data.clean_train_y,
        data.noise_x,
        noise_type=scenario.noise_type,
        rho=scenario.rho,
        num_classes=data.num_classes,
        seed=scenario.seed,
    )
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    training_labels = noisy_labels.copy()
    oracle = LabelVerificationOracle(noisy_labels, data.clean_train_y)
    configured_budgets = validate_budget_checkpoints(scenario.budgets)
    target_counts, increments = budget_schedule(data.sample_count, configured_budgets)
    checkpoint_budgets = (0.0, *configured_budgets)
    run_logger.info(
        "data_loaded protocol=full_pool_detection dataset=%s modality=%s sample_count=%d "
        "num_classes=%d budget_checkpoints=%s",
        dataset,
        data.modality,
        data.sample_count,
        data.num_classes,
        ",".join(f"{value:g}" for value in configured_budgets),
    )

    queried = np.zeros(data.sample_count, dtype=bool)
    queried_status = np.zeros(data.sample_count, dtype=np.int64)
    rows: list[dict] = []
    query_rows: list[dict] = []
    pseudo_rows: list[dict] = []
    detection_snapshots: list[dict] = []
    policy = create_method_policy(
        method,
        noisy_labels,
        modality=data.modality,
        seed=scenario.seed,
        native_features=(data.train_x if data.modality == "tabular" else None),
    )
    cached_artifacts = None
    previous_artifacts = None
    total_started = time.perf_counter()

    for checkpoint_index, configured_budget in enumerate(checkpoint_budgets):
        train_now = should_train_acquisition_model(method, checkpoint_index)
        model_reused = not train_now
        selector_train_seconds = 0.0
        if model_reused:
            if cached_artifacts is None:
                raise RuntimeError("acquisition model requested before initial training")
            artifacts = cached_artifacts
            round_train_seconds = 0.0
        else:
            if method == "active_label_correction":
                artifacts = train_robust_selector(
                    data,
                    training_labels,
                    config=model_config,
                    seed=scenario.seed,
                    cache_dir=data_root / "huggingface",
                )
                round_train_seconds = 0.0
                selector_train_seconds = artifacts.train_seconds
            else:
                transition, trusted = policy.training_transition(
                    previous_artifacts, training_labels, queried
                )
                artifacts = train_task_model(
                    data,
                    training_labels,
                    config=model_config,
                    seed=scenario.seed,
                    cache_dir=data_root / "huggingface",
                    evaluate_test=False,
                    noise_transition=transition,
                    trusted_mask=trusted,
                )
                round_train_seconds = artifacts.train_seconds
            cached_artifacts = artifacts
            previous_artifacts = artifacts

        prediction_started = time.perf_counter()
        prediction = policy.predict(
            artifacts, training_labels, queried, queried_status
        )
        global_score = np.asarray(prediction.score, dtype=np.float64)
        expected_shape = (data.sample_count,)
        if global_score.shape != expected_shape or not np.all(
            np.isfinite(global_score)
        ):
            raise RuntimeError("method did not produce one finite global score per sample")
        row = None
        if checkpoint_index > 0:
            detection_snapshots.append(
                {
                    "score": global_score.copy(),
                    "queried": queried.copy(),
                }
            )
            queried_count = int(queried.sum())
            found = int(queried_status[queried].sum())
            row = {
                "checkpoint": checkpoint_index,
                "configured_budget_fraction": configured_budget,
                "verification_budget": queried_count,
                "verification_fraction": queried_count / data.sample_count,
                "unverified_count": data.sample_count - queried_count,
                "unverified_fraction": 1.0 - queried_count / data.sample_count,
                **_metric_fields("full"),
                **_metric_fields("unverified"),
                "queried_mislabels_found": found,
                "query_precision": found / queried_count if queried_count else 0.0,
                "train_seconds": round_train_seconds,
                "task_model_reused": model_reused,
                "selector_train_seconds": selector_train_seconds,
                "prediction_and_query_seconds": 0.0,
            }
            rows.append(row)

        if checkpoint_index == len(configured_budgets):
            if row is not None:
                row["prediction_and_query_seconds"] = (
                    time.perf_counter() - prediction_started
                )
            continue
        query_seed = int(
            np.random.SeedSequence(
                [scenario.seed, checkpoint_index, 7919]
            ).generate_state(1)[0]
        )
        query_count = int(increments[checkpoint_index])
        result = policy.select(
            artifacts,
            training_labels,
            queried,
            queried_status,
            count=query_count,
            seed=query_seed,
        )
        selected = np.asarray(result.selected, dtype=np.int64)
        if len(selected) != query_count or len(np.unique(selected)) != len(selected):
            raise RuntimeError("query policy returned an invalid batch")
        if np.any(queried[selected]):
            raise RuntimeError("query policy selected an ineligible sample")
        pseudo_selected = (
            np.empty(0, dtype=np.int64)
            if result.pseudo_selected is None
            else np.asarray(result.pseudo_selected, dtype=np.int64)
        )
        if (
            np.intersect1d(selected, pseudo_selected).size
            or np.any(queried[pseudo_selected])
        ):
            raise RuntimeError("method produced an ineligible internal pseudo label")
        pseudo_prediction = np.asarray(artifacts.probability).argmax(axis=1)
        for sample_id in pseudo_selected:
            pseudo_rows.append(
                {
                    "budget_checkpoint": checkpoint_index + 1,
                    "configured_budget_fraction": configured_budgets[checkpoint_index],
                    "sample_id": int(sample_id),
                    "noisy_label": int(noisy_labels[sample_id]),
                    "pseudo_label": int(pseudo_prediction[sample_id]),
                }
            )
        if row is not None:
            row["prediction_and_query_seconds"] = (
                time.perf_counter() - prediction_started
            )
        verification = oracle.verify(selected)
        for sample_id, verified_label, status in zip(
            verification.positions,
            verification.true_labels,
            verification.mislabel_status,
        ):
            query_rows.append(
                {
                    "budget_checkpoint": checkpoint_index + 1,
                    "configured_budget_fraction": configured_budgets[checkpoint_index],
                    "sample_id": int(sample_id),
                    "was_mislabeled": int(status),
                    "noisy_label": int(noisy_labels[sample_id]),
                    "verified_label": int(verified_label),
                    "feedback_used_by_method": method_feedback_mode(method),
                    "acquisition_score": float(result.score[sample_id]),
                    "posterior_variance": (
                        float(result.posterior_variance[sample_id])
                        if result.posterior_variance is not None
                        else None
                    ),
                }
            )
        queried[selected] = True
        queried_status[selected] = verification.mislabel_status
        if method in CORRECTION_METHODS:
            training_labels[selected] = verification.true_labels
        policy.register_pseudo_labels(
            result.pseudo_selected,
            pseudo_prediction,
            training_labels,
        )
        run_logger.info(
            "verification_checkpoint_completed protocol=full_pool_detection checkpoint=%d "
            "configured_budget_fraction=%g batch_size=%d batch_mislabels=%d "
            "cumulative_queried=%d cumulative_mislabels=%d",
            checkpoint_index + 1,
            configured_budgets[checkpoint_index],
            len(selected),
            int(verification.mislabel_status.sum()),
            int(queried.sum()),
            int(queried_status[queried].sum()),
        )

    # The complete correctness mask is consulted only after the query trajectory.
    # During acquisition, the oracle reveals correctness only for queried samples.
    noise_mask = oracle.ground_truth_mask()
    if not np.array_equal(injected_mask, noise_mask):
        raise RuntimeError("noise mask and verification oracle disagree")
    full_status = noise_mask.astype(np.int64)
    full_metrics: list[DetectionMetrics] = []
    unverified_metrics: list[DetectionMetrics] = []
    for row, snapshot in zip(rows, detection_snapshots):
        full = evaluate_detection(
            full_status,
            snapshot["score"],
        )
        unverified = ~snapshot["queried"]
        remaining = evaluate_detection(
            full_status[unverified],
            snapshot["score"][unverified],
        )
        full_metrics.append(full)
        unverified_metrics.append(remaining)
        row.update(_metric_fields("full", full))
        row.update(_metric_fields("unverified", remaining))
        run_logger.info(
            "detection_checkpoint_evaluated checkpoint=%d "
            "configured_budget_fraction=%g full_auprc=%s full_auroc=%s "
            "unverified_auprc=%s unverified_auroc=%s",
            row["checkpoint"],
            row["configured_budget_fraction"],
            full.auprc,
            full.auroc,
            remaining.auprc,
            remaining.auroc,
        )

    actual_budgets = np.asarray(
        [row["verification_fraction"] for row in rows], dtype=np.float64
    )
    full_auprc_curve = np.asarray([item.auprc for item in full_metrics])
    unverified_auprc_curve = np.asarray(
        [item.auprc for item in unverified_metrics]
    )
    final_snapshot = detection_snapshots[-1]
    prediction_rows = [
        {
            "sample_id": sample_id,
            "queried": int(final_snapshot["queried"][sample_id]),
            "noisy_label": int(noisy_labels[sample_id]),
            "was_mislabeled": int(noise_mask[sample_id]),
            "mislabel_score_at_max_budget": float(
                final_snapshot["score"][sample_id]
            ),
            "evaluation_subset": (
                "verified"
                if final_snapshot["queried"][sample_id]
                else "unverified"
            ),
        }
        for sample_id in range(data.sample_count)
    ]
    feedback_mode = method_feedback_mode(method)
    update_schedule = method_update_schedule(method)
    summary = {
        "protocol": "full_pool_detection",
        "reuses_downstream_results": False,
        "dataset": dataset,
        "method": method,
        **asdict(scenario),
        "feedback_mode": feedback_mode,
        "update_schedule": update_schedule,
        "primary_evaluation_set": "unverified_training_set",
        "sample_count": data.sample_count,
        "injected_mislabel_count": int(noise_mask.sum()),
        "actual_budget": int(queried.sum()),
        "actual_budget_fraction": float(queried.mean()),
        "unverified_count_at_max_budget": int((~queried).sum()),
        "internal_pseudo_label_count_at_max_budget": int(policy.pseudo_mask.sum()),
        "aubc_full_auprc": _finite_or_none(
            _normalized_curve_area(
                full_auprc_curve, actual_budgets, actual_budgets[-1]
            )
        ),
        "aubc_unverified_auprc": _finite_or_none(
            _normalized_curve_area(
                unverified_auprc_curve, actual_budgets, actual_budgets[-1]
            )
        ),
        "diagnostic_query_precision_at_max_budget": float(rows[-1]["query_precision"]),
        "elapsed_seconds": time.perf_counter() - total_started,
    }
    for subset, metrics in (
        ("full", full_metrics[-1]),
        ("unverified", unverified_metrics[-1]),
    ):
        summary[f"{subset}_auprc_at_max_budget"] = _finite_or_none(metrics.auprc)
        summary[f"{subset}_auroc_at_max_budget"] = _finite_or_none(metrics.auroc)
    output = detection_output_dir(results_root, dataset, method, scenario)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "metrics.csv", rows)
    _write_csv(output / "queries.csv", query_rows)
    _write_csv(output / "pseudo_labels.csv", pseudo_rows)
    _write_csv(output / "predictions.csv", prediction_rows)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "protocol": "full_pool_detection",
                "reuses_downstream_results": False,
                "dataset": dataset,
                "method": method,
                "scenario": asdict(scenario),
                "dataset_config": dataset_config,
                "model_config": model_config,
                "method_config": METHOD_CONFIGS.get(method, {}),
                "feedback_mode": feedback_mode,
                "update_schedule": update_schedule,
                "primary_evaluation_set": "unverified_training_set",
                "evaluation_sets": ["full_training_set", "unverified_training_set"],
                "ground_truth_usage": "offline_metrics_after_complete_query_trajectory",
                "includes_zero_budget_checkpoint": False,
                "budget_unit": "fraction_of_full_training_set",
                "budget_checkpoint_target_counts": target_counts[1:].tolist(),
                "budget_checkpoint_query_increments": increments.tolist(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    return output
