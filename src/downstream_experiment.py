from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from config import load_config
from config.methods import method_config, resolve_gram_variant

from .detection import LabelVerificationOracle
from .baselines.training import train_robust_selector
from .methods import (
    CORRECTION_METHODS,
    METHODS,
    create_method_policy,
    method_feedback_mode,
    method_update_schedule,
    should_train_acquisition_model,
)
from .utils.budget import budget_schedule, validate_budget_checkpoints
from .utils.data import load_data
from .utils.noise import inject_noise
from .utils.results import round_metric_floats
from .utils.run_logging import run_logger
from .utils.training import train_downstream_model, train_task_model


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


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def downstream_output_dir(
    root: Path,
    dataset: str,
    method: str,
    scenario: DownstreamScenario,
    gram_variant: str | None = None,
) -> Path:
    output = (
        root
        / "downstream"
        / dataset
        / scenario.noise_type
        / f"rho{scenario.rho:g}"
        / method
    )
    selected_variant = resolve_gram_variant(method, gram_variant)
    if selected_variant is not None:
        output /= selected_variant
    return output / f"seed{scenario.seed}"


# Kept for callers that imported the previous private helper.
_output_dir = downstream_output_dir


def run_downstream_experiment(
    *,
    dataset: str,
    method: str,
    scenario: DownstreamScenario,
    data_root: str | Path,
    results_root: str | Path,
    gram_variant: str | None = None,
) -> Path:
    """Run a deployment experiment independently of detection results."""
    scenario.validate()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    selected_gram_variant = resolve_gram_variant(method, gram_variant)
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
    method_labels = noisy_labels.copy()
    repaired_labels = noisy_labels.copy()
    oracle = LabelVerificationOracle(noisy_labels, data.clean_train_y)
    noise_mask = oracle.ground_truth_mask()
    if not np.array_equal(injected_mask, noise_mask):
        raise RuntimeError("noise mask and verification oracle disagree")
    configured_budgets = validate_budget_checkpoints(scenario.budgets)
    target_counts, increments = budget_schedule(data.sample_count, configured_budgets)
    checkpoint_budgets = (0.0, *configured_budgets)
    run_logger.info(
        "data_loaded protocol=downstream dataset=%s modality=%s "
        "sample_count=%d num_classes=%d budget_checkpoints=%s",
        dataset,
        data.modality,
        data.sample_count,
        data.num_classes,
        ",".join(f"{value:g}" for value in configured_budgets),
    )

    queried = np.zeros(data.sample_count, dtype=bool)
    queried_status = np.zeros(data.sample_count, dtype=np.int64)
    snapshots: list[dict] = []
    rows: list[dict] = []
    query_rows: list[dict] = []
    kernel_weight_rows: list[dict] = []
    pseudo_rows: list[dict] = []
    policy = create_method_policy(
        method,
        noisy_labels,
        modality=data.modality,
        seed=scenario.seed,
        native_features=(data.train_x if data.modality == "tabular" else None),
        gram_variant=selected_gram_variant,
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
            acquisition_train_seconds = 0.0
        else:
            if method == "active_label_correction":
                artifacts = train_robust_selector(
                    data,
                    method_labels,
                    config=model_config,
                    seed=scenario.seed,
                    cache_dir=data_root / "huggingface",
                )
                acquisition_train_seconds = 0.0
                selector_train_seconds = artifacts.train_seconds
            else:
                transition, trusted = policy.training_transition(
                    previous_artifacts, method_labels, queried
                )
                artifacts = train_task_model(
                    data,
                    method_labels,
                    config=model_config,
                    seed=scenario.seed,
                    cache_dir=data_root / "huggingface",
                    evaluate_test=False,
                    noise_transition=transition,
                    trusted_mask=trusted,
                )
                acquisition_train_seconds = artifacts.train_seconds
            cached_artifacts = artifacts
            previous_artifacts = artifacts

        prediction_started = time.perf_counter()
        prediction = policy.predict(
            artifacts, method_labels, queried, queried_status
        )
        kernel_weight_state = None
        if method == "ours":
            kernel_weight_state = policy.kernel_weight_state()
            kernel_weight_rows.append(
                {
                    "checkpoint": checkpoint_index,
                    "configured_budget_fraction": configured_budget,
                    "queried_count": int(queried.sum()),
                    **kernel_weight_state,
                }
            )
        global_score = prediction.score
        predicted_issue = prediction.predicted_issue
        decision_rule = prediction.decision_rule
        global_score = np.asarray(global_score, dtype=np.float64)
        predicted_issue = np.asarray(predicted_issue, dtype=bool)
        if global_score.shape != (data.sample_count,) or predicted_issue.shape != (
            data.sample_count,
        ):
            raise RuntimeError("method did not extrapolate one decision per sample")
        if not np.all(np.isfinite(global_score)):
            raise RuntimeError("method produced a non-finite global score")

        retained = queried | (~queried & ~predicted_issue)
        prediction_seconds = time.perf_counter() - prediction_started
        queried_count = int(queried.sum())
        found = int(queried_status[queried].sum())
        if checkpoint_index > 0:
            snapshots.append(
                {
                    "checkpoint": checkpoint_index,
                    "configured_budget_fraction": configured_budget,
                    "queried": queried.copy(),
                    "repaired_labels": repaired_labels.copy(),
                    "global_score": global_score.copy(),
                    "predicted_issue": predicted_issue.copy(),
                    "retained": retained.copy(),
                    "queried_count": queried_count,
                    "found": found,
                    "acquisition_train_seconds": acquisition_train_seconds,
                    "acquisition_model_reused": model_reused,
                    "selector_train_seconds": selector_train_seconds,
                    "prediction_seconds": prediction_seconds,
                }
            )

        if checkpoint_index == len(configured_budgets):
            continue
        query_seed = int(
            np.random.SeedSequence(
                [scenario.seed, checkpoint_index, 7919]
            ).generate_state(1)[0]
        )
        query_count = int(increments[checkpoint_index])
        result = policy.select(
            artifacts,
            method_labels,
            queried,
            queried_status,
            count=query_count,
            seed=query_seed,
        )
        selected = np.asarray(result.selected, dtype=np.int64)
        if len(selected) != query_count or len(np.unique(selected)) != len(selected):
            raise RuntimeError("query policy returned an invalid batch")
        if np.any(queried[selected]):
            raise RuntimeError("query policy selected an already verified sample")
        pseudo_selected = (
            np.empty(0, dtype=np.int64)
            if result.pseudo_selected is None
            else np.asarray(result.pseudo_selected, dtype=np.int64)
        )
        if np.intersect1d(selected, pseudo_selected).size or np.any(
            queried[pseudo_selected]
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
        verification = oracle.verify(selected)
        for sample_id, verified_label, status in zip(
            verification.positions,
            verification.true_labels,
            verification.mislabel_status,
        ):
            query_row = {
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
            if kernel_weight_state is not None:
                query_row.update(
                    {
                        "repair_value": float(result.repair_value[sample_id]),
                        "informative_value": (
                            float(result.informative_value[sample_id])
                            if result.informative_value is not None else None
                        ),
                        "omega_margin": kernel_weight_state["omega_margin"],
                        "omega_gradient": kernel_weight_state["omega_gradient"],
                        "omega_identity": kernel_weight_state["omega_identity"],
                    }
                )
            query_rows.append(query_row)
        queried[selected] = True
        queried_status[selected] = verification.mislabel_status
        repaired_labels[selected] = verification.true_labels
        if method in CORRECTION_METHODS:
            method_labels[selected] = verification.true_labels
        policy.register_pseudo_labels(
            result.pseudo_selected,
            pseudo_prediction,
            method_labels,
        )
        run_logger.info(
            "verification_checkpoint_completed protocol=downstream checkpoint=%d "
            "configured_budget_fraction=%g batch_size=%d batch_mislabels=%d "
            "cumulative_queried=%d cumulative_mislabels=%d",
            checkpoint_index + 1,
            configured_budgets[checkpoint_index],
            len(selected),
            int(verification.mislabel_status.sum()),
            int(queried.sum()),
            int(queried_status[queried].sum()),
        )

    # Only after the complete acquisition trajectory do downstream models read
    # the clean test set. Their metrics cannot affect later queries.
    for snapshot in snapshots:
        retained = snapshot["retained"]
        retained_ids = np.flatnonzero(retained)
        snapshot_labels = snapshot["repaired_labels"]
        downstream_accuracy = None
        downstream_macro_f1 = None
        downstream_performance = None
        downstream_train_seconds = 0.0
        if len(retained_ids):
            downstream_seed = int(
                np.random.SeedSequence([scenario.seed, 65_537]).generate_state(1)[0]
            )
            downstream = train_downstream_model(
                data,
                snapshot_labels,
                retained_ids,
                config=model_config,
                seed=downstream_seed,
                cache_dir=data_root / "huggingface",
            )
            downstream_accuracy = downstream.test_accuracy
            downstream_macro_f1 = downstream.test_macro_f1
            downstream_performance = (
                downstream.test_accuracy
                if data.test_metric == "accuracy"
                else downstream.test_macro_f1
            )
            downstream_train_seconds = downstream.train_seconds
        retained_label_accuracy = (
            float((snapshot_labels[retained] == data.clean_train_y[retained]).mean())
            if retained.any()
            else None
        )
        queried_count = snapshot["queried_count"]
        found = snapshot["found"]
        rows.append(
            {
                "checkpoint": snapshot["checkpoint"],
                "configured_budget_fraction": snapshot[
                    "configured_budget_fraction"
                ],
                "verification_budget": queried_count,
                "verification_fraction": queried_count / data.sample_count,
                "queried_mislabels_found": found,
                "query_precision": found / queried_count if queried_count else 0.0,
                "predicted_issue_count_unqueried": int(
                    (
                        snapshot["predicted_issue"]
                        & ~snapshot["queried"]
                    ).sum()
                ),
                "retained_count": int(retained.sum()),
                "retained_fraction": float(retained.mean()),
                "diagnostic_retained_label_accuracy": retained_label_accuracy,
                "downstream_test_metric": data.test_metric,
                "downstream_test_performance": downstream_performance,
                "downstream_test_accuracy": downstream_accuracy,
                "downstream_test_macro_f1": downstream_macro_f1,
                "acquisition_train_seconds": snapshot[
                    "acquisition_train_seconds"
                ],
                "acquisition_model_reused": snapshot[
                    "acquisition_model_reused"
                ],
                "selector_train_seconds": snapshot["selector_train_seconds"],
                "prediction_seconds": snapshot["prediction_seconds"],
                "downstream_train_seconds": downstream_train_seconds,
            }
        )
        run_logger.info(
            "downstream_checkpoint_completed checkpoint=%d "
            "configured_budget_fraction=%g retained_count=%d "
            "test_metric=%s test_performance=%s train_seconds=%.3f",
            snapshot["checkpoint"],
            snapshot["configured_budget_fraction"],
            int(retained.sum()),
            data.test_metric,
            downstream_performance,
            downstream_train_seconds,
        )

    first_row = rows[0]
    final_row = rows[-1]
    feedback_mode = method_feedback_mode(method)
    update_schedule = method_update_schedule(method)
    final_snapshot = snapshots[-1]
    final_global_score = final_snapshot["global_score"]
    final_predicted_issue = final_snapshot["predicted_issue"]
    final_retained = final_snapshot["retained"]
    prediction_rows = [
        {
            "sample_id": sample_id,
            "queried": int(queried[sample_id]),
            "mislabel_score_at_max_budget": float(final_global_score[sample_id]),
            "predicted_issue_at_max_budget": int(final_predicted_issue[sample_id]),
            "retained_for_downstream": int(final_retained[sample_id]),
            "noisy_label": int(noisy_labels[sample_id]),
            "downstream_training_label": (
                int(repaired_labels[sample_id]) if final_retained[sample_id] else None
            ),
            "label_source": (
                "verified" if queried[sample_id] else "noisy_predicted_correct"
            )
            if final_retained[sample_id]
            else "filtered_predicted_issue",
        }
        for sample_id in range(data.sample_count)
    ]
    summary = {
        "protocol": "downstream_full_pool",
        "reuses_detection_results": False,
        "dataset": dataset,
        "method": method,
        "gram_variant": selected_gram_variant,
        **asdict(scenario),
        "feedback_mode": feedback_mode,
        "update_schedule": update_schedule,
        "binary_decision_rule": decision_rule,
        "sample_count": data.sample_count,
        "configured_budget_checkpoints": list(configured_budgets),
        "configured_min_budget_fraction": configured_budgets[0],
        "configured_max_budget_fraction": configured_budgets[-1],
        "actual_budget": int(queried.sum()),
        "actual_budget_fraction": float(queried.mean()),
        "internal_pseudo_label_count_at_max_budget": int(policy.pseudo_mask.sum()),
        "retained_count_at_max_budget": final_row["retained_count"],
        "retained_fraction_at_max_budget": final_row["retained_fraction"],
        "downstream_test_metric": data.test_metric,
        "downstream_test_performance_at_min_budget": first_row[
            "downstream_test_performance"
        ],
        "downstream_test_performance_at_max_budget": final_row[
            "downstream_test_performance"
        ],
        "elapsed_seconds": time.perf_counter() - total_started,
    }
    if kernel_weight_rows:
        summary.update(
            {
                "final_omega_margin": kernel_weight_rows[-1]["omega_margin"],
                "final_omega_gradient": kernel_weight_rows[-1]["omega_gradient"],
                "final_omega_identity": kernel_weight_rows[-1]["omega_identity"],
            }
        )
    output = downstream_output_dir(
        results_root, dataset, method, scenario, selected_gram_variant
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "downstream_metrics.csv", round_metric_floats(rows))
    _write_csv(output / "queries.csv", query_rows)
    _write_csv(output / "kernel_weights.csv", kernel_weight_rows)
    _write_csv(output / "pseudo_labels.csv", pseudo_rows)
    _write_csv(output / "extrapolation_predictions.csv", prediction_rows)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            round_metric_floats(summary),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "protocol": "downstream_full_pool",
                "reuses_detection_results": False,
                "dataset": dataset,
                "method": method,
                "gram_variant": selected_gram_variant,
                "scenario": asdict(scenario),
                "dataset_config": dataset_config,
                "model_config": model_config,
                "method_config": method_config(method, selected_gram_variant),
                "feedback_mode": feedback_mode,
                "update_schedule": update_schedule,
                "binary_decision_rule": decision_rule,
                "retention_rule": "queried_corrected_or_unqueried_predicted_correct",
                "includes_unverified_checkpoint": False,
                "downstream_initialization": "fresh_method_independent_seed",
                "downstream_runs_per_checkpoint": 1,
                "downstream_training_exposure_per_epoch": data.sample_count,
                "budget_unit": "fraction_of_full_training_set",
                "budget_checkpoint_target_counts": target_counts[1:].tolist(),
                "budget_checkpoint_query_increments": increments.tolist(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    return output
