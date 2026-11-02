"""Read detection scores, replaying saved feedback when snapshots are absent.

Never selects samples, calls an oracle, or writes to detection results. Legacy
warm-start baselines replay classifier updates using the original query logs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from config.methods import method_config
from .baselines.training import train_robust_selector
from .detection_experiment import DetectionScenario, detection_output_dir
from .methods import CORRECTION_METHODS, WARM_START_METHODS, create_method_policy
from .utils.budget import budget_schedule
from .utils.detection_results import DETECTION_SNAPSHOTS_FILE, load_detection_snapshots
from .utils.run_logging import run_logger
from .utils.training import seed_everything, train_task_model, warm_start_training_options
from .utils.training_cache import load_saved_training_artifact


@dataclass
class DetectionResults:
    directory: Path
    config: dict
    summary: dict
    snapshots: list[dict]
    score_source: str
    replay_train_seconds: float = 0.0


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _query_batches(output, scenario, noisy_labels, clean_labels):
    targets, increments = budget_schedule(len(noisy_labels), scenario.budgets)
    batches = [[] for _ in scenario.budgets]
    seen = set()
    for row in _read_csv(output / "queries.csv"):
        checkpoint, sample_id = int(row["budget_checkpoint"]), int(row["sample_id"])
        if not 1 <= checkpoint <= len(batches):
            raise ValueError("query log contains an unknown checkpoint")
        if not 0 <= sample_id < len(noisy_labels) or sample_id in seen:
            raise ValueError("query log contains an invalid or duplicate sample")
        if (
            float(row["configured_budget_fraction"]) != scenario.budgets[checkpoint - 1]
            or int(row["noisy_label"]) != noisy_labels[sample_id]
            or int(row["verified_label"]) != clean_labels[sample_id]
            or int(row["was_mislabeled"]) != int(noisy_labels[sample_id] != clean_labels[sample_id])
        ):
            raise ValueError("saved expert feedback does not match the current scenario")
        seen.add(sample_id)
        batches[checkpoint - 1].append(row)
    if [len(batch) for batch in batches] != increments.tolist():
        raise ValueError("saved query counts do not match the budget checkpoints")
    metrics = _read_csv(output / "metrics.csv")
    if len(metrics) != len(batches) or any(
        int(row["checkpoint"]) != index
        or float(row["configured_budget_fraction"]) != scenario.budgets[index - 1]
        or int(row["verification_budget"]) != targets[index]
        for index, row in enumerate(metrics, 1)
    ):
        raise ValueError("saved detection metrics do not match the query checkpoints")
    return batches, targets


def _final_scores(output, noisy_labels, batches):
    rows = _read_csv(output / "predictions.csv")
    n = len(noisy_labels)
    if len(rows) != n:
        raise ValueError("saved final predictions are incomplete")
    score, seen = np.empty(n, dtype=np.float64), set()
    queried = {int(row["sample_id"]) for batch in batches for row in batch}
    for row in rows:
        sample_id = int(row["sample_id"])
        if not 0 <= sample_id < n or sample_id in seen:
            raise ValueError("saved final predictions contain invalid or duplicate IDs")
        if (
            int(row["noisy_label"]) != noisy_labels[sample_id]
            or int(row["queried"]) != int(sample_id in queried)
        ):
            raise ValueError("saved predictions and query log disagree")
        seen.add(sample_id)
        score[sample_id] = float(row["mislabel_score_at_max_budget"])
    if not np.all(np.isfinite(score)):
        raise ValueError("saved final scores must be finite")
    return score


def _pseudo_batches(output, scenario, noisy_labels, query_batches, expected_count, num_classes):
    batches = [[] for _ in scenario.budgets]
    path = output / "pseudo_labels.csv"
    if not path.exists():
        if expected_count:
            raise FileNotFoundError(f"saved pseudo-label log is required: {path}")
        return batches
    queried, pseudo = set(), set()
    for row in _read_csv(path):
        checkpoint = int(row["budget_checkpoint"])
        if not 1 <= checkpoint <= len(batches):
            raise ValueError("pseudo-label log contains an unknown checkpoint")
        batches[checkpoint - 1].append(row)
    for index, batch in enumerate(batches):
        current_queries = {int(row["sample_id"]) for row in query_batches[index]}
        for row in batch:
            sample_id = int(row["sample_id"])
            if (
                not 0 <= sample_id < len(noisy_labels)
                or sample_id in queried | current_queries | pseudo
                or int(row["noisy_label"]) != noisy_labels[sample_id]
                or float(row["configured_budget_fraction"]) != scenario.budgets[index]
                or not 0 <= int(row["pseudo_label"]) < num_classes
            ):
                raise ValueError("saved pseudo-label log is inconsistent")
            pseudo.add(sample_id)
        queried |= current_queries
    if len(pseudo) != expected_count:
        raise ValueError("saved pseudo-label log and detection summary disagree")
    return batches


def _base_artifact(config, results_root, noisy_labels, load_model_state):
    protocol = config.get("training_protocol", {})
    fingerprint = protocol.get("shared_training_artifact_fingerprint")
    if not fingerprint:
        raise ValueError("legacy score recovery requires the saved base-training fingerprint")
    scenario = config["scenario"]
    relative = (
        results_root / "detection" / "_shared_training_artifacts" / config["dataset"]
        / scenario["noise_type"] / f"rho{scenario['rho']:g}" / f"seed{scenario['seed']}"
        / fingerprint[:16]
    )
    saved = protocol.get("shared_training_artifact_directory")
    directory = relative if relative.is_dir() or not saved else Path(saved)
    return load_saved_training_artifact(
        directory, fingerprint=fingerprint, load_model_state=load_model_state,
        expected_metadata={
            "dataset": config["dataset"], "noise_type": scenario["noise_type"],
            "rho": scenario["rho"], "seed": scenario["seed"],
            "dataset_config": config["dataset_config"], "model_config": config["model_config"],
            "noisy_label_count": len(noisy_labels),
            "noisy_labels_sha256": hashlib.sha256(noisy_labels.tobytes()).hexdigest(),
        },
    )


def _replay_scores(output, config, summary, scenario, data, noisy_labels,
                   batches, data_root, results_root):
    method, model_config = config["method"], config["model_config"]
    pseudo_batches = _pseudo_batches(
        output, scenario, noisy_labels, batches,
        summary.get("internal_pseudo_label_count_at_max_budget", 0), data.num_classes,
    )
    seed_everything(scenario.seed)
    policy = create_method_policy(
        method, noisy_labels, modality=data.modality, seed=scenario.seed,
        native_features=data.train_x if data.modality == "tabular" else None,
        gram_variant=config.get("gram_variant"),
    )
    training_labels, repaired_labels = noisy_labels.copy(), noisy_labels.copy()
    queried = np.zeros(len(noisy_labels), dtype=bool)
    status = np.zeros(len(noisy_labels), dtype=np.int64)
    train_seconds = 0.0
    if method == "active_label_correction":
        artifacts = train_robust_selector(
            data, training_labels, config=model_config, seed=scenario.seed,
            cache_dir=data_root / "huggingface",
        )
        train_seconds += artifacts.train_seconds
    else:
        artifacts = _base_artifact(config, results_root, noisy_labels, method in WARM_START_METHODS)
    total_epochs = artifacts.training_epochs
    # Reproduce the cold posterior and all updates in their original order.
    policy.predict(artifacts, training_labels, queried, status)
    snapshots = []
    for index, (budget, batch) in enumerate(zip(scenario.budgets, batches), 1):
        ids = np.array([int(row["sample_id"]) for row in batch], dtype=np.int64)
        labels = np.array([int(row["verified_label"]) for row in batch], dtype=np.int64)
        queried[ids] = True
        status[ids] = labels != noisy_labels[ids]
        repaired_labels[ids] = labels
        if method in CORRECTION_METHODS:
            training_labels[ids] = labels
        pseudo = pseudo_batches[index - 1]
        if pseudo:
            pseudo_ids = np.array([int(row["sample_id"]) for row in pseudo], dtype=np.int64)
            pseudo_labels = training_labels.copy()
            pseudo_labels[pseudo_ids] = [int(row["pseudo_label"]) for row in pseudo]
            policy.register_pseudo_labels(pseudo_ids, pseudo_labels, training_labels)
        if method in WARM_START_METHODS:
            epochs, learning_rate = warm_start_training_options(model_config)
            common = dict(
                config=model_config, seed=scenario.seed, cache_dir=data_root / "huggingface",
                initial_model_state=artifacts.model_state, training_epochs=epochs,
                learning_rate=learning_rate, epoch_offset=total_epochs,
            )
            if method == "active_label_correction":
                artifacts = train_robust_selector(
                    data, training_labels, initial_contribution=artifacts.contribution, **common,
                )
            else:
                transition, trusted = policy.training_transition(artifacts, training_labels, queried)
                artifacts = train_task_model(
                    data, training_labels, evaluate_test=False, capture_model_state=True,
                    noise_transition=transition, trusted_mask=trusted, **common,
                )
            train_seconds += artifacts.train_seconds
            total_epochs += artifacts.training_epochs
        prediction = policy.predict(artifacts, training_labels, queried, status)
        score = np.asarray(prediction.score, dtype=np.float64)
        if score.shape != noisy_labels.shape or not np.all(np.isfinite(score)):
            raise ValueError("replayed method did not produce a finite score for every sample")
        snapshots.append({
            "checkpoint": index, "configured_budget_fraction": budget,
            "score": score.copy(), "queried": queried.copy(),
            "repaired_labels": repaired_labels.copy(),
        })
        run_logger.info("detection_scores_replayed checkpoint=%d budget=%g queried=%d", index, budget, queried.sum())
    return snapshots, train_seconds


def ensure_detection_results(
    *, dataset, method, scenario, data_root, results_root, gram_variant,
    dataset_config, model_config, noisy_labels, clean_labels, data=None,
) -> DetectionResults:
    """Read matching results; recover missing snapshots without re-querying."""
    output = detection_output_dir(Path(results_root), dataset, method, scenario, gram_variant)
    with (output / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    with (output / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    expected = json.loads(json.dumps({
        "protocol": "full_pool_detection", "dataset": dataset, "method": method,
        "gram_variant": gram_variant, "scenario": asdict(scenario),
        "dataset_config": dataset_config, "model_config": model_config,
        "method_config": method_config(method, gram_variant),
    }))
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"saved detection configuration mismatch: {name}: {output}")
    batches, targets = _query_batches(output, scenario, noisy_labels, clean_labels)
    if summary["sample_count"] != len(noisy_labels) or summary["actual_budget"] != targets[-1]:
        raise ValueError("saved detection summary does not match the query log")
    final_score = _final_scores(output, noisy_labels, batches)
    replay_seconds = 0.0
    if (output / DETECTION_SNAPSHOTS_FILE).exists():
        snapshots = load_detection_snapshots(
            output, budgets=scenario.budgets, noisy_labels=noisy_labels,
            clean_labels=clean_labels, target_counts=targets[1:],
        )
        queried = np.zeros(len(noisy_labels), dtype=bool)
        for snapshot, batch in zip(snapshots, batches):
            queried[[int(row["sample_id"]) for row in batch]] = True
            if not np.array_equal(queried, snapshot["queried"]):
                raise ValueError("saved snapshots and query log disagree")
        source = "saved_snapshots"
    else:
        if data is None:
            raise ValueError("task data is required to recover legacy checkpoint scores")
        run_logger.info("recovering_legacy_scores method=%s source=%s", method, output)
        snapshots, replay_seconds = _replay_scores(
            output, config, summary, scenario, data, noisy_labels, batches,
            Path(data_root), Path(results_root),
        )
        source = "saved_query_replay"
    if not np.allclose(snapshots[-1]["score"], final_score, rtol=1e-5, atol=1e-7):
        error = float(np.max(np.abs(snapshots[-1]["score"] - final_score)))
        raise ValueError(
            f"checkpoint scores do not reproduce saved final predictions (max error {error:g}); "
            "no detection selection or expert verification was rerun"
        )
    return DetectionResults(output, config, summary, snapshots, source, replay_seconds)
