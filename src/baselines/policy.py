from __future__ import annotations

import numpy as np
from scipy import sparse

from config.methods import ACTIVE_LABEL_CORRECTION

from ..utils.neighbors import faiss_neighbors, pairwise_distance_scale
from ..utils.policy import (
    MethodPrediction,
    QueryResult,
    entropy,
    percentile_mislabel_score,
    query_candidates,
    top_scores,
)
from .scores import SCORE_CALIBRATION_METHODS, base_mislabel_score
from .verification import (
    active_label_cleaning_score,
    build_label_propagation_graph,
    calibrate_score,
    cleannet_acquisition,
    cleannet_score,
    dalc_quantities,
    estimate_noise_transition,
    graph_label_propagation,
    misdetect_acquisition,
    misdetect_score,
    noise_adjusted_posterior,
    observed_label_risk,
    robust_alc_acquisition,
    stratified_score_selection,
)


FROZEN_METHOD_BASE = {
    "robust_alc_frozen": "robust_alc",
    "dalc_frozen": "dalc",
}
FROZEN_METHODS = tuple(FROZEN_METHOD_BASE)

BASELINE_METHODS = (
    *SCORE_CALIBRATION_METHODS,
    "robust_alc",
    "robust_alc_frozen",
    "dalc",
    "dalc_frozen",
    "active_label_cleaning",
    "active_label_correction",
    "graph_label_propagation",
    "cleannet",
    "misdetect_b",
)

STATUS_ONLY_METHODS = {
    *SCORE_CALIBRATION_METHODS,
    "graph_label_propagation",
    "cleannet",
    "misdetect_b",
}
CORRECTION_METHODS = {
    "robust_alc",
    "robust_alc_frozen",
    "dalc",
    "dalc_frozen",
    "active_label_cleaning",
    "active_label_correction",
}
WARM_START_METHODS = {
    "robust_alc",
    "dalc",
    "active_label_cleaning",
    "active_label_correction",
}
FIXED_MODEL_METHODS = {*STATUS_ONLY_METHODS, *FROZEN_METHODS}
NOISE_MODEL_METHODS = {"robust_alc", "dalc"}


def _robust_query(
    artifacts,
    queried: np.ndarray,
    count: int,
    seed: int,
    candidate_mask: np.ndarray | None,
):
    entropy_values = entropy(artifacts.probability)
    suspiciousness = 1.0 - artifacts.contribution
    neighbor_count = min(ACTIVE_LABEL_CORRECTION["neighbors"], len(entropy_values) - 1)
    neighbors, squared_distance = faiss_neighbors(
        artifacts.embedding,
        neighbor_count,
        normalize=False,
    )
    sigma = pairwise_distance_scale(
        artifacts.embedding,
        seed=seed,
        pair_count=ACTIVE_LABEL_CORRECTION["distance_pairs"],
    )
    row = np.repeat(np.arange(len(entropy_values)), neighbors.shape[1])
    column = neighbors.reshape(-1)
    weights = np.exp(-squared_distance.reshape(-1) / sigma)
    directed = sparse.csr_matrix(
        (weights, (row, column)), shape=(len(entropy_values),) * 2
    )
    adjacency = directed.maximum(directed.T)
    transition = sparse.diags(
        1.0 / np.maximum(np.asarray(adjacency.sum(axis=1)).reshape(-1), 1.0e-12)
    ) @ adjacency
    active = entropy_values.copy()
    active[queried] = 0.0
    score = np.full(len(entropy_values), -np.inf, dtype=np.float64)
    selected = []
    available = np.zeros(len(queried), dtype=bool)
    available[query_candidates(queried, candidate_mask)] = True
    if count > int(available.sum()):
        raise ValueError("query count exceeds the number of eligible samples")
    rng = np.random.default_rng(seed)
    for step in range(count):
        available_ids = np.flatnonzero(available)
        candidate_count = min(count, len(available_ids))
        candidates = top_scores(
            suspiciousness,
            available_ids,
            candidate_count,
            seed + step,
        )
        maximum = active[candidates].max()
        tied = candidates[active[candidates] == maximum]
        choice = int(rng.choice(tied))
        selected.append(choice)
        score[choice] = active[choice]
        available[choice] = False
        active[choice] = 0.0
        delta = ACTIVE_LABEL_CORRECTION["diffusion_step_size"]
        for _ in range(ACTIVE_LABEL_CORRECTION["diffusion_steps"]):
            active = (1.0 - delta) * active + delta * (transition @ active)
        active = np.maximum(active, 0.0)
    return QueryResult(np.asarray(selected, dtype=np.int64), score)


def should_train_acquisition_model(method: str, checkpoint_index: int) -> bool:
    """Return whether this checkpoint trains a fresh acquisition model."""
    if method in FIXED_MODEL_METHODS:
        return checkpoint_index == 0
    return True


def method_feedback_mode(method: str) -> str:
    if method in CORRECTION_METHODS:
        return "verified_true_label"
    if method in STATUS_ONLY_METHODS:
        return "mislabel_status"
    raise ValueError(f"unknown method: {method}")


def method_update_schedule(method: str, *, warm_start: bool = False) -> str:
    if method in SCORE_CALIBRATION_METHODS:
        return "score_model_once_stratified_status_calibration_each_checkpoint"
    if method in {"graph_label_propagation", "cleannet", "misdetect_b"}:
        return "task_model_once_status_inference_each_budget_checkpoint"
    if method == "robust_alc_frozen":
        return "task_model_once_noise_transition_and_posterior_each_checkpoint"
    if method == "dalc_frozen":
        return "task_model_once_dual_query_and_noise_posterior_each_checkpoint"
    if method == "active_label_cleaning":
        schedule = "adapted_one_update_per_requested_budget_checkpoint"
        return f"{schedule}_warm_start" if warm_start else schedule
    if method == "active_label_correction":
        schedule = "native_update_and_entropy_propagation_per_budget_checkpoint"
        return f"{schedule}_warm_start" if warm_start else schedule
    if method == "robust_alc":
        schedule = "adapted_noise_model_and_classifier_update_per_budget_checkpoint"
        return f"{schedule}_warm_start" if warm_start else schedule
    if method == "dalc":
        schedule = "adapted_dual_query_and_noise_model_update_per_budget_checkpoint"
        return f"{schedule}_warm_start" if warm_start else schedule
    raise ValueError(f"unknown method: {method}")


class BaselinePolicy:
    """Stateful adapter for all baseline prediction and query policies."""

    def __init__(
        self,
        method: str,
        noisy_labels: np.ndarray,
        *,
        modality: str,
        seed: int,
        native_features: np.ndarray | None = None,
    ):
        if method not in BASELINE_METHODS:
            raise ValueError(
                f"unknown baseline {method!r}; choose from {BASELINE_METHODS}"
            )
        self.registered_method = method
        self.method = FROZEN_METHOD_BASE.get(method, method)
        self.noisy_labels = np.asarray(noisy_labels, dtype=np.int64).copy()
        self.modality = modality
        self.seed = int(seed)
        self.native_features = native_features
        self._lp_transition = None
        self.pseudo_mask = np.zeros(len(self.noisy_labels), dtype=bool)
        self.last_noise_transition = None

    def trusted_mask(self, queried: np.ndarray) -> np.ndarray:
        trusted = np.asarray(queried, dtype=bool).copy()
        if self.method == "dalc":
            trusted |= self.pseudo_mask
        return trusted

    def training_transition(
        self,
        previous_artifacts,
        current_labels: np.ndarray,
        queried: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.method not in NOISE_MODEL_METHODS or previous_artifacts is None:
            return None, None
        trusted = self.trusted_mask(queried)
        transition = estimate_noise_transition(
            previous_artifacts.probability,
            self.noisy_labels,
            current_labels,
            trusted,
        )
        self.last_noise_transition = transition
        return transition, trusted

    def register_pseudo_labels(
        self,
        ids: np.ndarray | None,
        predicted_labels: np.ndarray,
        current_labels: np.ndarray,
    ) -> None:
        if ids is None or len(ids) == 0:
            return
        ids = np.asarray(ids, dtype=np.int64)
        self.pseudo_mask[ids] = True
        current_labels[ids] = np.asarray(predicted_labels, dtype=np.int64)[ids]

    def predict(
        self,
        artifacts,
        current_labels: np.ndarray,
        queried: np.ndarray,
        queried_status: np.ndarray,
    ) -> MethodPrediction:
        labels = self.noisy_labels
        base_issue = np.asarray(artifacts.probability).argmax(axis=1) != labels

        if self.method in SCORE_CALIBRATION_METHODS:
            native = base_mislabel_score(
                self.method,
                artifacts,
                labels,
                feature_override=self.native_features,
            )
            score = calibrate_score(native, queried, queried_status)
            has_calibration = (
                int(np.sum(queried)) >= 2
                and len(np.unique(queried_status[queried])) == 2
            )
            issue = score >= 0.5 if has_calibration else base_issue
            rule = (
                "verified_isotonic_score_probability_at_least_0.5"
                if has_calibration
                else "task_class_fallback_until_calibration_has_both_statuses"
            )
            return MethodPrediction(
                score,
                issue,
                rule,
            )

        if self.method == "robust_alc":
            score = observed_label_risk(artifacts, labels)
            trusted = self.trusted_mask(queried)
            transition = estimate_noise_transition(
                artifacts.probability,
                labels,
                current_labels,
                trusted,
            )
            posterior = noise_adjusted_posterior(
                artifacts.probability,
                labels,
                transition,
                current_labels,
                trusted,
            )
            untrusted = ~trusted
            score[untrusted] = 1.0 - posterior[
                np.flatnonzero(untrusted), labels[untrusted]
            ]
            self.last_noise_transition = transition
            return MethodPrediction(
                score,
                base_issue,
                "final_classifier_class_differs_from_noisy_label",
            )

        if self.method == "active_label_correction":
            score = observed_label_risk(artifacts, labels)
            return MethodPrediction(
                score,
                base_issue,
                "final_classifier_class_differs_from_noisy_label",
            )

        if self.method == "dalc":
            trusted = self.trusted_mask(queried)
            posterior, _, transition = dalc_quantities(
                artifacts,
                labels,
                current_labels,
                trusted,
            )
            self.last_noise_transition = transition
            score = 1.0 - posterior[np.arange(len(labels)), labels]
            return MethodPrediction(
                score,
                posterior.argmax(axis=1) != labels,
                "noise_adjusted_class_differs_from_noisy_label",
            )

        if self.method == "active_label_cleaning":
            observed = np.clip(
                np.asarray(artifacts.probability)[np.arange(len(labels)), labels],
                1.0e-12,
                1.0,
            )
            correctness_component = -np.log(observed) / np.log(
                artifacts.probability.shape[1]
            )
            return MethodPrediction(
                percentile_mislabel_score(correctness_component),
                base_issue,
                "final_selector_class_differs_from_noisy_label",
            )

        if self.method == "graph_label_propagation":
            if self._lp_transition is None:
                self._lp_transition = build_label_propagation_graph(
                    artifacts.embedding
                )
            prior = observed_label_risk(artifacts, labels) + percentile_mislabel_score(
                artifacts.early_loss
            )
            score = graph_label_propagation(
                self._lp_transition, prior, queried, queried_status
            )
            return MethodPrediction(
                score, score >= 0.5, "propagated_status_probability_at_least_0.5"
            )

        if self.method == "cleannet":
            score = cleannet_score(
                artifacts, labels, queried, queried_status, self.seed
            )
            return MethodPrediction(
                score, score >= 0.5, "cleannet_mislabel_probability_at_least_0.5"
            )

        if self.method == "misdetect_b":
            score = misdetect_score(
                artifacts, labels, queried, queried_status, self.seed
            )
            return MethodPrediction(
                score, score >= 0.5, "misdetect_binary_probability_at_least_0.5"
            )

        raise ValueError(f"unknown method: {self.method}")

    def select(
        self,
        artifacts,
        current_labels: np.ndarray,
        queried: np.ndarray,
        queried_status: np.ndarray,
        *,
        count: int,
        seed: int,
        candidate_mask: np.ndarray | None = None,
    ) -> QueryResult:
        candidates = query_candidates(queried, candidate_mask)
        if count > len(candidates):
            raise ValueError("query count exceeds the number of eligible samples")

        prediction = self.predict(artifacts, current_labels, queried, queried_status)
        if self.method in SCORE_CALIBRATION_METHODS:
            native = base_mislabel_score(
                self.method,
                artifacts,
                self.noisy_labels,
                feature_override=self.native_features,
            )
            selected = stratified_score_selection(native, candidates, count, seed)
            return QueryResult(selected, native)

        if self.method == "robust_alc":
            acquisition, transition = robust_alc_acquisition(
                artifacts,
                self.noisy_labels,
                current_labels,
                self.trusted_mask(queried),
            )
            self.last_noise_transition = transition
            return QueryResult(
                top_scores(acquisition, candidates, count, seed), acquisition
            )

        if self.method == "dalc":
            # Official DALC-E uses high-entropy oracle instances together with
            # an equally sized low-entropy pseudo-labelled source.
            entropy_values = entropy(np.asarray(artifacts.probability))
            selected = top_scores(entropy_values, candidates, count, seed)
            pseudo_candidates = np.setdiff1d(
                candidates[~self.pseudo_mask[candidates]], selected, assume_unique=False
            )
            pseudo_count = min(count, len(pseudo_candidates))
            tie = np.random.default_rng(seed + 1).random(len(pseudo_candidates))
            order = np.lexsort((tie, entropy_values[pseudo_candidates]))
            pseudo = pseudo_candidates[order[:pseudo_count]]
            return QueryResult(selected, entropy_values, pseudo_selected=pseudo)

        if self.method == "active_label_cleaning":
            acquisition = active_label_cleaning_score(
                artifacts.probability, current_labels
            )
            return QueryResult(
                top_scores(acquisition, candidates, count, seed), acquisition
            )

        if self.method == "active_label_correction":
            return _robust_query(
                artifacts, queried, count, seed, candidate_mask
            )

        if self.method == "graph_label_propagation":
            acquisition = 1.0 - np.abs(2.0 * prediction.score - 1.0)
            return QueryResult(
                top_scores(acquisition, candidates, count, seed), acquisition
            )

        if self.method == "cleannet":
            clean_references = queried & (queried_status == 0)
            acquisition = cleannet_acquisition(
                prediction.score, self.noisy_labels, clean_references
            )
            return QueryResult(
                top_scores(acquisition, candidates, count, seed), acquisition
            )

        if self.method == "misdetect_b":
            acquisition = misdetect_acquisition(
                artifacts, self.noisy_labels, prediction.score
            )
            return QueryResult(
                top_scores(acquisition, candidates, count, seed), acquisition
            )

        raise ValueError(f"unknown method: {self.method}")
