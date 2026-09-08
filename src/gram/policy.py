"""GRAM graph-prior construction, posterior inference, and acquisition policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from scipy.special import expit, ndtr
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

from config.methods import DEFAULT_GRAM_VARIANT, GRAM_VARIANTS

from ..utils.neighbors import faiss_neighbors
from ..utils.policy import MethodPrediction, QueryResult, query_candidates, top_scores


GRAM_METHOD = "ours"


@dataclass
class PriorGraph:
    neighbors: np.ndarray
    features: np.ndarray


@dataclass
class TrajectoryKernels:
    margin_features: np.ndarray
    gradient_features: np.ndarray


@dataclass
class KernelWeightFit:
    weights: np.ndarray
    objective: float | None
    iterations: int
    converged: bool
    learned: bool
    reason: str


def diagnostic_priors(artifacts, labels: np.ndarray, k: int = 50):
    labels = np.asarray(labels, dtype=np.int64)
    confidence = 1.0 - artifacts.probability[np.arange(len(labels)), labels]
    neighbors, _ = faiss_neighbors(
        artifacts.embedding, min(k, len(labels) - 1), normalize=True
    )
    disagreement = 1.0 - (labels[neighbors] == labels[:, None]).mean(axis=1)
    diagnostics = np.column_stack((confidence, artifacts.early_loss, disagreement))
    return diagnostics, neighbors


def _knn_graph(values: np.ndarray, k: int = 50):
    values = np.asarray(values, dtype=np.float64)
    scale = values.std(axis=0)
    standardized = (values - values.mean(axis=0)) / np.where(
        scale > 1.0e-12, scale, 1.0
    )
    k = min(k, len(values) - 1)
    search = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=-1)
    distance, neighbors = search.fit(standardized).kneighbors(standardized)
    distance, neighbors = distance[:, 1:], neighbors[:, 1:]
    bandwidth = np.maximum(distance[:, -1], np.finfo(np.float64).eps)
    row = np.repeat(np.arange(len(values)), k)
    column = neighbors.reshape(-1)
    denominator = np.sqrt(bandwidth[row] * bandwidth[column])
    weight = np.exp(-0.5 * np.square(distance.reshape(-1) / denominator))
    adjacency = sparse.csr_matrix((weight, (row, column)), shape=(len(values),) * 2)
    return adjacency.maximum(adjacency.T).tocsr(), neighbors


def _trajectory_knn_graph(values: np.ndarray, k: int = 50):
    """Build a scalable self-tuning graph from one trajectory view."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("trajectory values must be a finite N x T matrix")
    scale = values.std(axis=0)
    standardized = (values - values.mean(axis=0)) / np.where(
        scale > 1.0e-12, scale, 1.0
    )
    k = min(k, len(values) - 1)
    neighbors, squared_distance = faiss_neighbors(
        standardized, k, normalize=False
    )
    bandwidth_squared = np.maximum(
        squared_distance[:, -1], np.finfo(np.float64).eps
    )
    row = np.repeat(np.arange(len(values)), k)
    column = neighbors.reshape(-1)
    denominator = np.sqrt(
        bandwidth_squared[row] * bandwidth_squared[column]
    )
    weight = np.exp(
        -0.5 * squared_distance.reshape(-1) / np.maximum(
            denominator, np.finfo(np.float64).eps
        )
    )
    adjacency = sparse.csr_matrix(
        (weight, (row, column)), shape=(len(values),) * 2
    )
    return adjacency.maximum(adjacency.T).tocsr(), neighbors


def _heat_kernel_features(
    adjacency: sparse.csr_matrix,
    rank: int,
    *,
    trace_normalize: bool,
) -> np.ndarray:
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = sparse.diags(1.0 / np.sqrt(np.maximum(degree, 1.0e-12)))
    normalized_adjacency = inverse @ adjacency @ inverse
    sample_count = adjacency.shape[0]
    active_rank = min(rank, sample_count - 1)
    if active_rank <= 0:
        raise ValueError("a graph kernel requires at least two samples")
    if sample_count <= rank + 1:
        values, vectors = np.linalg.eigh(normalized_adjacency.toarray())
        values, vectors = values[-active_rank:], vectors[:, -active_rank:]
    else:
        values, vectors = eigsh(
            normalized_adjacency, k=active_rank, which="LA"
        )
    order = np.argsort(values)[::-1]
    laplacian_values = np.maximum(1.0 - values[order], 0.0)
    features = vectors[:, order] * np.exp(-0.5 * laplacian_values)[None, :]
    if trace_normalize:
        trace = float(np.einsum("ij,ij->", features, features))
        features *= math.sqrt(sample_count / max(trace, 1.0e-12))
    else:
        features *= math.sqrt(sample_count / active_rank)
    return features


def build_prior_graph(
    diagnostics: np.ndarray, rank: int = 64, k: int = 50
) -> PriorGraph:
    adjacency, neighbors = _knn_graph(diagnostics, k)
    features = _heat_kernel_features(
        adjacency, rank, trace_normalize=False
    )
    return PriorGraph(neighbors=neighbors, features=features)


def build_trajectory_kernel(
    trajectory: np.ndarray, rank: int = 64, k: int = 50
) -> PriorGraph:
    adjacency, neighbors = _trajectory_knn_graph(trajectory, k)
    features = _heat_kernel_features(adjacency, rank, trace_normalize=True)
    return PriorGraph(neighbors=neighbors, features=features)


class GraphGPClassifier:
    """Low-rank graph GP with an optional independent per-sample component.

    The latent kernel is ``features @ features.T + identity_weight * I``.
    Independent latent variables are instantiated only for observed samples;
    this retains the exact diagonal component without materializing an n by n
    identity feature matrix.
    """

    def __init__(
        self,
        features: np.ndarray,
        prior_mean: np.ndarray,
        identity_weight: float = 1.0,
    ):
        self.features = np.asarray(features, dtype=np.float64)
        self.prior_mean = np.asarray(prior_mean, dtype=np.float64)
        self.identity_weight = float(identity_weight)
        if self.identity_weight < 0.0:
            raise ValueError("identity kernel weight must be non-negative")

    def posterior(self, observed_ids: np.ndarray, values: np.ndarray):
        rank = self.features.shape[1]
        weight = np.zeros(rank, dtype=np.float64)
        independent = np.zeros(len(observed_ids), dtype=np.float64)
        if len(observed_ids):
            observed = self.features[observed_ids]
            offset = self.prior_mean[observed_ids]
            for _ in range(50):
                probability = expit(offset + observed @ weight + independent)
                curvature = np.maximum(probability * (1.0 - probability), 1.0e-8)
                residual = probability - values
                if self.identity_weight:
                    diagonal = 1.0 / self.identity_weight + curvature
                    effective_curvature = curvature / (
                        1.0 + self.identity_weight * curvature
                    )
                    precision = np.eye(rank) + observed.T @ (
                        effective_curvature[:, None] * observed
                    )
                    gradient_independent = (
                        independent / self.identity_weight + residual
                    )
                    gradient = weight + observed.T @ residual
                    reduced_gradient = gradient - observed.T @ (
                        (curvature / diagonal) * gradient_independent
                    )
                else:
                    diagonal = None
                    precision = np.eye(rank) + observed.T @ (
                        curvature[:, None] * observed
                    )
                    reduced_gradient = weight + observed.T @ residual
                factor = linalg.cholesky(precision, lower=True, check_finite=False)
                step = linalg.cho_solve(
                    (factor, True), reduced_gradient, check_finite=False
                )
                weight -= step
                max_step = float(np.max(np.abs(step)))
                if self.identity_weight:
                    independent_step = (
                        gradient_independent - curvature * (observed @ step)
                    ) / diagonal
                    independent -= independent_step
                    max_step = max(
                        max_step, float(np.max(np.abs(independent_step)))
                    )
                if max_step < 1.0e-8:
                    break
            probability = expit(offset + observed @ weight + independent)
            curvature = np.maximum(probability * (1.0 - probability), 1.0e-8)
            if self.identity_weight:
                effective_curvature = curvature / (
                    1.0 + self.identity_weight * curvature
                )
            else:
                effective_curvature = curvature
            precision = np.eye(rank) + observed.T @ (
                effective_curvature[:, None] * observed
            )
            factor = linalg.cholesky(precision, lower=True, check_finite=False)
        else:
            factor = np.eye(rank)
        latent_mean = self.prior_mean + self.features @ weight
        if len(observed_ids) and self.identity_weight:
            latent_mean[observed_ids] += independent
        latent = linalg.solve_triangular(
            factor, self.features.T, lower=True, check_finite=False
        ).T
        graph_variance = np.einsum("ij,ij->i", latent, latent)
        variance = graph_variance + self.identity_weight
        if len(observed_ids) and self.identity_weight:
            observed_curvature = curvature
            shrinkage = 1.0 / (
                1.0 + self.identity_weight * observed_curvature
            )
            independent_variance = self.identity_weight * shrinkage
            variance[observed_ids] = (
                np.square(shrinkage) * graph_variance[observed_ids]
                + independent_variance
            )
        probability = expit(
            latent_mean / np.sqrt(1.0 + math.pi * variance / 8.0)
        )
        return probability, variance


def _weighted_features(
    kernels: TrajectoryKernels, weights: np.ndarray
) -> np.ndarray:
    blocks = []
    for weight, features in zip(
        weights[:2],
        (kernels.margin_features, kernels.gradient_features),
    ):
        if weight > 0.0 and features.shape[1]:
            blocks.append(math.sqrt(float(weight)) * features)
    if not blocks:
        return np.empty((len(kernels.margin_features), 0), dtype=np.float64)
    return np.column_stack(blocks)


class GaussianGraphGP:
    """Gaussian surrogate GP for a low-rank mixture plus diagonal kernel."""

    def __init__(
        self,
        kernels: TrajectoryKernels,
        prior_mean: np.ndarray,
        weights: np.ndarray,
        observation_noise_variance: float,
    ):
        self.kernels = kernels
        self.prior_mean = np.asarray(prior_mean, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.observation_noise_variance = float(observation_noise_variance)
        if self.weights.shape != (3,) or np.any(self.weights < 0.0):
            raise ValueError("kernel weights must be three non-negative values")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("kernel weights must sum to one")
        if self.observation_noise_variance <= 0.0:
            raise ValueError("observation noise variance must be positive")

    def posterior(self, observed_ids: np.ndarray, values: np.ndarray):
        """Return mislabel probabilities, latent variances, and latent means."""
        observed_ids = np.asarray(observed_ids, dtype=np.int64)
        targets = 2.0 * np.asarray(values, dtype=np.float64) - 1.0
        features = _weighted_features(self.kernels, self.weights)
        identity_weight = float(self.weights[2])
        total_noise = identity_weight + self.observation_noise_variance
        rank = features.shape[1]
        if len(observed_ids) and rank:
            observed = features[observed_ids]
            precision = np.eye(rank) + observed.T @ observed / total_noise
            factor = linalg.cholesky(precision, lower=True, check_finite=False)
            residual = targets - self.prior_mean[observed_ids]
            projected = observed.T @ residual
            coefficient = linalg.cho_solve(
                (factor, True), projected / total_noise, check_finite=False
            )
        else:
            observed = np.empty((len(observed_ids), rank), dtype=np.float64)
            factor = np.eye(rank)
            residual = targets - self.prior_mean[observed_ids]
            coefficient = np.zeros(rank, dtype=np.float64)

        latent_mean = self.prior_mean + features @ coefficient
        if len(observed_ids) and identity_weight:
            independent_mean = identity_weight * (
                residual - observed @ coefficient
            ) / total_noise
            latent_mean[observed_ids] += independent_mean

        if rank:
            latent = linalg.solve_triangular(
                factor, features.T, lower=True, check_finite=False
            ).T
            graph_variance = np.einsum("ij,ij->i", latent, latent)
        else:
            graph_variance = np.zeros(len(self.prior_mean), dtype=np.float64)
        variance = graph_variance + identity_weight
        if len(observed_ids) and identity_weight:
            shrinkage = self.observation_noise_variance / total_noise
            variance[observed_ids] = (
                np.square(shrinkage) * graph_variance[observed_ids]
                + identity_weight * shrinkage
            )
        probability = ndtr(
            latent_mean
            / np.sqrt(
                np.maximum(
                    variance + self.observation_noise_variance, 1.0e-12
                )
            )
        )
        return probability, variance, latent_mean


def _simplex_weights(theta: np.ndarray, active: tuple[int, ...]) -> np.ndarray:
    logits = np.r_[np.asarray(theta, dtype=np.float64), 0.0]
    logits -= logits.max()
    active_weights = np.exp(logits)
    active_weights /= active_weights.sum()
    weights = np.zeros(3, dtype=np.float64)
    weights[np.asarray(active, dtype=np.int64)] = active_weights
    return weights


def _theta_from_weights(weights: np.ndarray, active: tuple[int, ...]) -> np.ndarray:
    active_weights = np.maximum(
        np.asarray(weights, dtype=np.float64)[np.asarray(active)], 1.0e-12
    )
    active_weights /= active_weights.sum()
    return np.log(active_weights[:-1]) - math.log(active_weights[-1])


def _gaussian_marginal_objective(
    residual: np.ndarray,
    margin_features: np.ndarray,
    gradient_features: np.ndarray,
    weights: np.ndarray,
    observation_noise_variance: float,
    reference_weights: np.ndarray,
    regularization: float,
) -> float:
    kernels = TrajectoryKernels(margin_features, gradient_features)
    features = _weighted_features(kernels, weights)
    diagonal = float(weights[2]) + observation_noise_variance
    if features.shape[1]:
        precision = np.eye(features.shape[1]) + features.T @ features / diagonal
        factor = linalg.cholesky(precision, lower=True, check_finite=False)
        projected = features.T @ residual
        correction = float(
            projected
            @ linalg.cho_solve(
                (factor, True), projected, check_finite=False
            )
        )
        quadratic = float(residual @ residual) / diagonal - correction / (
            diagonal * diagonal
        )
        log_determinant = (
            len(residual) * math.log(diagonal)
            + 2.0 * float(np.log(np.diag(factor)).sum())
        )
    else:
        quadratic = float(residual @ residual) / diagonal
        log_determinant = len(residual) * math.log(diagonal)
    negative_log_likelihood = 0.5 * (quadratic + log_determinant) / len(residual)
    penalty = regularization * float(
        np.square(weights - reference_weights).sum()
    )
    return negative_log_likelihood + penalty


def learn_kernel_weights(
    kernels: TrajectoryKernels,
    prior_mean: np.ndarray,
    observed_ids: np.ndarray,
    values: np.ndarray,
    previous_weights: np.ndarray,
    config: dict,
) -> KernelWeightFit:
    active_names = tuple(config["active_kernel_components"])
    component_index = {"margin": 0, "gradient": 1, "identity": 2}
    active = tuple(component_index[name] for name in active_names)
    initial = np.asarray(config["initial_kernel_weights"], dtype=np.float64)
    adaptive = config["kernel_weights"] == "adaptive"
    observed_ids = np.asarray(observed_ids, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    if not adaptive:
        return KernelWeightFit(initial, None, 0, True, False, "fixed")
    if len(observed_ids) < config["minimum_queried_for_weight_learning"]:
        return KernelWeightFit(
            previous_weights, None, 0, True, False, "insufficient_queries"
        )
    if (
        config["require_both_status_classes_for_weight_learning"]
        and len(np.unique(values)) < 2
    ):
        return KernelWeightFit(
            previous_weights, None, 0, True, False, "single_status_class"
        )

    targets = 2.0 * values - 1.0
    residual = targets - np.asarray(prior_mean, dtype=np.float64)[observed_ids]
    margin = kernels.margin_features[observed_ids]
    gradient = kernels.gradient_features[observed_ids]
    margin_gram = margin.T @ margin
    gradient_gram = gradient.T @ gradient
    cross_gram = margin.T @ gradient
    margin_projection = margin.T @ residual
    gradient_projection = gradient.T @ residual
    residual_square = float(residual @ residual)
    reference = initial
    theta = _theta_from_weights(previous_weights, active)

    def objective(candidate: np.ndarray) -> float:
        weights = _simplex_weights(candidate, active)
        margin_weight, gradient_weight, identity_weight = weights
        diagonal = identity_weight + config["observation_noise_variance"]
        blocks = []
        projections = []
        if margin_weight > 0.0 and margin.shape[1]:
            blocks.append("margin")
            projections.append(math.sqrt(margin_weight) * margin_projection)
        if gradient_weight > 0.0 and gradient.shape[1]:
            blocks.append("gradient")
            projections.append(math.sqrt(gradient_weight) * gradient_projection)
        if blocks == ["margin", "gradient"]:
            cross = math.sqrt(margin_weight * gradient_weight) * cross_gram
            gram = np.block(
                [
                    [margin_weight * margin_gram, cross],
                    [cross.T, gradient_weight * gradient_gram],
                ]
            )
        elif blocks == ["margin"]:
            gram = margin_weight * margin_gram
        elif blocks == ["gradient"]:
            gram = gradient_weight * gradient_gram
        else:
            gram = np.empty((0, 0), dtype=np.float64)
        if len(gram):
            precision = np.eye(len(gram)) + gram / diagonal
            factor = linalg.cholesky(precision, lower=True, check_finite=False)
            projected = np.concatenate(projections)
            correction = float(
                projected
                @ linalg.cho_solve(
                    (factor, True), projected, check_finite=False
                )
            )
            quadratic = residual_square / diagonal - correction / (
                diagonal * diagonal
            )
            log_determinant = (
                len(residual) * math.log(diagonal)
                + 2.0 * float(np.log(np.diag(factor)).sum())
            )
        else:
            quadratic = residual_square / diagonal
            log_determinant = len(residual) * math.log(diagonal)
        negative_log_likelihood = 0.5 * (
            quadratic + log_determinant
        ) / len(residual)
        penalty = config["weight_regularization"] * float(
            np.square(weights - reference).sum()
        )
        return negative_log_likelihood + penalty

    initial_objective = objective(theta)
    result = minimize(
        objective,
        theta,
        method="L-BFGS-B",
        options={"maxiter": 100, "ftol": 1.0e-10},
    )
    candidate_weights = _simplex_weights(result.x, active)
    candidate_objective = float(result.fun)
    if not np.isfinite(candidate_objective) or candidate_objective > initial_objective:
        candidate_weights = np.asarray(previous_weights, dtype=np.float64)
        candidate_objective = initial_objective
    return KernelWeightFit(
        candidate_weights,
        candidate_objective,
        int(result.nit),
        bool(result.success),
        True,
        str(result.message),
    )


def prior_probability(diagnostics: np.ndarray) -> np.ndarray:
    combined = diagnostics.mean(axis=1)
    probability = rankdata(combined, method="average") / (len(combined) + 1.0)
    return np.clip(probability, 1.0e-4, 1.0 - 1.0e-4)


def prior_mean(diagnostics: np.ndarray) -> np.ndarray:
    probability = prior_probability(diagnostics)
    return np.log(probability / (1.0 - probability))


def gaussian_prior_mean(diagnostics: np.ndarray) -> np.ndarray:
    return 2.0 * prior_probability(diagnostics) - 1.0


def predict_gram(
    diagnostics: np.ndarray,
    graph: PriorGraph,
    queried: np.ndarray,
    queried_status: np.ndarray,
    identity_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    observed_ids = np.flatnonzero(queried)
    classifier = GraphGPClassifier(
        graph.features, prior_mean(diagnostics), identity_weight
    )
    return classifier.posterior(
        observed_ids, queried_status[observed_ids].astype(np.float64)
    )


def select_gram(
    diagnostics: np.ndarray,
    graph: PriorGraph,
    queried: np.ndarray,
    queried_status: np.ndarray,
    *,
    count: int,
    seed: int,
    identity_weight: float = 1.0,
    acquisition: str = "posterior_variance",
    candidate_mask: np.ndarray | None = None,
) -> QueryResult:
    probability, variance = predict_gram(
        diagnostics, graph, queried, queried_status, identity_weight
    )
    if acquisition == "posterior_variance":
        acquisition_score = variance
    elif acquisition == "ucb":
        acquisition_score = probability + np.sqrt(np.maximum(variance, 0.0))
    else:
        raise ValueError(f"unknown GRAM acquisition function {acquisition!r}")
    candidates = query_candidates(queried, candidate_mask)
    if count > len(candidates):
        raise ValueError("query count exceeds the number of eligible samples")
    selected = top_scores(acquisition_score, candidates, count, seed)
    return QueryResult(selected, acquisition_score, variance)


class GramPolicy:
    """Stateful policy adapter for the proposed GRAM method."""

    def __init__(
        self,
        noisy_labels: np.ndarray,
        *,
        modality: str,
        seed: int,
        native_features: np.ndarray | None = None,
        variant: str = DEFAULT_GRAM_VARIANT,
    ):
        del modality, native_features
        self.noisy_labels = np.asarray(noisy_labels, dtype=np.int64).copy()
        self.seed = int(seed)
        if variant not in GRAM_VARIANTS:
            raise ValueError(
                f"unknown GRAM variant {variant!r}; choose from {tuple(GRAM_VARIANTS)}"
            )
        self.variant = variant
        self.config = GRAM_VARIANTS[variant]
        self.pseudo_mask = np.zeros(len(self.noisy_labels), dtype=bool)
        self.last_noise_transition = None
        self._diagnostics = None
        self._legacy_graph = None
        self._trajectory_kernels = None
        initial_weights = self.config.get("initial_kernel_weights")
        self._kernel_weights = (
            None
            if initial_weights is None
            else np.asarray(initial_weights, dtype=np.float64)
        )
        self._weight_fit = None
        self._posterior_key = None
        self._posterior_value = None
        self._posterior_mean = None

    def _ensure_prior(self, artifacts) -> None:
        if self._diagnostics is None:
            legacy = self.config["inference"] == "bernoulli_logistic_laplace"
            diagnostic_neighbors = self.config.get(
                "diagnostic_neighbors", self.config["graph_neighbors"]
            )
            self._diagnostics, _ = diagnostic_priors(
                artifacts,
                self.noisy_labels,
                k=diagnostic_neighbors,
            )
            if legacy:
                self._legacy_graph = build_prior_graph(
                    self._diagnostics,
                    rank=self.config["graph_rank"],
                    k=self.config["graph_neighbors"],
                )
                return

            active = set(self.config["active_kernel_components"])
            sample_count = len(self.noisy_labels)
            empty = np.empty((sample_count, 0), dtype=np.float64)
            if "margin" in active:
                margin = build_trajectory_kernel(
                    artifacts.margin_trajectory,
                    rank=self.config["margin_graph_rank"],
                    k=self.config["graph_neighbors"],
                ).features
            else:
                margin = empty
            if "gradient" in active:
                gradient = build_trajectory_kernel(
                    artifacts.gradient_trajectory,
                    rank=self.config["gradient_graph_rank"],
                    k=self.config["graph_neighbors"],
                ).features
            else:
                gradient = empty
            self._trajectory_kernels = TrajectoryKernels(margin, gradient)

    def _state_key(
        self, queried: np.ndarray, queried_status: np.ndarray
    ) -> tuple[bytes, bytes]:
        observed_ids = np.flatnonzero(queried).astype(np.int64, copy=False)
        observed_status = np.asarray(queried_status, dtype=np.int64)[observed_ids]
        return observed_ids.tobytes(), observed_status.tobytes()

    def _posterior(
        self,
        artifacts,
        queried: np.ndarray,
        queried_status: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_prior(artifacts)
        key = self._state_key(queried, queried_status)
        if key == self._posterior_key:
            return self._posterior_value

        observed_ids = np.flatnonzero(queried)
        observed_values = np.asarray(queried_status, dtype=np.float64)[observed_ids]
        if self.config["inference"] == "bernoulli_logistic_laplace":
            probability, variance = predict_gram(
                self._diagnostics,
                self._legacy_graph,
                queried,
                queried_status,
                self.config["identity_weight"],
            )
            self._weight_fit = None
            self._posterior_mean = None
        else:
            mean = gaussian_prior_mean(self._diagnostics)
            if len(observed_ids) == 0:
                self._weight_fit = KernelWeightFit(
                    self._kernel_weights.copy(),
                    None,
                    0,
                    True,
                    False,
                    "cold_start",
                )
            else:
                self._weight_fit = learn_kernel_weights(
                    self._trajectory_kernels,
                    mean,
                    observed_ids,
                    observed_values,
                    self._kernel_weights,
                    self.config,
                )
                self._kernel_weights = self._weight_fit.weights.copy()
            classifier = GaussianGraphGP(
                self._trajectory_kernels,
                mean,
                self._kernel_weights,
                self.config["observation_noise_variance"],
            )
            probability, variance, self._posterior_mean = classifier.posterior(
                observed_ids, observed_values
            )

        self._posterior_key = key
        self._posterior_value = probability, variance
        return self._posterior_value

    def kernel_weight_state(self) -> dict:
        if self._weight_fit is None:
            return {
                "omega_margin": None,
                "omega_gradient": None,
                "omega_identity": None,
                "weight_objective": None,
                "weight_iterations": 0,
                "weight_converged": None,
                "weight_learned": False,
                "weight_update_reason": "legacy_kernel",
            }
        fit = self._weight_fit
        return {
            "omega_margin": float(fit.weights[0]),
            "omega_gradient": float(fit.weights[1]),
            "omega_identity": float(fit.weights[2]),
            "weight_objective": (
                None if fit.objective is None else float(fit.objective)
            ),
            "weight_iterations": fit.iterations,
            "weight_converged": fit.converged,
            "weight_learned": fit.learned,
            "weight_update_reason": fit.reason,
        }

    def trusted_mask(self, queried: np.ndarray) -> np.ndarray:
        return np.asarray(queried, dtype=bool).copy()

    def training_transition(
        self,
        previous_artifacts,
        current_labels: np.ndarray,
        queried: np.ndarray,
    ) -> tuple[None, None]:
        del previous_artifacts, current_labels, queried
        return None, None

    def register_pseudo_labels(
        self,
        ids: np.ndarray | None,
        predicted_labels: np.ndarray,
        current_labels: np.ndarray,
    ) -> None:
        del predicted_labels, current_labels
        if ids is not None and len(ids):
            raise ValueError("GRAM does not produce pseudo labels")

    def predict(
        self,
        artifacts,
        current_labels: np.ndarray,
        queried: np.ndarray,
        queried_status: np.ndarray,
    ) -> MethodPrediction:
        del current_labels
        score, _ = self._posterior(artifacts, queried, queried_status)
        return MethodPrediction(
            score, score >= 0.5, "status_posterior_at_least_0.5"
        )

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
        del current_labels
        probability, variance = self._posterior(
            artifacts, queried, queried_status
        )
        if self.config["acquisition"] == "posterior_variance":
            acquisition_score = variance
        elif self.config["acquisition"] == "latent_ucb":
            if self._posterior_mean is None:
                raise ValueError("latent UCB requires Gaussian-surrogate inference")
            acquisition_score = self._posterior_mean + self.config[
                "ucb_beta"
            ] * np.sqrt(np.maximum(variance, 0.0))
        elif self.config["acquisition"] == "ucb":
            acquisition_score = probability + np.sqrt(np.maximum(variance, 0.0))
        else:
            raise ValueError(
                f"unknown GRAM acquisition function {self.config['acquisition']!r}"
            )
        candidates = query_candidates(queried, candidate_mask)
        if count > len(candidates):
            raise ValueError("query count exceeds the number of eligible samples")
        selected = top_scores(acquisition_score, candidates, count, seed)
        return QueryResult(selected, acquisition_score, variance)
