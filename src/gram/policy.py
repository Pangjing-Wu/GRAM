"""GRAM graph-prior construction, posterior inference, and acquisition policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

from ..utils.neighbors import faiss_neighbors
from ..utils.policy import MethodPrediction, QueryResult, query_candidates, top_scores


GRAM_METHOD = "ours"


@dataclass
class PriorGraph:
    neighbors: np.ndarray
    features: np.ndarray


def diagnostic_priors(artifacts, labels: np.ndarray, k: int = 10):
    labels = np.asarray(labels, dtype=np.int64)
    confidence = 1.0 - artifacts.probability[np.arange(len(labels)), labels]
    neighbors, _ = faiss_neighbors(
        artifacts.embedding, min(k, len(labels) - 1), normalize=True
    )
    disagreement = 1.0 - (labels[neighbors] == labels[:, None]).mean(axis=1)
    diagnostics = np.column_stack((confidence, artifacts.early_loss, disagreement))
    return diagnostics, neighbors


def _knn_graph(values: np.ndarray, k: int = 10):
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


def build_prior_graph(
    diagnostics: np.ndarray, rank: int = 64, k: int = 10
) -> PriorGraph:
    adjacency, neighbors = _knn_graph(diagnostics, k)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = sparse.diags(1.0 / np.sqrt(np.maximum(degree, 1.0e-12)))
    normalized_adjacency = inverse @ adjacency @ inverse
    sample_count = len(diagnostics)
    active_rank = min(rank, sample_count - 1)
    if sample_count <= rank + 1:
        values, vectors = np.linalg.eigh(normalized_adjacency.toarray())
        values, vectors = values[-active_rank:], vectors[:, -active_rank:]
    else:
        values, vectors = eigsh(
            normalized_adjacency, k=active_rank, which="LA"
        )
    order = np.argsort(values)[::-1]
    laplacian_values = np.maximum(1.0 - values[order], 0.0)
    vectors = vectors[:, order]
    features = (
        vectors
        * np.exp(-0.5 * laplacian_values)[None, :]
        * math.sqrt(sample_count / active_rank)
    )
    return PriorGraph(neighbors=neighbors, features=features)


class GraphGPClassifier:
    def __init__(self, features: np.ndarray, prior_mean: np.ndarray):
        self.features = np.asarray(features, dtype=np.float64)
        self.prior_mean = np.asarray(prior_mean, dtype=np.float64)

    def posterior(self, observed_ids: np.ndarray, values: np.ndarray):
        rank = self.features.shape[1]
        weight = np.zeros(rank, dtype=np.float64)
        if len(observed_ids):
            observed = self.features[observed_ids]
            offset = self.prior_mean[observed_ids]
            for _ in range(50):
                probability = expit(offset + observed @ weight)
                curvature = np.maximum(probability * (1.0 - probability), 1.0e-8)
                precision = np.eye(rank) + observed.T @ (curvature[:, None] * observed)
                gradient = weight + observed.T @ (probability - values)
                factor = linalg.cholesky(precision, lower=True, check_finite=False)
                step = linalg.cho_solve(
                    (factor, True), gradient, check_finite=False
                )
                weight -= step
                if np.max(np.abs(step)) < 1.0e-8:
                    break
            probability = expit(offset + observed @ weight)
            curvature = np.maximum(probability * (1.0 - probability), 1.0e-8)
            precision = np.eye(rank) + observed.T @ (curvature[:, None] * observed)
            factor = linalg.cholesky(precision, lower=True, check_finite=False)
        else:
            factor = np.eye(rank)
        latent_mean = self.prior_mean + self.features @ weight
        latent = linalg.solve_triangular(
            factor, self.features.T, lower=True, check_finite=False
        ).T
        variance = np.einsum("ij,ij->i", latent, latent)
        probability = expit(
            latent_mean / np.sqrt(1.0 + math.pi * variance / 8.0)
        )
        return probability, variance


def prior_mean(diagnostics: np.ndarray) -> np.ndarray:
    combined = diagnostics.mean(axis=1)
    probability = rankdata(combined, method="average") / (len(combined) + 1.0)
    probability = np.clip(probability, 1.0e-4, 1.0 - 1.0e-4)
    return np.log(probability / (1.0 - probability))


def predict_gram(
    diagnostics: np.ndarray,
    graph: PriorGraph,
    queried: np.ndarray,
    queried_status: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_ids = np.flatnonzero(queried)
    classifier = GraphGPClassifier(graph.features, prior_mean(diagnostics))
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
    candidate_mask: np.ndarray | None = None,
) -> QueryResult:
    probability, variance = predict_gram(
        diagnostics, graph, queried, queried_status
    )
    acquisition = probability + np.sqrt(np.maximum(variance, 0.0))
    candidates = query_candidates(queried, candidate_mask)
    if count > len(candidates):
        raise ValueError("query count exceeds the number of eligible samples")
    selected = top_scores(acquisition, candidates, count, seed)
    return QueryResult(selected, acquisition, variance)


class GramPolicy:
    """Stateful policy adapter for the proposed GRAM method."""

    def __init__(
        self,
        noisy_labels: np.ndarray,
        *,
        modality: str,
        seed: int,
        native_features: np.ndarray | None = None,
    ):
        del modality, native_features
        self.noisy_labels = np.asarray(noisy_labels, dtype=np.int64).copy()
        self.seed = int(seed)
        self.pseudo_mask = np.zeros(len(self.noisy_labels), dtype=bool)
        self.last_noise_transition = None
        self._diagnostics = None
        self._graph = None

    def _ensure_prior(self, artifacts) -> None:
        if self._diagnostics is None:
            self._diagnostics, _ = diagnostic_priors(
                artifacts, self.noisy_labels
            )
            self._graph = build_prior_graph(self._diagnostics)

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
        self._ensure_prior(artifacts)
        score, _ = predict_gram(
            self._diagnostics, self._graph, queried, queried_status
        )
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
        self._ensure_prior(artifacts)
        return select_gram(
            self._diagnostics,
            self._graph,
            queried,
            queried_status,
            count=count,
            seed=seed,
            candidate_mask=candidate_mask,
        )
