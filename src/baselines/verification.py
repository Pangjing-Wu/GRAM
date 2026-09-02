"""Verification-aware baselines adapted to the benchmark's shared backbone.

The original papers use different architectures and data loaders.  This module
keeps their acquisition/update semantics while consuming the common training
artifacts, query mask, and oracle feedback used by both benchmark protocols.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..utils.neighbors import faiss_neighbors


def observed_label_risk(artifacts, noisy_labels: np.ndarray) -> np.ndarray:
    probability = np.asarray(artifacts.probability, dtype=np.float64)
    labels = np.asarray(noisy_labels, dtype=np.int64)
    return 1.0 - probability[np.arange(len(labels)), labels]


def active_label_cleaning_score(
    probability: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Active Label Cleaning's joint label-correctness acquisition score."""
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    normalized_log_probability = np.log(probability + 1.0e-12) / np.log(
        probability.shape[1]
    )
    noisiness = -normalized_log_probability[np.arange(len(labels)), labels]
    ambiguity = -np.sum(probability * normalized_log_probability, axis=-1)
    # PosteriorBasedSelector(JOINT) clips annotation difficulty at gamma=0.30.
    annotation_difficulty = np.maximum(ambiguity - 0.30, 0.0)
    return noisiness - annotation_difficulty


def estimate_noise_transition(
    probability: np.ndarray,
    original_noisy_labels: np.ndarray,
    trusted_labels: np.ndarray,
    trusted_mask: np.ndarray,
    *,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Estimate P(noisy=b | true=a) from soft and verified assignments."""
    probability = np.asarray(probability, dtype=np.float64)
    noisy = np.asarray(original_noisy_labels, dtype=np.int64)
    trusted_labels = np.asarray(trusted_labels, dtype=np.int64)
    trusted = np.asarray(trusted_mask, dtype=bool)
    classes = probability.shape[1]
    counts = np.full((classes, classes), smoothing, dtype=np.float64)
    untrusted = ~trusted
    if np.any(untrusted):
        for noisy_class in range(classes):
            ids = untrusted & (noisy == noisy_class)
            if np.any(ids):
                counts[:, noisy_class] += probability[ids].sum(axis=0)
    if np.any(trusted):
        # Trusted oracle/pseudo examples receive enough weight to influence the
        # transition without erasing the soft estimate on large datasets.
        trusted_weight = max(1.0, float(untrusted.sum()) / max(float(trusted.sum()), 1.0))
        np.add.at(
            counts,
            (trusted_labels[trusted], noisy[trusted]),
            trusted_weight,
        )
    return counts / counts.sum(axis=1, keepdims=True)


def noise_adjusted_posterior(
    probability: np.ndarray,
    noisy_labels: np.ndarray,
    transition: np.ndarray,
    trusted_labels: np.ndarray | None = None,
    trusted_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute P(true class | x, observed noisy class) by Bayes correction."""
    probability = np.asarray(probability, dtype=np.float64)
    noisy = np.asarray(noisy_labels, dtype=np.int64)
    likelihood = transition[:, noisy].T
    posterior = probability * likelihood
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1.0e-12)
    if trusted_mask is not None and trusted_labels is not None:
        trusted = np.asarray(trusted_mask, dtype=bool)
        ids = np.flatnonzero(trusted)
        posterior[ids] = 0.0
        posterior[ids, np.asarray(trusted_labels, dtype=np.int64)[ids]] = 1.0
    return posterior


def robust_alc_acquisition(
    artifacts,
    original_noisy_labels: np.ndarray,
    current_labels: np.ndarray,
    trusted_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiclass expected-gradient-length criterion from Robust ALC."""
    transition = estimate_noise_transition(
        artifacts.probability,
        original_noisy_labels,
        current_labels,
        trusted_mask,
    )
    posterior = noise_adjusted_posterior(
        artifacts.probability,
        original_noisy_labels,
        transition,
        current_labels,
        trusted_mask,
    )
    noisy = np.asarray(original_noisy_labels, dtype=np.int64)
    correction_probability = 1.0 - posterior[np.arange(len(noisy)), noisy]
    feature_norm = np.sqrt(
        1.0 + np.sum(np.asarray(artifacts.embedding, dtype=np.float64) ** 2, axis=1)
    )
    return correction_probability * feature_norm, transition


def dalc_quantities(
    artifacts,
    original_noisy_labels: np.ndarray,
    current_labels: np.ndarray,
    trusted_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transition = estimate_noise_transition(
        artifacts.probability,
        original_noisy_labels,
        current_labels,
        trusted_mask,
    )
    posterior = noise_adjusted_posterior(
        artifacts.probability,
        original_noisy_labels,
        transition,
        current_labels,
        trusted_mask,
    )
    entropy = -np.sum(
        np.clip(posterior, 1.0e-12, 1.0)
        * np.log(np.clip(posterior, 1.0e-12, 1.0)),
        axis=1,
    )
    return posterior, entropy, transition


def calibrate_score(
    base_risk: np.ndarray, queried: np.ndarray, queried_status: np.ndarray
) -> np.ndarray:
    risk = np.asarray(base_risk, dtype=np.float64)
    ids = np.flatnonzero(queried)
    if len(ids) < 2 or len(np.unique(queried_status[ids])) < 2:
        return _percentile(risk)
    model = IsotonicRegression(y_min=1.0e-4, y_max=1.0 - 1.0e-4, out_of_bounds="clip")
    return np.asarray(model.fit(risk[ids], queried_status[ids]).predict(risk))


def stratified_score_selection(
    base_risk: np.ndarray,
    candidates: np.ndarray,
    count: int,
    seed: int,
    bins: int = 10,
) -> np.ndarray:
    """Sample evenly across score quantiles for AUM-B-style calibration."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if count > len(candidates):
        raise ValueError("query count exceeds eligible calibration candidates")
    rng = np.random.default_rng(seed)
    order = candidates[np.argsort(base_risk[candidates], kind="stable")]
    groups = [group for group in np.array_split(order, min(bins, len(order))) if len(group)]
    allocation = np.full(len(groups), count // len(groups), dtype=np.int64)
    allocation[: count % len(groups)] += 1
    selected: list[int] = []
    leftovers: list[int] = []
    for group, take in zip(groups, allocation):
        shuffled = rng.permutation(group)
        actual = min(int(take), len(group))
        selected.extend(shuffled[:actual].tolist())
        leftovers.extend(shuffled[actual:].tolist())
    if len(selected) < count:
        selected.extend(rng.permutation(leftovers)[: count - len(selected)].tolist())
    return np.asarray(selected, dtype=np.int64)


def build_label_propagation_graph(
    embedding: np.ndarray, neighbors: int = 10
) -> sparse.csr_matrix:
    embedding = np.asarray(embedding, dtype=np.float64)
    k = min(neighbors, len(embedding) - 1)
    indices, squared_distance = faiss_neighbors(embedding, k, normalize=True)
    positive_distance = squared_distance[squared_distance > 0]
    sigma = (
        float(np.median(positive_distance))
        if len(positive_distance)
        else np.finfo(np.float64).eps
    )
    row = np.repeat(np.arange(len(embedding)), k)
    column = indices.reshape(-1)
    weight = np.exp(-squared_distance.reshape(-1) / max(sigma, 1.0e-12))
    directed = sparse.csr_matrix(
        (weight, (row, column)), shape=(len(embedding), len(embedding))
    )
    adjacency = directed.maximum(directed.T).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return sparse.diags(1.0 / np.maximum(degree, 1.0e-12)) @ adjacency


def graph_label_propagation(
    transition: sparse.csr_matrix,
    prior_risk: np.ndarray,
    queried: np.ndarray,
    queried_status: np.ndarray,
    *,
    alpha: float = 0.9,
    iterations: int = 100,
) -> np.ndarray:
    prior = np.clip(_percentile(prior_risk), 1.0e-4, 1.0 - 1.0e-4)
    probability = prior.copy()
    ids = np.flatnonzero(queried)
    for _ in range(iterations):
        update = alpha * (transition @ probability) + (1.0 - alpha) * prior
        update[ids] = queried_status[ids]
        if np.max(np.abs(update - probability)) < 1.0e-8:
            probability = update
            break
        probability = update
    return np.clip(probability, 0.0, 1.0)


def _detector_fit_predict(
    features: np.ndarray,
    train_ids: np.ndarray,
    targets: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    fallback: np.ndarray,
) -> np.ndarray:
    if len(train_ids) < 2 or len(np.unique(targets)) < 2:
        return np.asarray(fallback, dtype=np.float64)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(
        features[train_ids],
        targets,
        sample_weight=sample_weight,
    )
    return model.predict_proba(features)[:, 1]


def cleannet_score(
    artifacts,
    noisy_labels: np.ndarray,
    queried: np.ndarray,
    queried_status: np.ndarray,
    seed: int,
) -> np.ndarray:
    """CleanNet-style shared relevance head with B-limited clean references."""
    labels = np.asarray(noisy_labels, dtype=np.int64)
    embedding = np.asarray(artifacts.embedding, dtype=np.float64)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1.0e-12)
    similarity = np.zeros(len(labels), dtype=np.float64)
    has_reference = np.zeros(len(labels), dtype=np.float64)
    verified_clean = queried & (queried_status == 0)
    for label in np.unique(labels):
        reference_ids = np.flatnonzero(verified_clean & (labels == label))
        sample_ids = np.flatnonzero(labels == label)
        if len(reference_ids):
            prototype = embedding[reference_ids].mean(axis=0)
            prototype /= max(np.linalg.norm(prototype), 1.0e-12)
            similarity[sample_ids] = embedding[sample_ids] @ prototype
            has_reference[sample_ids] = 1.0
    base = observed_label_risk(artifacts, labels)
    features = np.column_stack((similarity, has_reference, base, artifacts.early_loss))
    ids = np.flatnonzero(queried)
    return _detector_fit_predict(
        _standardize(features),
        ids,
        queried_status[ids],
        None,
        seed,
        _percentile(base),
    )


def cleannet_acquisition(
    score: np.ndarray,
    noisy_labels: np.ndarray,
    queried: np.ndarray,
) -> np.ndarray:
    uncertainty = 1.0 - np.abs(2.0 * np.asarray(score) - 1.0)
    labels = np.asarray(noisy_labels, dtype=np.int64)
    clean_reference_count = np.bincount(
        labels[queried], minlength=int(labels.max()) + 1
    )
    return uncertainty + 0.05 / (1.0 + clean_reference_count[labels])


def misdetect_features(artifacts, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(artifacts.probability, dtype=np.float64)
    observed = probability[np.arange(len(labels)), labels]
    one_hot = np.eye(probability.shape[1])[labels]
    residual_norm = np.linalg.norm(probability - one_hot, axis=1)
    head_gradient_norm = residual_norm * np.sqrt(
        1.0 + np.sum(np.asarray(artifacts.embedding, dtype=np.float64) ** 2, axis=1)
    )
    features = np.column_stack(
        (artifacts.early_loss, artifacts.final_loss, 1.0 - observed, head_gradient_norm)
    )
    return _standardize(features), head_gradient_norm


def misdetect_score(
    artifacts,
    noisy_labels: np.ndarray,
    queried: np.ndarray,
    queried_status: np.ndarray,
    seed: int,
) -> np.ndarray:
    features, _ = misdetect_features(artifacts, noisy_labels)
    early = np.asarray(artifacts.early_loss, dtype=np.float64)
    lower, upper = early.mean() - early.std(), early.mean() + early.std()
    pseudo_clean = early <= lower
    pseudo_dirty = early >= upper
    pseudo_ids = np.flatnonzero((pseudo_clean | pseudo_dirty) & ~queried)
    ids = np.concatenate((pseudo_ids, np.flatnonzero(queried)))
    targets = np.concatenate(
        (pseudo_dirty[pseudo_ids].astype(np.int64), queried_status[queried])
    )
    weight = np.concatenate(
        (
            np.ones(len(pseudo_ids), dtype=np.float64),
            np.full(int(queried.sum()), max(10.0, len(pseudo_ids) / max(int(queried.sum()), 1))),
        )
    )
    return _detector_fit_predict(
        features,
        ids,
        targets,
        weight,
        seed,
        _percentile(early),
    )


def misdetect_acquisition(
    artifacts, noisy_labels: np.ndarray, detector_score: np.ndarray
) -> np.ndarray:
    _, influence = misdetect_features(artifacts, noisy_labels)
    early = np.asarray(artifacts.early_loss, dtype=np.float64)
    ambiguous = np.abs(early - early.mean()) < early.std()
    influence = _percentile(influence)
    uncertainty = 1.0 - np.abs(2.0 * detector_score - 1.0)
    return np.where(ambiguous, influence + uncertainty, 0.1 * influence)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(scale > 1.0e-12, scale, 1.0)


def _percentile(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    return rankdata(score, method="average") / (len(score) + 1.0)
