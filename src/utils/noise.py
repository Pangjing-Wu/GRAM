from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


def _flip_ids(sample_count: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    count = int(np.floor(rho * sample_count + 0.5))
    return np.sort(rng.choice(sample_count, size=count, replace=False))


def inject_noise(
    labels: np.ndarray,
    features,
    *,
    noise_type: str,
    rho: float,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject exactly ``round(rho * N)`` label changes.

    ``symmetric`` samples IDs uniformly and assigns each a uniformly selected
    different class. ``pairflip`` maps a selected class ``c`` to
    ``(c + 1) mod C``. ``instance`` uses seeded linear class scores over
    normalized noise features: the target is the highest-scoring non-clean
    class and the sampling weight grows with its score margin over the clean
    class. All IDs are sampled without replacement.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if noise_type not in {"symmetric", "pairflip", "instance"}:
        raise ValueError("noise_type must be symmetric, pairflip, or instance")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1701]))
    noisy = labels.copy()

    if noise_type == "instance":
        matrix = normalize(features, norm="l2", copy=True)
        dimension = matrix.shape[1]
        generator = rng.normal(
            0.0, 1.0 / np.sqrt(dimension), size=(dimension, num_classes)
        )
        logits = matrix @ generator
        logits = np.asarray(logits, dtype=np.float64)
        clean_score = logits[np.arange(len(labels)), labels].copy()
        logits[np.arange(len(labels)), labels] = -np.inf
        target = logits.argmax(axis=1)
        margin = logits.max(axis=1) - clean_score
        weights = np.exp(np.clip(margin - margin.max(), -30.0, 0.0))
        count = int(np.floor(rho * len(labels) + 0.5))
        ids = np.sort(
            rng.choice(
                len(labels), size=count, replace=False, p=weights / weights.sum()
            )
        )
        noisy[ids] = target[ids]
    else:
        ids = _flip_ids(len(labels), rho, rng)
        if noise_type == "pairflip":
            noisy[ids] = (labels[ids] + 1) % num_classes
        else:
            offsets = rng.integers(1, num_classes, size=len(ids))
            noisy[ids] = (labels[ids] + offsets) % num_classes

    mask = noisy != labels
    expected = int(np.floor(rho * len(labels) + 0.5))
    if int(mask.sum()) != expected:
        raise RuntimeError("noise injection did not produce the requested number of flips")
    return noisy, mask
