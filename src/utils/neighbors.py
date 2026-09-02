from __future__ import annotations

import numpy as np


def faiss_neighbors(values: np.ndarray, k: int, *, normalize: bool):
    import faiss

    matrix = np.ascontiguousarray(values, dtype=np.float32).copy()
    if normalize:
        faiss.normalize_L2(matrix)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(matrix.shape[1], 32, faiss.METRIC_L2)
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = max(64, 4 * (k + 1))
    index.add(matrix)
    distance, indices = index.search(matrix, min(k + 2, len(matrix)))
    selected_ids = np.empty((len(matrix), k), dtype=np.int64)
    selected_distance = np.empty((len(matrix), k), dtype=np.float64)
    for row in range(len(matrix)):
        keep = indices[row] != row
        ids = indices[row][keep][:k]
        distances = distance[row][keep][:k]
        if len(ids) != k:
            raise RuntimeError("nearest-neighbor search returned too few neighbors")
        selected_ids[row] = ids
        selected_distance[row] = distances
    return selected_ids, selected_distance


def pairwise_distance_scale(
    values: np.ndarray,
    *,
    seed: int,
    pair_count: int,
) -> float:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(matrix) < 2:
        raise ValueError("pairwise distance scale requires at least two samples")
    rng = np.random.default_rng(seed)
    total = 0.0
    processed = 0
    chunk_size = 4096
    while processed < pair_count:
        size = min(chunk_size, pair_count - processed)
        left = rng.integers(0, len(matrix), size=size)
        right = (left + rng.integers(1, len(matrix), size=size)) % len(matrix)
        delta = matrix[left] - matrix[right]
        total += float(np.sqrt(np.einsum("ij,ij->i", delta, delta)).sum())
        processed += size
    mean_distance = total / pair_count
    return max(mean_distance * mean_distance, np.finfo(np.float64).eps)
