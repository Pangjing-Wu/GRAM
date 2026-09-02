"""Shared cumulative verification-budget schedule utilities."""

from __future__ import annotations

import numpy as np


DEFAULT_BUDGET_INTERVAL = 0.05
DENSE_LOW_BUDGET_CHECKPOINTS = (0.001, 0.005, 0.01, 0.025)


def noise_rate_budget_checkpoints(
    noise_rate: float, *, interval: float = DEFAULT_BUDGET_INTERVAL
) -> tuple[float, ...]:
    """Return dense low-budget checkpoints plus a regular grid through the noise rate."""
    noise_rate = float(noise_rate)
    interval = float(interval)
    if not np.isfinite(noise_rate) or not 0.0 < noise_rate <= 1.0:
        raise ValueError("noise rate must lie in (0, 1]")
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("budget interval must be positive and finite")
    step_count = int(np.floor(noise_rate / interval + 0.5))
    if step_count <= 0 or not np.isclose(
        step_count * interval, noise_rate, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("noise rate must be an integer multiple of the budget interval")
    checkpoints = {
        *(
            value
            for value in DENSE_LOW_BUDGET_CHECKPOINTS
            if value <= noise_rate + 1.0e-12
        ),
        *(round(step * interval, 12) for step in range(1, step_count + 1)),
    }
    return tuple(sorted(checkpoints))


def validate_budget_checkpoints(values, *, maximum: float = 1.0) -> tuple[float, ...]:
    budgets = tuple(float(value) for value in values)
    if not budgets:
        raise ValueError("at least one positive budget checkpoint is required")
    array = np.asarray(budgets, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("budget checkpoints must be finite")
    if np.any(array <= 0.0) or np.any(array > maximum + 1.0e-12):
        raise ValueError(f"budget checkpoints must lie in (0, {maximum:g}]")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError("budget checkpoints must be strictly increasing")
    return budgets


def rounded_budget_count(sample_count: int, fraction: float) -> int:
    return int(np.floor(float(fraction) * sample_count + 0.5))


def budget_schedule(
    sample_count: int, checkpoints: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative verification counts and successive query increments."""
    targets = np.asarray(
        [0, *(rounded_budget_count(sample_count, value) for value in checkpoints)],
        dtype=np.int64,
    )
    increments = np.diff(targets)
    if np.any(increments <= 0):
        raise ValueError(
            "adjacent budget checkpoints round to the same sample count; "
            "increase their spacing for this dataset"
        )
    return targets, increments


def budget_tag(checkpoints: tuple[float, ...]) -> str:
    def tag(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    return "-".join(tag(value) for value in checkpoints)
