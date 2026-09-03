"""Helpers for normalizing persisted experiment results."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any


METRIC_DECIMAL_PLACES = 5


def round_metric_floats(value: Any) -> Any:
    """Recursively round floating-point metric values for serialization."""
    if isinstance(value, dict):
        return {key: round_metric_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_metric_floats(item) for item in value]
    if isinstance(value, tuple):
        return tuple(round_metric_floats(item) for item in value)
    if isinstance(value, Real) and not isinstance(value, (Integral, bool)):
        return round(float(value), METRIC_DECIMAL_PLACES)
    return value
