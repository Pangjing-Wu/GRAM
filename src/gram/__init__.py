"""Implementation of the proposed GRAM method."""

from config.methods import DEFAULT_GRAM_VARIANT, GRAM_VARIANTS

from .policy import GRAM_METHOD, GramPolicy

__all__ = [
    "DEFAULT_GRAM_VARIANT",
    "GRAM_METHOD",
    "GRAM_VARIANTS",
    "GramPolicy",
]
