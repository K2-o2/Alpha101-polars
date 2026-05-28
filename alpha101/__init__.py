"""Polars-native Alpha101 factor expressions."""

from alpha101 import alphas as _alphas
from alpha101.pipeline import alpha_names, compute_alphas, get_alpha, get_alpha_factor

ALPHA_FACTORS = _alphas.ALPHA_FACTORS

_ALPHA_FUNCTION_NAMES = [f"alpha{i:03d}" for i in range(1, 102)]
for _name in _ALPHA_FUNCTION_NAMES:
    globals()[_name] = getattr(_alphas, _name)

__all__ = [
    "ALPHA_FACTORS",
    *_ALPHA_FUNCTION_NAMES,
    "alpha_names",
    "compute_alphas",
    "get_alpha",
    "get_alpha_factor",
]
