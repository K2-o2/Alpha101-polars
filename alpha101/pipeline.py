"""Helpers for computing Alpha101 expressions on Polars frames."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from alpha101.alphas import ALPHA_FACTORS
from alpha101.factor import AlphaFactor
from alpha101.schema import DATE, REQUIRED_COLUMNS, SYMBOL


def alpha_names() -> list[str]:
    return sorted(ALPHA_FACTORS)


def get_alpha(name: str) -> AlphaFactor:
    try:
        return ALPHA_FACTORS[name]()
    except KeyError as exc:
        available = ", ".join(alpha_names())
        raise ValueError(
            f"Unknown alpha {name!r}. Available alphas: {available}"
        ) from exc


def get_alpha_factor(name: str) -> AlphaFactor:
    return get_alpha(name)


def validate_schema(frame: pl.DataFrame | pl.LazyFrame) -> None:
    columns = set(
        frame.collect_schema().names()
        if isinstance(frame, pl.LazyFrame)
        else frame.columns
    )
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def compute_alphas(
    frame: pl.DataFrame | pl.LazyFrame,
    names: Iterable[str] | None = None,
    sort: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    validate_schema(frame)
    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    if sort:
        lazy = lazy.sort([SYMBOL, DATE])
    factors = [
        get_alpha(name) for name in (alpha_names() if names is None else list(names))
    ]
    for factor in factors:
        for stage in factor.stages:
            lazy = lazy.with_columns(list(stage))
    result = lazy.with_columns([factor.expr.alias(factor.name) for factor in factors])
    temporary_columns = sorted(
        {column for factor in factors for column in factor.temporary_columns}
    )
    if temporary_columns:
        result = result.drop(temporary_columns)
    return result.collect() if isinstance(frame, pl.DataFrame) else result
