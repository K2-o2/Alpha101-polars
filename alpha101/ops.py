"""Polars-native building blocks for Alpha101 formulas."""

from __future__ import annotations

import polars as pl

from alpha101 import schema


def col(name: str | pl.Expr) -> pl.Expr:
    if isinstance(name, pl.Expr):
        return name
    return pl.col(name)


def signed_power(expr: str | pl.Expr, power: float) -> pl.Expr:
    expr = col(expr)
    return expr.sign() * expr.abs().pow(power)


def delay(expr: str | pl.Expr, period: int, symbol_col: str = schema.SYMBOL) -> pl.Expr:
    return col(expr).shift(period).over(symbol_col)


def delta(expr: str | pl.Expr, period: int, symbol_col: str = schema.SYMBOL) -> pl.Expr:
    expr = col(expr)
    return expr - delay(expr, period, symbol_col)


def ts_sum(
    expr: str | pl.Expr, window: int, symbol_col: str = schema.SYMBOL
) -> pl.Expr:
    return col(expr).rolling_sum(window).over(symbol_col)


def ts_product(
    expr: str | pl.Expr, window: int, symbol_col: str = schema.SYMBOL
) -> pl.Expr:
    return (
        col(expr)
        .rolling_map(lambda values: values.product(), window_size=window)
        .over(symbol_col)
    )


def decay_linear(
    expr: str | pl.Expr, window: int, symbol_col: str = schema.SYMBOL
) -> pl.Expr:
    weights = list(range(1, window + 1))
    weight_sum = sum(weights)
    return (
        col(expr)
        .rolling_map(
            lambda values: sum(value * weight for value, weight in zip(values, weights))
            / weight_sum,
            window_size=window,
        )
        .over(symbol_col)
    )


def ts_mean(
    expr: str | pl.Expr, window: int, symbol_col: str = schema.SYMBOL
) -> pl.Expr:
    return col(expr).rolling_mean(window).over(symbol_col)


def ts_std(
    expr: str | pl.Expr, window: int, symbol_col: str = schema.SYMBOL
) -> pl.Expr:
    return col(expr).rolling_std(window).over(symbol_col)


def ts_min(
    expr: str | pl.Expr, window: int, symbol_col: str = schema.SYMBOL
) -> pl.Expr:
    return col(expr).rolling_min(window).over(symbol_col)


def ts_max(
    expr: str | pl.Expr, window: int, symbol_col: str = schema.SYMBOL
) -> pl.Expr:
    return col(expr).rolling_max(window).over(symbol_col)


def ts_rank(
    expr: str | pl.Expr, window: int, symbol_col: str = schema.SYMBOL
) -> pl.Expr:
    return col(expr).rolling_rank(window, method="average").over(symbol_col)


def ts_arg_max(
    expr: str | pl.Expr,
    window: int,
    symbol_col: str = schema.SYMBOL,
) -> pl.Expr:
    return (
        col(expr)
        .rolling_map(lambda values: values.arg_max() + 1, window_size=window)
        .over(symbol_col)
    )


def ts_argmax(
    expr: str | pl.Expr, window: int, symbol_col: str = schema.SYMBOL
) -> pl.Expr:
    return ts_arg_max(expr, window, symbol_col)


def rank(expr: str | pl.Expr, date_col: str = schema.DATE) -> pl.Expr:
    expr = col(expr)
    return expr.rank(method="average").over(date_col) / expr.count().over(date_col)


def ts_corr(
    left: str | pl.Expr,
    right: str | pl.Expr,
    window: int,
    symbol_col: str = schema.SYMBOL,
) -> pl.Expr:
    left_expr = col(left)
    right_expr = col(right)
    left_mean = ts_mean(left_expr, window, symbol_col)
    right_mean = ts_mean(right_expr, window, symbol_col)
    cov = ts_cov(left_expr, right_expr, window, symbol_col)
    left_var = ts_mean(left_expr.pow(2), window, symbol_col) - left_mean.pow(2)
    right_var = ts_mean(right_expr.pow(2), window, symbol_col) - right_mean.pow(2)
    denominator = left_var.sqrt() * right_var.sqrt()
    corr = cov / denominator
    return pl.when((denominator > 0) & corr.is_finite()).then(corr).otherwise(None)


def ts_cov(
    left: str | pl.Expr,
    right: str | pl.Expr,
    window: int,
    symbol_col: str = schema.SYMBOL,
) -> pl.Expr:
    left_expr = col(left)
    right_expr = col(right)
    return ts_mean(left_expr * right_expr, window, symbol_col) - (
        ts_mean(left_expr, window, symbol_col) * ts_mean(right_expr, window, symbol_col)
    )


def scale(expr: str | pl.Expr, date_col: str = schema.DATE) -> pl.Expr:
    expr = col(expr)
    return expr / expr.abs().sum().over(date_col)
