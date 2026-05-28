"""Representative WorldQuant Alpha101 factors expressed as staged Polars plans."""

from __future__ import annotations

import polars as pl

from alpha101 import schema
from alpha101.factor import AlphaFactor
from alpha101.ops import (
    delta,
    delay,
    rank,
    signed_power,
    ts_arg_max,
    ts_corr,
    ts_cov,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std,
    ts_sum,
)


def _tmp(alpha: str, name: str) -> str:
    return f"__{alpha}_{name}"


def alpha001() -> AlphaFactor:
    arg_max = _tmp("alpha001", "arg_max")
    base = (
        pl.when(pl.col(schema.RETURNS) < 0)
        .then(ts_std(schema.RETURNS, 20))
        .otherwise(pl.col(schema.CLOSE))
    )
    return AlphaFactor(
        name="alpha001",
        stages=((ts_arg_max(signed_power(base, 2), 5).alias(arg_max),),),
        expr=rank(arg_max) - 0.5,
        temporary_columns=(arg_max,),
    )


def alpha002() -> AlphaFactor:
    delta_log_volume = _tmp("alpha002", "delta_log_volume")
    intraday_return = _tmp("alpha002", "intraday_return")
    left = _tmp("alpha002", "rank_delta_log_volume")
    right = _tmp("alpha002", "rank_intraday_return")
    return AlphaFactor(
        name="alpha002",
        stages=(
            (
                delta(pl.col(schema.VOLUME).log(), 2).alias(delta_log_volume),
                ((pl.col(schema.CLOSE) - pl.col(schema.OPEN)) / pl.col(schema.OPEN)).alias(intraday_return),
            ),
            (
                rank(delta_log_volume).alias(left),
                rank(intraday_return).alias(right),
            ),
        ),
        expr=-ts_corr(left, right, 6),
        temporary_columns=(delta_log_volume, intraday_return, left, right),
    )


def alpha003() -> AlphaFactor:
    # With a tiny universe, cross-sectional ranks can stay constant for many days.
    # In that case the rolling correlation is undefined and ts_corr returns null.
    left = _tmp("alpha003", "rank_open")
    right = _tmp("alpha003", "rank_volume")
    return AlphaFactor(
        name="alpha003",
        stages=((rank(schema.OPEN).alias(left), rank(schema.VOLUME).alias(right)),),
        expr=-ts_corr(left, right, 10),
        temporary_columns=(left, right),
    )


def alpha004() -> AlphaFactor:
    ranked_low = _tmp("alpha004", "rank_low")
    return AlphaFactor(
        name="alpha004",
        stages=((rank(schema.LOW).alias(ranked_low),),),
        expr=-ts_rank(ranked_low, 9),
        temporary_columns=(ranked_low,),
    )


def alpha005() -> AlphaFactor:
    open_minus_mean_vwap = _tmp("alpha005", "open_minus_mean_vwap")
    close_minus_vwap = _tmp("alpha005", "close_minus_vwap")
    ranked_open_minus_mean_vwap = _tmp("alpha005", "rank_open_minus_mean_vwap")
    ranked_close_minus_vwap = _tmp("alpha005", "rank_close_minus_vwap")
    return AlphaFactor(
        name="alpha005",
        stages=(
            (
                (pl.col(schema.OPEN) - ts_mean(schema.VWAP, 10)).alias(open_minus_mean_vwap),
                (pl.col(schema.CLOSE) - pl.col(schema.VWAP)).alias(close_minus_vwap),
            ),
            (
                rank(open_minus_mean_vwap).alias(ranked_open_minus_mean_vwap),
                rank(close_minus_vwap).alias(ranked_close_minus_vwap),
            ),
        ),
        expr=pl.col(ranked_open_minus_mean_vwap) * -pl.col(ranked_close_minus_vwap).abs(),
        temporary_columns=(
            open_minus_mean_vwap,
            close_minus_vwap,
            ranked_open_minus_mean_vwap,
            ranked_close_minus_vwap,
        ),
    )


def alpha006() -> AlphaFactor:
    return AlphaFactor(
        name="alpha006",
        stages=(),
        expr=-ts_corr(schema.OPEN, schema.VOLUME, 10),
    )


def alpha007() -> AlphaFactor:
    adv20 = _tmp("alpha007", "adv20")
    close_delta = _tmp("alpha007", "delta_close_7")
    ranked_abs_delta = _tmp("alpha007", "ts_rank_abs_delta")
    return AlphaFactor(
        name="alpha007",
        stages=(
            (
                ts_mean(schema.VOLUME, 20).alias(adv20),
                delta(schema.CLOSE, 7).alias(close_delta),
            ),
            (ts_rank(pl.col(close_delta).abs(), 60).alias(ranked_abs_delta),),
        ),
        expr=pl.when(pl.col(adv20) < pl.col(schema.VOLUME))
        .then(-pl.col(ranked_abs_delta) * pl.col(close_delta).sign())
        .otherwise(-1),
        temporary_columns=(adv20, close_delta, ranked_abs_delta),
    )


def alpha008() -> AlphaFactor:
    product = _tmp("alpha008", "open_return_product")
    diff = _tmp("alpha008", "diff")
    return AlphaFactor(
        name="alpha008",
        stages=(
            ((ts_sum(schema.OPEN, 5) * ts_sum(schema.RETURNS, 5)).alias(product),),
            ((pl.col(product) - delay(product, 10)).alias(diff),),
        ),
        expr=-rank(diff),
        temporary_columns=(product, diff),
    )


def alpha009() -> AlphaFactor:
    close_delta = _tmp("alpha009", "delta_close")
    delta_min = _tmp("alpha009", "delta_min")
    delta_max = _tmp("alpha009", "delta_max")
    return AlphaFactor(
        name="alpha009",
        stages=(
            (delta(schema.CLOSE, 1).alias(close_delta),),
            (
                ts_min(close_delta, 5).alias(delta_min),
                ts_max(close_delta, 5).alias(delta_max),
            ),
        ),
        expr=pl.when(pl.col(delta_min) > 0)
        .then(pl.col(close_delta))
        .when(pl.col(delta_max) < 0)
        .then(pl.col(close_delta))
        .otherwise(-pl.col(close_delta)),
        temporary_columns=(close_delta, delta_min, delta_max),
    )


def alpha010() -> AlphaFactor:
    close_delta = _tmp("alpha010", "delta_close")
    delta_min = _tmp("alpha010", "delta_min")
    delta_max = _tmp("alpha010", "delta_max")
    value = _tmp("alpha010", "value")
    return AlphaFactor(
        name="alpha010",
        stages=(
            (delta(schema.CLOSE, 1).alias(close_delta),),
            (
                ts_min(close_delta, 4).alias(delta_min),
                ts_max(close_delta, 4).alias(delta_max),
            ),
            (
                pl.when(pl.col(delta_min) > 0)
                .then(pl.col(close_delta))
                .when(pl.col(delta_max) < 0)
                .then(pl.col(close_delta))
                .otherwise(-pl.col(close_delta))
                .alias(value),
            ),
        ),
        expr=rank(value),
        temporary_columns=(close_delta, delta_min, delta_max, value),
    )


def alpha011() -> AlphaFactor:
    vwap_minus_close = _tmp("alpha011", "vwap_minus_close")
    max_spread = _tmp("alpha011", "max_spread")
    min_spread = _tmp("alpha011", "min_spread")
    volume_delta = _tmp("alpha011", "delta_volume")
    rank_max_spread = _tmp("alpha011", "rank_max_spread")
    rank_min_spread = _tmp("alpha011", "rank_min_spread")
    rank_volume_delta = _tmp("alpha011", "rank_volume_delta")
    return AlphaFactor(
        name="alpha011",
        stages=(
            (
                (pl.col(schema.VWAP) - pl.col(schema.CLOSE)).alias(vwap_minus_close),
                delta(schema.VOLUME, 3).alias(volume_delta),
            ),
            (
                ts_max(vwap_minus_close, 3).alias(max_spread),
                ts_min(vwap_minus_close, 3).alias(min_spread),
            ),
            (
                rank(max_spread).alias(rank_max_spread),
                rank(min_spread).alias(rank_min_spread),
                rank(volume_delta).alias(rank_volume_delta),
            ),
        ),
        expr=(pl.col(rank_max_spread) + pl.col(rank_min_spread)) * pl.col(rank_volume_delta),
        temporary_columns=(
            vwap_minus_close,
            max_spread,
            min_spread,
            volume_delta,
            rank_max_spread,
            rank_min_spread,
            rank_volume_delta,
        ),
    )


def alpha012() -> AlphaFactor:
    return AlphaFactor(
        name="alpha012",
        stages=(),
        expr=delta(schema.VOLUME, 1).sign() * -delta(schema.CLOSE, 1),
    )


def alpha013() -> AlphaFactor:
    ranked_close = _tmp("alpha013", "rank_close")
    ranked_volume = _tmp("alpha013", "rank_volume")
    cov = _tmp("alpha013", "cov")
    return AlphaFactor(
        name="alpha013",
        stages=(
            (rank(schema.CLOSE).alias(ranked_close), rank(schema.VOLUME).alias(ranked_volume)),
            (ts_cov(ranked_close, ranked_volume, 5).alias(cov),),
        ),
        expr=-rank(cov),
        temporary_columns=(ranked_close, ranked_volume, cov),
    )


def alpha014() -> AlphaFactor:
    returns_delta = _tmp("alpha014", "delta_returns")
    ranked_delta = _tmp("alpha014", "rank_delta_returns")
    return AlphaFactor(
        name="alpha014",
        stages=(
            (delta(schema.RETURNS, 3).alias(returns_delta),),
            (rank(returns_delta).alias(ranked_delta),),
        ),
        expr=-pl.col(ranked_delta) * ts_corr(schema.OPEN, schema.VOLUME, 10),
        temporary_columns=(returns_delta, ranked_delta),
    )


def alpha015() -> AlphaFactor:
    ranked_high = _tmp("alpha015", "rank_high")
    ranked_volume = _tmp("alpha015", "rank_volume")
    corr = _tmp("alpha015", "corr")
    ranked_corr = _tmp("alpha015", "rank_corr")
    corr_sum = _tmp("alpha015", "sum_rank_corr")
    return AlphaFactor(
        name="alpha015",
        stages=(
            (rank(schema.HIGH).alias(ranked_high), rank(schema.VOLUME).alias(ranked_volume)),
            (ts_corr(ranked_high, ranked_volume, 3).alias(corr),),
            (rank(corr).alias(ranked_corr),),
            (ts_sum(ranked_corr, 3).alias(corr_sum),),
        ),
        expr=-pl.col(corr_sum),
        temporary_columns=(ranked_high, ranked_volume, corr, ranked_corr, corr_sum),
    )


def alpha016() -> AlphaFactor:
    ranked_high = _tmp("alpha016", "rank_high")
    ranked_volume = _tmp("alpha016", "rank_volume")
    cov = _tmp("alpha016", "cov")
    return AlphaFactor(
        name="alpha016",
        stages=(
            (rank(schema.HIGH).alias(ranked_high), rank(schema.VOLUME).alias(ranked_volume)),
            (ts_cov(ranked_high, ranked_volume, 5).alias(cov),),
        ),
        expr=-rank(cov),
        temporary_columns=(ranked_high, ranked_volume, cov),
    )


def alpha017() -> AlphaFactor:
    adv20 = _tmp("alpha017", "adv20")
    volume_over_adv20 = _tmp("alpha017", "volume_over_adv20")
    close_ts_rank = _tmp("alpha017", "ts_rank_close")
    close_delta = _tmp("alpha017", "delta_close")
    close_delta_delta = _tmp("alpha017", "delta_delta_close")
    volume_adv_ts_rank = _tmp("alpha017", "ts_rank_volume_adv")
    ranked_close_ts_rank = _tmp("alpha017", "rank_ts_rank_close")
    ranked_close_delta_delta = _tmp("alpha017", "rank_delta_delta_close")
    ranked_volume_adv_ts_rank = _tmp("alpha017", "rank_ts_rank_volume_adv")
    return AlphaFactor(
        name="alpha017",
        stages=(
            (
                ts_mean(schema.VOLUME, 20).alias(adv20),
                ts_rank(schema.CLOSE, 10).alias(close_ts_rank),
                delta(schema.CLOSE, 1).alias(close_delta),
            ),
            (
                (pl.col(schema.VOLUME) / pl.col(adv20)).alias(volume_over_adv20),
                delta(close_delta, 1).alias(close_delta_delta),
            ),
            (ts_rank(volume_over_adv20, 5).alias(volume_adv_ts_rank),),
            (
                rank(close_ts_rank).alias(ranked_close_ts_rank),
                rank(close_delta_delta).alias(ranked_close_delta_delta),
                rank(volume_adv_ts_rank).alias(ranked_volume_adv_ts_rank),
            ),
        ),
        expr=(
            -pl.col(ranked_close_ts_rank)
            * pl.col(ranked_close_delta_delta)
            * pl.col(ranked_volume_adv_ts_rank)
        ),
        temporary_columns=(
            adv20,
            volume_over_adv20,
            close_ts_rank,
            close_delta,
            close_delta_delta,
            volume_adv_ts_rank,
            ranked_close_ts_rank,
            ranked_close_delta_delta,
            ranked_volume_adv_ts_rank,
        ),
    )


def alpha018() -> AlphaFactor:
    value = _tmp("alpha018", "value")
    spread = (pl.col(schema.CLOSE) - pl.col(schema.OPEN)).abs()
    expr = (
        ts_std(spread, 5)
        + (pl.col(schema.CLOSE) - pl.col(schema.OPEN))
        + ts_corr(schema.CLOSE, schema.OPEN, 10)
    )
    return AlphaFactor(
        name="alpha018",
        stages=((expr.alias(value),),),
        expr=-rank(value),
        temporary_columns=(value,),
    )


def alpha019() -> AlphaFactor:
    close_delay = _tmp("alpha019", "delay_close")
    close_delta = _tmp("alpha019", "delta_close")
    returns_sum = _tmp("alpha019", "sum_returns")
    ranked_returns_sum = _tmp("alpha019", "rank_sum_returns")
    return AlphaFactor(
        name="alpha019",
        stages=(
            (
                delay(schema.CLOSE, 7).alias(close_delay),
                delta(schema.CLOSE, 7).alias(close_delta),
                ts_sum(schema.RETURNS, 250).alias(returns_sum),
            ),
            (rank(1 + pl.col(returns_sum)).alias(ranked_returns_sum),),
        ),
        expr=(
            -((pl.col(schema.CLOSE) - pl.col(close_delay)) + pl.col(close_delta)).sign()
            * (1 + pl.col(ranked_returns_sum))
        ),
        temporary_columns=(close_delay, close_delta, returns_sum, ranked_returns_sum),
    )


def alpha020() -> AlphaFactor:
    diff_high = _tmp("alpha020", "open_delay_high")
    diff_close = _tmp("alpha020", "open_delay_close")
    diff_low = _tmp("alpha020", "open_delay_low")
    rank_high = _tmp("alpha020", "rank_open_delay_high")
    rank_close = _tmp("alpha020", "rank_open_delay_close")
    rank_low = _tmp("alpha020", "rank_open_delay_low")
    return AlphaFactor(
        name="alpha020",
        stages=(
            (
                (pl.col(schema.OPEN) - delay(schema.HIGH, 1)).alias(diff_high),
                (pl.col(schema.OPEN) - delay(schema.CLOSE, 1)).alias(diff_close),
                (pl.col(schema.OPEN) - delay(schema.LOW, 1)).alias(diff_low),
            ),
            (
                rank(diff_high).alias(rank_high),
                rank(diff_close).alias(rank_close),
                rank(diff_low).alias(rank_low),
            ),
        ),
        expr=-pl.col(rank_high) * pl.col(rank_close) * pl.col(rank_low),
        temporary_columns=(diff_high, diff_close, diff_low, rank_high, rank_close, rank_low),
    )


ALPHA_FACTORS = {
    "alpha001": alpha001,
    "alpha002": alpha002,
    "alpha003": alpha003,
    "alpha004": alpha004,
    "alpha005": alpha005,
    "alpha006": alpha006,
    "alpha007": alpha007,
    "alpha008": alpha008,
    "alpha009": alpha009,
    "alpha010": alpha010,
    "alpha011": alpha011,
    "alpha012": alpha012,
    "alpha013": alpha013,
    "alpha014": alpha014,
    "alpha015": alpha015,
    "alpha016": alpha016,
    "alpha017": alpha017,
    "alpha018": alpha018,
    "alpha019": alpha019,
    "alpha020": alpha020,
}
