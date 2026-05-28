"""Representative WorldQuant Alpha101 factors expressed as staged Polars plans."""

from __future__ import annotations

import polars as pl

from alpha101 import schema
from alpha101.factor import AlphaFactor
from alpha101.ops import (
    decay_linear,
    delta,
    delay,
    rank,
    scale,
    signed_power,
    ts_arg_max,
    ts_corr,
    ts_cov,
    ts_max,
    ts_mean,
    ts_min,
    ts_product,
    ts_rank,
    ts_std,
    ts_sum,
)


def _tmp(alpha: str, name: str) -> str:
    return f"__{alpha}_{name}"


def alpha001() -> AlphaFactor:
    # Alpha#001: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
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
    # Alpha#002: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    delta_log_volume = _tmp("alpha002", "delta_log_volume")
    intraday_return = _tmp("alpha002", "intraday_return")
    left = _tmp("alpha002", "rank_delta_log_volume")
    right = _tmp("alpha002", "rank_intraday_return")
    return AlphaFactor(
        name="alpha002",
        stages=(
            (
                delta(pl.col(schema.VOLUME).log(), 2).alias(delta_log_volume),
                (
                    (pl.col(schema.CLOSE) - pl.col(schema.OPEN)) / pl.col(schema.OPEN)
                ).alias(intraday_return),
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
    # Alpha#003: (-1 * correlation(rank(open), rank(volume), 10))
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
    # Alpha#004: (-1 * Ts_Rank(rank(low), 9))
    ranked_low = _tmp("alpha004", "rank_low")
    return AlphaFactor(
        name="alpha004",
        stages=((rank(schema.LOW).alias(ranked_low),),),
        expr=-ts_rank(ranked_low, 9),
        temporary_columns=(ranked_low,),
    )


def alpha005() -> AlphaFactor:
    # Alpha#005: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    open_minus_mean_vwap = _tmp("alpha005", "open_minus_mean_vwap")
    close_minus_vwap = _tmp("alpha005", "close_minus_vwap")
    ranked_open_minus_mean_vwap = _tmp("alpha005", "rank_open_minus_mean_vwap")
    ranked_close_minus_vwap = _tmp("alpha005", "rank_close_minus_vwap")
    return AlphaFactor(
        name="alpha005",
        stages=(
            (
                (pl.col(schema.OPEN) - ts_mean(schema.VWAP, 10)).alias(
                    open_minus_mean_vwap
                ),
                (pl.col(schema.CLOSE) - pl.col(schema.VWAP)).alias(close_minus_vwap),
            ),
            (
                rank(open_minus_mean_vwap).alias(ranked_open_minus_mean_vwap),
                rank(close_minus_vwap).alias(ranked_close_minus_vwap),
            ),
        ),
        expr=pl.col(ranked_open_minus_mean_vwap)
        * -pl.col(ranked_close_minus_vwap).abs(),
        temporary_columns=(
            open_minus_mean_vwap,
            close_minus_vwap,
            ranked_open_minus_mean_vwap,
            ranked_close_minus_vwap,
        ),
    )


def alpha006() -> AlphaFactor:
    # Alpha#006: (-1 * correlation(open, volume, 10))
    return AlphaFactor(
        name="alpha006",
        stages=(),
        expr=-ts_corr(schema.OPEN, schema.VOLUME, 10),
    )


def alpha007() -> AlphaFactor:
    # Alpha#007: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60))
    #             * sign(delta(close, 7))) : (-1))
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
    # Alpha#008: (-1 * rank(((sum(open, 5) * sum(returns, 5))
    #             - delay((sum(open, 5) * sum(returns, 5)), 10))))
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
    # Alpha#009: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1)
    #             : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
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
    # Alpha#010: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1)
    #             : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
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
    # Alpha#011: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3)))
    #             * rank(delta(volume, 3)))
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
        expr=(pl.col(rank_max_spread) + pl.col(rank_min_spread))
        * pl.col(rank_volume_delta),
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
    # Alpha#012: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    return AlphaFactor(
        name="alpha012",
        stages=(),
        expr=delta(schema.VOLUME, 1).sign() * -delta(schema.CLOSE, 1),
    )


def alpha013() -> AlphaFactor:
    # Alpha#013: (-1 * rank(covariance(rank(close), rank(volume), 5)))
    ranked_close = _tmp("alpha013", "rank_close")
    ranked_volume = _tmp("alpha013", "rank_volume")
    cov = _tmp("alpha013", "cov")
    return AlphaFactor(
        name="alpha013",
        stages=(
            (
                rank(schema.CLOSE).alias(ranked_close),
                rank(schema.VOLUME).alias(ranked_volume),
            ),
            (ts_cov(ranked_close, ranked_volume, 5).alias(cov),),
        ),
        expr=-rank(cov),
        temporary_columns=(ranked_close, ranked_volume, cov),
    )


def alpha014() -> AlphaFactor:
    # Alpha#014: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
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
    # Alpha#015: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    ranked_high = _tmp("alpha015", "rank_high")
    ranked_volume = _tmp("alpha015", "rank_volume")
    corr = _tmp("alpha015", "corr")
    ranked_corr = _tmp("alpha015", "rank_corr")
    corr_sum = _tmp("alpha015", "sum_rank_corr")
    return AlphaFactor(
        name="alpha015",
        stages=(
            (
                rank(schema.HIGH).alias(ranked_high),
                rank(schema.VOLUME).alias(ranked_volume),
            ),
            (ts_corr(ranked_high, ranked_volume, 3).alias(corr),),
            (rank(corr).alias(ranked_corr),),
            (ts_sum(ranked_corr, 3).alias(corr_sum),),
        ),
        expr=-pl.col(corr_sum),
        temporary_columns=(ranked_high, ranked_volume, corr, ranked_corr, corr_sum),
    )


def alpha016() -> AlphaFactor:
    # Alpha#016: (-1 * rank(covariance(rank(high), rank(volume), 5)))
    ranked_high = _tmp("alpha016", "rank_high")
    ranked_volume = _tmp("alpha016", "rank_volume")
    cov = _tmp("alpha016", "cov")
    return AlphaFactor(
        name="alpha016",
        stages=(
            (
                rank(schema.HIGH).alias(ranked_high),
                rank(schema.VOLUME).alias(ranked_volume),
            ),
            (ts_cov(ranked_high, ranked_volume, 5).alias(cov),),
        ),
        expr=-rank(cov),
        temporary_columns=(ranked_high, ranked_volume, cov),
    )


def alpha017() -> AlphaFactor:
    # Alpha#017: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)))
    #             * rank(ts_rank((volume / adv20), 5)))
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
    # Alpha#018: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open))
    #             + correlation(close, open, 10))))
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
    # Alpha#019: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7))))
    #             * (1 + rank((1 + sum(returns, 250)))))
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
    # Alpha#020: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1))))
    #             * rank((open - delay(low, 1))))
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
        temporary_columns=(
            diff_high,
            diff_close,
            diff_low,
            rank_high,
            rank_close,
            rank_low,
        ),
    )


def alpha021() -> AlphaFactor:
    # Alpha#021: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1)
    #             : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1
    #             : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1))))
    mean_close_8 = _tmp("alpha021", "mean_close_8")
    std_close_8 = _tmp("alpha021", "std_close_8")
    mean_close_2 = _tmp("alpha021", "mean_close_2")
    adv20 = _tmp("alpha021", "adv20")
    volume_over_adv20 = _tmp("alpha021", "volume_over_adv20")
    return AlphaFactor(
        name="alpha021",
        stages=(
            (
                ts_mean(schema.CLOSE, 8).alias(mean_close_8),
                ts_std(schema.CLOSE, 8).alias(std_close_8),
                ts_mean(schema.CLOSE, 2).alias(mean_close_2),
                ts_mean(schema.VOLUME, 20).alias(adv20),
            ),
            ((pl.col(schema.VOLUME) / pl.col(adv20)).alias(volume_over_adv20),),
        ),
        expr=pl.when(
            (pl.col(mean_close_8) + pl.col(std_close_8)) < pl.col(mean_close_2)
        )
        .then(-1)
        .when(pl.col(mean_close_2) < (pl.col(mean_close_8) - pl.col(std_close_8)))
        .then(1)
        .when(pl.col(volume_over_adv20) >= 1)
        .then(1)
        .otherwise(-1),
        temporary_columns=(
            mean_close_8,
            std_close_8,
            mean_close_2,
            adv20,
            volume_over_adv20,
        ),
    )


def alpha022() -> AlphaFactor:
    # Alpha#022: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    corr = _tmp("alpha022", "corr")
    corr_delta = _tmp("alpha022", "delta_corr")
    close_std = _tmp("alpha022", "std_close")
    ranked_close_std = _tmp("alpha022", "rank_std_close")
    return AlphaFactor(
        name="alpha022",
        stages=(
            (
                ts_corr(schema.HIGH, schema.VOLUME, 5).alias(corr),
                ts_std(schema.CLOSE, 20).alias(close_std),
            ),
            (
                delta(corr, 5).alias(corr_delta),
                rank(close_std).alias(ranked_close_std),
            ),
        ),
        expr=-(pl.col(corr_delta) * pl.col(ranked_close_std)),
        temporary_columns=(corr, corr_delta, close_std, ranked_close_std),
    )


def alpha023() -> AlphaFactor:
    # Alpha#023: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    mean_high_20 = _tmp("alpha023", "mean_high_20")
    high_delta = _tmp("alpha023", "delta_high")
    return AlphaFactor(
        name="alpha023",
        stages=(
            (
                ts_mean(schema.HIGH, 20).alias(mean_high_20),
                delta(schema.HIGH, 2).alias(high_delta),
            ),
        ),
        expr=pl.when(pl.col(mean_high_20) < pl.col(schema.HIGH))
        .then(-pl.col(high_delta))
        .otherwise(0),
        temporary_columns=(mean_high_20, high_delta),
    )


def alpha024() -> AlphaFactor:
    # Alpha#024: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05)
    #             || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05))
    #             ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
    mean_close_100 = _tmp("alpha024", "mean_close_100")
    mean_close_delta = _tmp("alpha024", "delta_mean_close_100")
    delayed_close = _tmp("alpha024", "delay_close_100")
    ratio = _tmp("alpha024", "ratio")
    close_min = _tmp("alpha024", "ts_min_close")
    close_delta = _tmp("alpha024", "delta_close")
    return AlphaFactor(
        name="alpha024",
        stages=(
            (
                ts_mean(schema.CLOSE, 100).alias(mean_close_100),
                delay(schema.CLOSE, 100).alias(delayed_close),
                ts_min(schema.CLOSE, 100).alias(close_min),
                delta(schema.CLOSE, 3).alias(close_delta),
            ),
            (delta(mean_close_100, 100).alias(mean_close_delta),),
            ((pl.col(mean_close_delta) / pl.col(delayed_close)).alias(ratio),),
        ),
        expr=pl.when(pl.col(ratio) <= 0.05)
        .then(-(pl.col(schema.CLOSE) - pl.col(close_min)))
        .otherwise(-pl.col(close_delta)),
        temporary_columns=(
            mean_close_100,
            mean_close_delta,
            delayed_close,
            ratio,
            close_min,
            close_delta,
        ),
    )


def alpha025() -> AlphaFactor:
    # Alpha#025: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    adv20 = _tmp("alpha025", "adv20")
    value = _tmp("alpha025", "value")
    return AlphaFactor(
        name="alpha025",
        stages=(
            (ts_mean(schema.VOLUME, 20).alias(adv20),),
            (
                (
                    -pl.col(schema.RETURNS)
                    * pl.col(adv20)
                    * pl.col(schema.VWAP)
                    * (pl.col(schema.HIGH) - pl.col(schema.CLOSE))
                ).alias(value),
            ),
        ),
        expr=rank(value),
        temporary_columns=(adv20, value),
    )


def alpha026() -> AlphaFactor:
    # Alpha#026: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    ts_rank_volume = _tmp("alpha026", "ts_rank_volume")
    ts_rank_high = _tmp("alpha026", "ts_rank_high")
    corr = _tmp("alpha026", "corr")
    corr_max = _tmp("alpha026", "ts_max_corr")
    return AlphaFactor(
        name="alpha026",
        stages=(
            (
                ts_rank(schema.VOLUME, 5).alias(ts_rank_volume),
                ts_rank(schema.HIGH, 5).alias(ts_rank_high),
            ),
            (ts_corr(ts_rank_volume, ts_rank_high, 5).alias(corr),),
            (ts_max(corr, 3).alias(corr_max),),
        ),
        expr=-pl.col(corr_max),
        temporary_columns=(ts_rank_volume, ts_rank_high, corr, corr_max),
    )


def alpha027() -> AlphaFactor:
    # Alpha#027: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0)))
    #             ? (-1) : 1)
    ranked_volume = _tmp("alpha027", "rank_volume")
    ranked_vwap = _tmp("alpha027", "rank_vwap")
    corr = _tmp("alpha027", "corr")
    mean_corr = _tmp("alpha027", "mean_corr")
    ranked_mean_corr = _tmp("alpha027", "rank_mean_corr")
    return AlphaFactor(
        name="alpha027",
        stages=(
            (
                rank(schema.VOLUME).alias(ranked_volume),
                rank(schema.VWAP).alias(ranked_vwap),
            ),
            (ts_corr(ranked_volume, ranked_vwap, 6).alias(corr),),
            ((ts_sum(corr, 2) / 2).alias(mean_corr),),
            (rank(mean_corr).alias(ranked_mean_corr),),
        ),
        expr=pl.when(pl.col(ranked_mean_corr) > 0.5).then(-1).otherwise(1),
        temporary_columns=(
            ranked_volume,
            ranked_vwap,
            corr,
            mean_corr,
            ranked_mean_corr,
        ),
    )


def alpha028() -> AlphaFactor:
    # Alpha#028: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    adv20 = _tmp("alpha028", "adv20")
    corr = _tmp("alpha028", "corr")
    value = _tmp("alpha028", "value")
    return AlphaFactor(
        name="alpha028",
        stages=(
            (ts_mean(schema.VOLUME, 20).alias(adv20),),
            (ts_corr(adv20, schema.LOW, 5).alias(corr),),
            (
                (
                    pl.col(corr)
                    + ((pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2)
                    - pl.col(schema.CLOSE)
                ).alias(value),
            ),
        ),
        expr=scale(value),
        temporary_columns=(adv20, corr, value),
    )


def alpha029() -> AlphaFactor:
    # Alpha#029: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank(
    #             (-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5)
    #             + ts_rank(delay((-1 * returns), 6), 5))
    close_minus_one_delta = _tmp("alpha029", "delta_close_minus_one")
    ranked_delta = _tmp("alpha029", "rank_delta")
    neg_ranked_delta = _tmp("alpha029", "neg_rank_delta")
    rank_1 = _tmp("alpha029", "rank_1")
    rank_2 = _tmp("alpha029", "rank_2")
    ts_min_rank = _tmp("alpha029", "ts_min_rank")
    log_value = _tmp("alpha029", "log_value")
    scaled_log = _tmp("alpha029", "scaled_log")
    rank_3 = _tmp("alpha029", "rank_3")
    rank_4 = _tmp("alpha029", "rank_4")
    product_value = _tmp("alpha029", "product")
    left = _tmp("alpha029", "left")
    delayed_returns = _tmp("alpha029", "delay_neg_returns")
    right = _tmp("alpha029", "right")
    return AlphaFactor(
        name="alpha029",
        stages=(
            (
                delta(pl.col(schema.CLOSE) - 1, 5).alias(close_minus_one_delta),
                delay(-pl.col(schema.RETURNS), 6).alias(delayed_returns),
            ),
            (rank(close_minus_one_delta).alias(ranked_delta),),
            ((-pl.col(ranked_delta)).alias(neg_ranked_delta),),
            (rank(neg_ranked_delta).alias(rank_1),),
            (rank(rank_1).alias(rank_2),),
            (ts_min(rank_2, 2).alias(ts_min_rank),),
            (pl.col(ts_min_rank).log().alias(log_value),),
            (scale(log_value).alias(scaled_log),),
            (rank(scaled_log).alias(rank_3),),
            (rank(rank_3).alias(rank_4),),
            (
                ts_product(rank_4, 1).alias(product_value),
                ts_rank(delayed_returns, 5).alias(right),
            ),
            (ts_min(product_value, 5).alias(left),),
        ),
        expr=pl.col(left) + pl.col(right),
        temporary_columns=(
            close_minus_one_delta,
            ranked_delta,
            neg_ranked_delta,
            rank_1,
            rank_2,
            ts_min_rank,
            log_value,
            scaled_log,
            rank_3,
            rank_4,
            product_value,
            left,
            delayed_returns,
            right,
        ),
    )


def alpha030() -> AlphaFactor:
    # Alpha#030: (((1.0 - rank(((sign((close - delay(close, 1)))
    #             + sign((delay(close, 1) - delay(close, 2))))
    #             + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    delayed_close_1 = _tmp("alpha030", "delay_close_1")
    delayed_close_2 = _tmp("alpha030", "delay_close_2")
    delayed_close_3 = _tmp("alpha030", "delay_close_3")
    signal = _tmp("alpha030", "signal")
    ranked_signal = _tmp("alpha030", "rank_signal")
    volume_sum_5 = _tmp("alpha030", "sum_volume_5")
    volume_sum_20 = _tmp("alpha030", "sum_volume_20")
    return AlphaFactor(
        name="alpha030",
        stages=(
            (
                delay(schema.CLOSE, 1).alias(delayed_close_1),
                delay(schema.CLOSE, 2).alias(delayed_close_2),
                delay(schema.CLOSE, 3).alias(delayed_close_3),
                ts_sum(schema.VOLUME, 5).alias(volume_sum_5),
                ts_sum(schema.VOLUME, 20).alias(volume_sum_20),
            ),
            (
                (
                    (pl.col(schema.CLOSE) - pl.col(delayed_close_1)).sign()
                    + (pl.col(delayed_close_1) - pl.col(delayed_close_2)).sign()
                    + (pl.col(delayed_close_2) - pl.col(delayed_close_3)).sign()
                ).alias(signal),
            ),
            (rank(signal).alias(ranked_signal),),
        ),
        expr=((1.0 - pl.col(ranked_signal)) * pl.col(volume_sum_5))
        / pl.col(volume_sum_20),
        temporary_columns=(
            delayed_close_1,
            delayed_close_2,
            delayed_close_3,
            signal,
            ranked_signal,
            volume_sum_5,
            volume_sum_20,
        ),
    )


def alpha031() -> AlphaFactor:
    # Alpha#031: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10))))
    #             + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    delta_close_10 = _tmp("alpha031", "delta_close_10")
    rank_delta_1 = _tmp("alpha031", "rank_delta_1")
    rank_delta_2 = _tmp("alpha031", "rank_delta_2")
    neg_rank_delta = _tmp("alpha031", "neg_rank_delta")
    decayed = _tmp("alpha031", "decayed")
    rank_decayed_1 = _tmp("alpha031", "rank_decayed_1")
    rank_decayed_2 = _tmp("alpha031", "rank_decayed_2")
    rank_decayed_3 = _tmp("alpha031", "rank_decayed_3")
    delta_close_3 = _tmp("alpha031", "delta_close_3")
    rank_neg_delta_3 = _tmp("alpha031", "rank_neg_delta_3")
    adv20 = _tmp("alpha031", "adv20")
    corr = _tmp("alpha031", "corr")
    scaled_corr = _tmp("alpha031", "scaled_corr")
    return AlphaFactor(
        name="alpha031",
        stages=(
            (
                delta(schema.CLOSE, 10).alias(delta_close_10),
                delta(schema.CLOSE, 3).alias(delta_close_3),
                ts_mean(schema.VOLUME, 20).alias(adv20),
            ),
            (
                rank(delta_close_10).alias(rank_delta_1),
                rank(-pl.col(delta_close_3)).alias(rank_neg_delta_3),
                ts_corr(adv20, schema.LOW, 12).alias(corr),
            ),
            (
                rank(rank_delta_1).alias(rank_delta_2),
                scale(corr).alias(scaled_corr),
            ),
            ((-pl.col(rank_delta_2)).alias(neg_rank_delta),),
            (decay_linear(neg_rank_delta, 10).alias(decayed),),
            (rank(decayed).alias(rank_decayed_1),),
            (rank(rank_decayed_1).alias(rank_decayed_2),),
            (rank(rank_decayed_2).alias(rank_decayed_3),),
        ),
        expr=pl.col(rank_decayed_3)
        + pl.col(rank_neg_delta_3)
        + pl.col(scaled_corr).sign(),
        temporary_columns=(
            delta_close_10,
            rank_delta_1,
            rank_delta_2,
            neg_rank_delta,
            decayed,
            rank_decayed_1,
            rank_decayed_2,
            rank_decayed_3,
            delta_close_3,
            rank_neg_delta_3,
            adv20,
            corr,
            scaled_corr,
        ),
    )


def alpha032() -> AlphaFactor:
    # Alpha#032: (scale(((sum(close, 7) / 7) - close))
    #             + (20 * scale(correlation(vwap, delay(close, 5), 230))))
    mean_close_7_minus_close = _tmp("alpha032", "mean_close_7_minus_close")
    delayed_close = _tmp("alpha032", "delay_close")
    corr = _tmp("alpha032", "corr")
    return AlphaFactor(
        name="alpha032",
        stages=(
            (
                (ts_sum(schema.CLOSE, 7) / 7 - pl.col(schema.CLOSE)).alias(
                    mean_close_7_minus_close
                ),
                delay(schema.CLOSE, 5).alias(delayed_close),
            ),
            (ts_corr(schema.VWAP, delayed_close, 230).alias(corr),),
        ),
        expr=scale(mean_close_7_minus_close) + (20 * scale(corr)),
        temporary_columns=(mean_close_7_minus_close, delayed_close, corr),
    )


def alpha033() -> AlphaFactor:
    # Alpha#033: rank((-1 * ((1 - (open / close))^1)))
    return AlphaFactor(
        name="alpha033",
        stages=(),
        expr=rank(-((1 - (pl.col(schema.OPEN) / pl.col(schema.CLOSE))).pow(1))),
    )


def alpha034() -> AlphaFactor:
    # Alpha#034: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5))))
    #             + (1 - rank(delta(close, 1)))))
    std_ratio = _tmp("alpha034", "std_ratio")
    close_delta = _tmp("alpha034", "delta_close")
    ranked_std_ratio = _tmp("alpha034", "rank_std_ratio")
    ranked_delta = _tmp("alpha034", "rank_delta")
    value = _tmp("alpha034", "value")
    return AlphaFactor(
        name="alpha034",
        stages=(
            (
                (ts_std(schema.RETURNS, 2) / ts_std(schema.RETURNS, 5)).alias(
                    std_ratio
                ),
                delta(schema.CLOSE, 1).alias(close_delta),
            ),
            (
                rank(std_ratio).alias(ranked_std_ratio),
                rank(close_delta).alias(ranked_delta),
            ),
            (
                ((1 - pl.col(ranked_std_ratio)) + (1 - pl.col(ranked_delta))).alias(
                    value
                ),
            ),
        ),
        expr=rank(value),
        temporary_columns=(
            std_ratio,
            close_delta,
            ranked_std_ratio,
            ranked_delta,
            value,
        ),
    )


def alpha035() -> AlphaFactor:
    # Alpha#035: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16)))
    #             * (1 - Ts_Rank(returns, 32)))
    price_range = _tmp("alpha035", "price_range")
    ts_rank_volume = _tmp("alpha035", "ts_rank_volume")
    ts_rank_price_range = _tmp("alpha035", "ts_rank_price_range")
    ts_rank_returns = _tmp("alpha035", "ts_rank_returns")
    return AlphaFactor(
        name="alpha035",
        stages=(
            (
                (
                    (pl.col(schema.CLOSE) + pl.col(schema.HIGH)) - pl.col(schema.LOW)
                ).alias(price_range),
            ),
            (
                ts_rank(schema.VOLUME, 32).alias(ts_rank_volume),
                ts_rank(price_range, 16).alias(ts_rank_price_range),
                ts_rank(schema.RETURNS, 32).alias(ts_rank_returns),
            ),
        ),
        expr=(
            pl.col(ts_rank_volume)
            * (1 - pl.col(ts_rank_price_range))
            * (1 - pl.col(ts_rank_returns))
        ),
        temporary_columns=(
            price_range,
            ts_rank_volume,
            ts_rank_price_range,
            ts_rank_returns,
        ),
    )


def alpha036() -> AlphaFactor:
    # Alpha#036: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15)))
    #             + (0.7 * rank((open - close))))
    #             + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5))))
    #             + rank(abs(correlation(vwap, adv20, 6))))
    #             + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    close_open_spread = _tmp("alpha036", "close_open_spread")
    delayed_volume = _tmp("alpha036", "delay_volume")
    corr_spread_volume = _tmp("alpha036", "corr_spread_volume")
    rank_corr_spread_volume = _tmp("alpha036", "rank_corr_spread_volume")
    rank_open_minus_close = _tmp("alpha036", "rank_open_minus_close")
    delayed_neg_returns = _tmp("alpha036", "delay_neg_returns")
    ts_rank_delayed_returns = _tmp("alpha036", "ts_rank_delay_returns")
    rank_ts_rank_delayed_returns = _tmp("alpha036", "rank_ts_rank_delay_returns")
    adv20 = _tmp("alpha036", "adv20")
    corr_vwap_adv20 = _tmp("alpha036", "corr_vwap_adv20")
    rank_abs_corr_vwap_adv20 = _tmp("alpha036", "rank_abs_corr_vwap_adv20")
    mean_close_200 = _tmp("alpha036", "mean_close_200")
    long_term_value = _tmp("alpha036", "long_term_value")
    rank_long_term_value = _tmp("alpha036", "rank_long_term_value")
    return AlphaFactor(
        name="alpha036",
        stages=(
            (
                (pl.col(schema.CLOSE) - pl.col(schema.OPEN)).alias(close_open_spread),
                delay(schema.VOLUME, 1).alias(delayed_volume),
                delay(-pl.col(schema.RETURNS), 6).alias(delayed_neg_returns),
                ts_mean(schema.VOLUME, 20).alias(adv20),
                ts_mean(schema.CLOSE, 200).alias(mean_close_200),
            ),
            (
                ts_corr(close_open_spread, delayed_volume, 15).alias(
                    corr_spread_volume
                ),
                ts_rank(delayed_neg_returns, 5).alias(ts_rank_delayed_returns),
                ts_corr(schema.VWAP, adv20, 6).alias(corr_vwap_adv20),
                (
                    (pl.col(mean_close_200) - pl.col(schema.OPEN))
                    * pl.col(close_open_spread)
                ).alias(long_term_value),
            ),
            (
                rank(corr_spread_volume).alias(rank_corr_spread_volume),
                rank(pl.col(schema.OPEN) - pl.col(schema.CLOSE)).alias(
                    rank_open_minus_close
                ),
                rank(ts_rank_delayed_returns).alias(rank_ts_rank_delayed_returns),
                rank(pl.col(corr_vwap_adv20).abs()).alias(rank_abs_corr_vwap_adv20),
                rank(long_term_value).alias(rank_long_term_value),
            ),
        ),
        expr=(
            2.21 * pl.col(rank_corr_spread_volume)
            + 0.7 * pl.col(rank_open_minus_close)
            + 0.73 * pl.col(rank_ts_rank_delayed_returns)
            + pl.col(rank_abs_corr_vwap_adv20)
            + 0.6 * pl.col(rank_long_term_value)
        ),
        temporary_columns=(
            close_open_spread,
            delayed_volume,
            corr_spread_volume,
            rank_corr_spread_volume,
            rank_open_minus_close,
            delayed_neg_returns,
            ts_rank_delayed_returns,
            rank_ts_rank_delayed_returns,
            adv20,
            corr_vwap_adv20,
            rank_abs_corr_vwap_adv20,
            mean_close_200,
            long_term_value,
            rank_long_term_value,
        ),
    )


def alpha037() -> AlphaFactor:
    # Alpha#037: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    open_minus_close = _tmp("alpha037", "open_minus_close")
    delayed_open_minus_close = _tmp("alpha037", "delay_open_minus_close")
    corr = _tmp("alpha037", "corr")
    ranked_corr = _tmp("alpha037", "rank_corr")
    ranked_open_minus_close = _tmp("alpha037", "rank_open_minus_close")
    return AlphaFactor(
        name="alpha037",
        stages=(
            ((pl.col(schema.OPEN) - pl.col(schema.CLOSE)).alias(open_minus_close),),
            (delay(open_minus_close, 1).alias(delayed_open_minus_close),),
            (ts_corr(delayed_open_minus_close, schema.CLOSE, 200).alias(corr),),
            (
                rank(corr).alias(ranked_corr),
                rank(open_minus_close).alias(ranked_open_minus_close),
            ),
        ),
        expr=pl.col(ranked_corr) + pl.col(ranked_open_minus_close),
        temporary_columns=(
            open_minus_close,
            delayed_open_minus_close,
            corr,
            ranked_corr,
            ranked_open_minus_close,
        ),
    )


def alpha038() -> AlphaFactor:
    # Alpha#038: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    close_ts_rank = _tmp("alpha038", "ts_rank_close")
    rank_close_ts_rank = _tmp("alpha038", "rank_ts_rank_close")
    rank_close_open_ratio = _tmp("alpha038", "rank_close_open_ratio")
    return AlphaFactor(
        name="alpha038",
        stages=(
            (ts_rank(schema.CLOSE, 10).alias(close_ts_rank),),
            (
                rank(close_ts_rank).alias(rank_close_ts_rank),
                rank(pl.col(schema.CLOSE) / pl.col(schema.OPEN)).alias(
                    rank_close_open_ratio
                ),
            ),
        ),
        expr=-pl.col(rank_close_ts_rank) * pl.col(rank_close_open_ratio),
        temporary_columns=(close_ts_rank, rank_close_ts_rank, rank_close_open_ratio),
    )


def alpha039() -> AlphaFactor:
    # Alpha#039: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9))))))
    #             * (1 + rank(sum(returns, 250))))
    adv20 = _tmp("alpha039", "adv20")
    volume_over_adv20 = _tmp("alpha039", "volume_over_adv20")
    decayed_volume = _tmp("alpha039", "decayed_volume")
    ranked_decayed_volume = _tmp("alpha039", "rank_decayed_volume")
    close_delta = _tmp("alpha039", "delta_close")
    left_value = _tmp("alpha039", "left_value")
    ranked_left_value = _tmp("alpha039", "rank_left_value")
    returns_sum = _tmp("alpha039", "sum_returns")
    ranked_returns_sum = _tmp("alpha039", "rank_sum_returns")
    return AlphaFactor(
        name="alpha039",
        stages=(
            (
                ts_mean(schema.VOLUME, 20).alias(adv20),
                delta(schema.CLOSE, 7).alias(close_delta),
                ts_sum(schema.RETURNS, 250).alias(returns_sum),
            ),
            ((pl.col(schema.VOLUME) / pl.col(adv20)).alias(volume_over_adv20),),
            (decay_linear(volume_over_adv20, 9).alias(decayed_volume),),
            (
                rank(decayed_volume).alias(ranked_decayed_volume),
                rank(returns_sum).alias(ranked_returns_sum),
            ),
            (
                (pl.col(close_delta) * (1 - pl.col(ranked_decayed_volume))).alias(
                    left_value
                ),
            ),
            (rank(left_value).alias(ranked_left_value),),
        ),
        expr=-pl.col(ranked_left_value) * (1 + pl.col(ranked_returns_sum)),
        temporary_columns=(
            adv20,
            volume_over_adv20,
            decayed_volume,
            ranked_decayed_volume,
            close_delta,
            left_value,
            ranked_left_value,
            returns_sum,
            ranked_returns_sum,
        ),
    )


def alpha040() -> AlphaFactor:
    # Alpha#040: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    high_std = _tmp("alpha040", "std_high")
    ranked_high_std = _tmp("alpha040", "rank_std_high")
    corr = _tmp("alpha040", "corr")
    return AlphaFactor(
        name="alpha040",
        stages=(
            (
                ts_std(schema.HIGH, 10).alias(high_std),
                ts_corr(schema.HIGH, schema.VOLUME, 10).alias(corr),
            ),
            (rank(high_std).alias(ranked_high_std),),
        ),
        expr=-pl.col(ranked_high_std) * pl.col(corr),
        temporary_columns=(high_std, ranked_high_std, corr),
    )


def alpha041() -> AlphaFactor:
    # Alpha#041: (((high * low)^0.5) - vwap)
    return AlphaFactor(
        name="alpha041",
        stages=(),
        expr=(pl.col(schema.HIGH) * pl.col(schema.LOW)).sqrt() - pl.col(schema.VWAP),
    )


def alpha042() -> AlphaFactor:
    # Alpha#042: (rank((vwap - close)) / rank((vwap + close)))
    numerator = _tmp("alpha042", "rank_vwap_close_diff")
    denominator = _tmp("alpha042", "rank_vwap_close_sum")
    return AlphaFactor(
        name="alpha042",
        stages=(
            (
                rank(pl.col(schema.VWAP) - pl.col(schema.CLOSE)).alias(numerator),
                rank(pl.col(schema.VWAP) + pl.col(schema.CLOSE)).alias(denominator),
            ),
        ),
        expr=pl.col(numerator) / pl.col(denominator),
        temporary_columns=(numerator, denominator),
    )


def alpha043() -> AlphaFactor:
    # Alpha#043: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    adv20 = _tmp("alpha043", "adv20")
    volume_over_adv20 = _tmp("alpha043", "volume_over_adv20")
    close_delta = _tmp("alpha043", "delta_close")
    left = _tmp("alpha043", "ts_rank_volume_adv")
    right = _tmp("alpha043", "ts_rank_neg_delta")
    return AlphaFactor(
        name="alpha043",
        stages=(
            (
                ts_mean(schema.VOLUME, 20).alias(adv20),
                delta(schema.CLOSE, 7).alias(close_delta),
            ),
            ((pl.col(schema.VOLUME) / pl.col(adv20)).alias(volume_over_adv20),),
            (
                ts_rank(volume_over_adv20, 20).alias(left),
                ts_rank(-pl.col(close_delta), 8).alias(right),
            ),
        ),
        expr=pl.col(left) * pl.col(right),
        temporary_columns=(adv20, volume_over_adv20, close_delta, left, right),
    )


def alpha044() -> AlphaFactor:
    # Alpha#044: (-1 * correlation(high, rank(volume), 5))
    ranked_volume = _tmp("alpha044", "rank_volume")
    return AlphaFactor(
        name="alpha044",
        stages=((rank(schema.VOLUME).alias(ranked_volume),),),
        expr=-ts_corr(schema.HIGH, ranked_volume, 5),
        temporary_columns=(ranked_volume,),
    )


def alpha045() -> AlphaFactor:
    # Alpha#045: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2))
    #             * rank(correlation(sum(close, 5), sum(close, 20), 2))))
    delayed_close = _tmp("alpha045", "delay_close")
    delayed_close_mean = _tmp("alpha045", "mean_delay_close")
    ranked_delayed_close_mean = _tmp("alpha045", "rank_mean_delay_close")
    close_sum_5 = _tmp("alpha045", "sum_close_5")
    close_sum_20 = _tmp("alpha045", "sum_close_20")
    corr_close_volume = _tmp("alpha045", "corr_close_volume")
    corr_close_sums = _tmp("alpha045", "corr_close_sums")
    ranked_corr_close_sums = _tmp("alpha045", "rank_corr_close_sums")
    return AlphaFactor(
        name="alpha045",
        stages=(
            (
                delay(schema.CLOSE, 5).alias(delayed_close),
                ts_sum(schema.CLOSE, 5).alias(close_sum_5),
                ts_sum(schema.CLOSE, 20).alias(close_sum_20),
                ts_corr(schema.CLOSE, schema.VOLUME, 2).alias(corr_close_volume),
            ),
            (
                (ts_sum(delayed_close, 20) / 20).alias(delayed_close_mean),
                ts_corr(close_sum_5, close_sum_20, 2).alias(corr_close_sums),
            ),
            (
                rank(delayed_close_mean).alias(ranked_delayed_close_mean),
                rank(corr_close_sums).alias(ranked_corr_close_sums),
            ),
        ),
        expr=-(
            pl.col(ranked_delayed_close_mean)
            * pl.col(corr_close_volume)
            * pl.col(ranked_corr_close_sums)
        ),
        temporary_columns=(
            delayed_close,
            delayed_close_mean,
            ranked_delayed_close_mean,
            close_sum_5,
            close_sum_20,
            corr_close_volume,
            corr_close_sums,
            ranked_corr_close_sums,
        ),
    )


def alpha046() -> AlphaFactor:
    # Alpha#046: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10)
    #             - ((delay(close, 10) - close) / 10))) ? (-1)
    #             : (((((delay(close, 20) - delay(close, 10)) / 10)
    #             - ((delay(close, 10) - close) / 10)) < 0) ? 1
    #             : ((-1) * (close - delay(close, 1)))))
    delay_1 = _tmp("alpha046", "delay_close_1")
    delay_10 = _tmp("alpha046", "delay_close_10")
    delay_20 = _tmp("alpha046", "delay_close_20")
    trend = _tmp("alpha046", "trend")
    return AlphaFactor(
        name="alpha046",
        stages=(
            (
                delay(schema.CLOSE, 1).alias(delay_1),
                delay(schema.CLOSE, 10).alias(delay_10),
                delay(schema.CLOSE, 20).alias(delay_20),
            ),
            (
                (
                    ((pl.col(delay_20) - pl.col(delay_10)) / 10)
                    - ((pl.col(delay_10) - pl.col(schema.CLOSE)) / 10)
                ).alias(trend),
            ),
        ),
        expr=pl.when(pl.col(trend) > 0.25)
        .then(-1)
        .when(pl.col(trend) < 0)
        .then(1)
        .otherwise(-(pl.col(schema.CLOSE) - pl.col(delay_1))),
        temporary_columns=(delay_1, delay_10, delay_20, trend),
    )


def alpha047() -> AlphaFactor:
    # Alpha#047: ((((rank((1 / close)) * volume) / adv20)
    #             * ((high * rank((high - close))) / (sum(high, 5) / 5)))
    #             - rank((vwap - delay(vwap, 5))))
    adv20 = _tmp("alpha047", "adv20")
    delayed_vwap = _tmp("alpha047", "delay_vwap")
    high_mean_5 = _tmp("alpha047", "mean_high_5")
    ranked_inverse_close = _tmp("alpha047", "rank_inverse_close")
    ranked_high_close = _tmp("alpha047", "rank_high_close")
    ranked_vwap_delta = _tmp("alpha047", "rank_vwap_delta")
    return AlphaFactor(
        name="alpha047",
        stages=(
            (
                ts_mean(schema.VOLUME, 20).alias(adv20),
                delay(schema.VWAP, 5).alias(delayed_vwap),
                (ts_sum(schema.HIGH, 5) / 5).alias(high_mean_5),
            ),
            (
                rank(1 / pl.col(schema.CLOSE)).alias(ranked_inverse_close),
                rank(pl.col(schema.HIGH) - pl.col(schema.CLOSE)).alias(
                    ranked_high_close
                ),
                rank(pl.col(schema.VWAP) - pl.col(delayed_vwap)).alias(
                    ranked_vwap_delta
                ),
            ),
        ),
        expr=(
            (
                (pl.col(ranked_inverse_close) * pl.col(schema.VOLUME) / pl.col(adv20))
                * (
                    (pl.col(schema.HIGH) * pl.col(ranked_high_close))
                    / pl.col(high_mean_5)
                )
            )
            - pl.col(ranked_vwap_delta)
        ),
        temporary_columns=(
            adv20,
            delayed_vwap,
            high_mean_5,
            ranked_inverse_close,
            ranked_high_close,
            ranked_vwap_delta,
        ),
    )


def alpha048() -> AlphaFactor:
    # Alpha#048: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250)
    #             * delta(close, 1)) / close), IndClass.subindustry)
    #             / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    # Requires IndNeutralize (subindustry classification); not implemented.
    raise NotImplementedError(
        "alpha048 requires IndNeutralize (subindustry classification)"
    )


def alpha049() -> AlphaFactor:
    # Alpha#049: (((((delay(close, 20) - delay(close, 10)) / 10)
    #             - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1
    #             : ((-1) * (close - delay(close, 1))))
    delay_1 = _tmp("alpha049", "delay_close_1")
    delay_10 = _tmp("alpha049", "delay_close_10")
    delay_20 = _tmp("alpha049", "delay_close_20")
    trend = _tmp("alpha049", "trend")
    return AlphaFactor(
        name="alpha049",
        stages=(
            (
                delay(schema.CLOSE, 1).alias(delay_1),
                delay(schema.CLOSE, 10).alias(delay_10),
                delay(schema.CLOSE, 20).alias(delay_20),
            ),
            (
                (
                    ((pl.col(delay_20) - pl.col(delay_10)) / 10)
                    - ((pl.col(delay_10) - pl.col(schema.CLOSE)) / 10)
                ).alias(trend),
            ),
        ),
        expr=pl.when(pl.col(trend) < -0.1)
        .then(1)
        .otherwise(-(pl.col(schema.CLOSE) - pl.col(delay_1))),
        temporary_columns=(delay_1, delay_10, delay_20, trend),
    )


def alpha050() -> AlphaFactor:
    # Alpha#050: (-1 * ts_max(correlation(rank(volume), rank(vwap), 5), 5))
    ranked_volume = _tmp("alpha050", "rank_volume")
    ranked_vwap = _tmp("alpha050", "rank_vwap")
    corr = _tmp("alpha050", "corr")
    ranked_corr = _tmp("alpha050", "rank_corr")
    max_ranked_corr = _tmp("alpha050", "max_rank_corr")
    return AlphaFactor(
        name="alpha050",
        stages=(
            (
                rank(schema.VOLUME).alias(ranked_volume),
                rank(schema.VWAP).alias(ranked_vwap),
            ),
            (ts_corr(ranked_volume, ranked_vwap, 5).alias(corr),),
            (rank(corr).alias(ranked_corr),),
            (ts_max(ranked_corr, 5).alias(max_ranked_corr),),
        ),
        expr=-pl.col(max_ranked_corr),
        temporary_columns=(
            ranked_volume,
            ranked_vwap,
            corr,
            ranked_corr,
            max_ranked_corr,
        ),
    )


def alpha051() -> AlphaFactor:
    # Alpha#051: (((((delay(close, 20) - delay(close, 10)) / 10)
    #             - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1
    #             : ((-1) * (close - delay(close, 1))))
    delay_1 = _tmp("alpha051", "delay_close_1")
    delay_10 = _tmp("alpha051", "delay_close_10")
    delay_20 = _tmp("alpha051", "delay_close_20")
    trend = _tmp("alpha051", "trend")
    return AlphaFactor(
        name="alpha051",
        stages=(
            (
                delay(schema.CLOSE, 1).alias(delay_1),
                delay(schema.CLOSE, 10).alias(delay_10),
                delay(schema.CLOSE, 20).alias(delay_20),
            ),
            (
                (
                    ((pl.col(delay_20) - pl.col(delay_10)) / 10)
                    - ((pl.col(delay_10) - pl.col(schema.CLOSE)) / 10)
                ).alias(trend),
            ),
        ),
        expr=pl.when(pl.col(trend) < -0.05)
        .then(1)
        .otherwise(-(pl.col(schema.CLOSE) - pl.col(delay_1))),
        temporary_columns=(delay_1, delay_10, delay_20, trend),
    )


def alpha052() -> AlphaFactor:
    # Alpha#052: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5))
    #             * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    low_min = _tmp("alpha052", "min_low")
    delayed_low_min = _tmp("alpha052", "delay_min_low")
    returns_sum_20 = _tmp("alpha052", "sum_returns_20")
    returns_sum_240 = _tmp("alpha052", "sum_returns_240")
    returns_diff_mean = _tmp("alpha052", "returns_diff_mean")
    ranked_returns_diff_mean = _tmp("alpha052", "rank_returns_diff_mean")
    volume_ts_rank = _tmp("alpha052", "ts_rank_volume")
    return AlphaFactor(
        name="alpha052",
        stages=(
            (
                ts_min(schema.LOW, 5).alias(low_min),
                ts_sum(schema.RETURNS, 20).alias(returns_sum_20),
                ts_sum(schema.RETURNS, 240).alias(returns_sum_240),
                ts_rank(schema.VOLUME, 5).alias(volume_ts_rank),
            ),
            (
                delay(low_min, 5).alias(delayed_low_min),
                ((pl.col(returns_sum_240) - pl.col(returns_sum_20)) / 220).alias(
                    returns_diff_mean
                ),
            ),
            (rank(returns_diff_mean).alias(ranked_returns_diff_mean),),
        ),
        expr=(
            ((-pl.col(low_min)) + pl.col(delayed_low_min))
            * pl.col(ranked_returns_diff_mean)
            * pl.col(volume_ts_rank)
        ),
        temporary_columns=(
            low_min,
            delayed_low_min,
            returns_sum_20,
            returns_sum_240,
            returns_diff_mean,
            ranked_returns_diff_mean,
            volume_ts_rank,
        ),
    )


def alpha053() -> AlphaFactor:
    # Alpha#053: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    value = _tmp("alpha053", "value")
    raw_value = (
        (pl.col(schema.CLOSE) - pl.col(schema.LOW))
        - (pl.col(schema.HIGH) - pl.col(schema.CLOSE))
    ) / (pl.col(schema.CLOSE) - pl.col(schema.LOW))
    return AlphaFactor(
        name="alpha053",
        stages=(
            (
                pl.when(raw_value.is_finite())
                .then(raw_value)
                .otherwise(None)
                .alias(value),
            ),
        ),
        expr=-delta(value, 9),
        temporary_columns=(value,),
    )


def alpha054() -> AlphaFactor:
    # Alpha#054: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    raw_value = (
        -(pl.col(schema.LOW) - pl.col(schema.CLOSE)) * pl.col(schema.OPEN).pow(5)
    ) / ((pl.col(schema.LOW) - pl.col(schema.HIGH)) * pl.col(schema.CLOSE).pow(5))
    return AlphaFactor(
        name="alpha054",
        stages=(),
        expr=pl.when(raw_value.is_finite()).then(raw_value).otherwise(None),
    )


def alpha055() -> AlphaFactor:
    # Alpha#055: (-1 * correlation(rank(((close - ts_min(low, 12))
    #             / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
    low_min = _tmp("alpha055", "min_low")
    high_max = _tmp("alpha055", "max_high")
    price_position = _tmp("alpha055", "price_position")
    ranked_price_position = _tmp("alpha055", "rank_price_position")
    ranked_volume = _tmp("alpha055", "rank_volume")
    return AlphaFactor(
        name="alpha055",
        stages=(
            (
                ts_min(schema.LOW, 12).alias(low_min),
                ts_max(schema.HIGH, 12).alias(high_max),
            ),
            (
                (
                    (pl.col(schema.CLOSE) - pl.col(low_min))
                    / (pl.col(high_max) - pl.col(low_min))
                ).alias(price_position),
            ),
            (
                rank(price_position).alias(ranked_price_position),
                rank(schema.VOLUME).alias(ranked_volume),
            ),
        ),
        expr=-ts_corr(ranked_price_position, ranked_volume, 6),
        temporary_columns=(
            low_min,
            high_max,
            price_position,
            ranked_price_position,
            ranked_volume,
        ),
    )


def alpha056() -> AlphaFactor:
    # Alpha#056: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3)))
    #             * rank((returns * cap)))))
    # Requires market-cap (cap) data; not implemented.
    raise NotImplementedError("alpha056 requires market-cap (cap) data")


def alpha057() -> AlphaFactor:
    # Alpha#057: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    arg_max = _tmp("alpha057", "arg_max_close")
    ranked_arg_max = _tmp("alpha057", "rank_arg_max_close")
    decayed_rank = _tmp("alpha057", "decay_rank_arg_max")
    return AlphaFactor(
        name="alpha057",
        stages=(
            (ts_arg_max(schema.CLOSE, 30).alias(arg_max),),
            (rank(arg_max).alias(ranked_arg_max),),
            (decay_linear(ranked_arg_max, 2).alias(decayed_rank),),
        ),
        expr=-((pl.col(schema.CLOSE) - pl.col(schema.VWAP)) / pl.col(decayed_rank)),
        temporary_columns=(arg_max, ranked_arg_max, decayed_rank),
    )


def alpha058() -> AlphaFactor:
    # Alpha#058: (-1 * Ts_Rank(decay_linear(correlation(
    #             IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
    # Requires IndNeutralize (sector classification); not implemented.
    raise NotImplementedError("alpha058 requires IndNeutralize (sector classification)")


def alpha059() -> AlphaFactor:
    # Alpha#059: (-1 * Ts_Rank(decay_linear(correlation(
    #             IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry),
    #             volume, 4.25197), 16.2289), 8.19648))
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha059 requires IndNeutralize (industry classification)"
    )


def alpha060() -> AlphaFactor:
    # Alpha#060: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close))
    #             / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
    price_volume = _tmp("alpha060", "price_volume")
    ranked_price_volume = _tmp("alpha060", "rank_price_volume")
    scaled_ranked_price_volume = _tmp("alpha060", "scale_rank_price_volume")
    arg_max = _tmp("alpha060", "arg_max_close")
    ranked_arg_max = _tmp("alpha060", "rank_arg_max_close")
    scaled_ranked_arg_max = _tmp("alpha060", "scale_rank_arg_max")
    return AlphaFactor(
        name="alpha060",
        stages=(
            (
                (
                    (
                        (
                            (pl.col(schema.CLOSE) - pl.col(schema.LOW))
                            - (pl.col(schema.HIGH) - pl.col(schema.CLOSE))
                        )
                        / (pl.col(schema.HIGH) - pl.col(schema.LOW))
                    )
                    * pl.col(schema.VOLUME)
                ).alias(price_volume),
                ts_arg_max(schema.CLOSE, 10).alias(arg_max),
            ),
            (
                rank(price_volume).alias(ranked_price_volume),
                rank(arg_max).alias(ranked_arg_max),
            ),
            (
                scale(ranked_price_volume).alias(scaled_ranked_price_volume),
                scale(ranked_arg_max).alias(scaled_ranked_arg_max),
            ),
        ),
        expr=-(2 * pl.col(scaled_ranked_price_volume) - pl.col(scaled_ranked_arg_max)),
        temporary_columns=(
            price_volume,
            ranked_price_volume,
            scaled_ranked_price_volume,
            arg_max,
            ranked_arg_max,
            scaled_ranked_arg_max,
        ),
    )


def alpha061() -> AlphaFactor:
    # Alpha#061: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    vwap_minus_min = _tmp("alpha061", "vwap_minus_min")
    adv180 = _tmp("alpha061", "adv180")
    corr = _tmp("alpha061", "corr")
    rank_left = _tmp("alpha061", "rank_left")
    rank_right = _tmp("alpha061", "rank_right")
    return AlphaFactor(
        name="alpha061",
        stages=(
            (
                (pl.col(schema.VWAP) - ts_min(schema.VWAP, 16)).alias(vwap_minus_min),
                ts_mean(schema.VOLUME, 180).alias(adv180),
            ),
            (ts_corr(schema.VWAP, adv180, 18).alias(corr),),
            (
                rank(vwap_minus_min).alias(rank_left),
                rank(corr).alias(rank_right),
            ),
        ),
        expr=(pl.col(rank_left) < pl.col(rank_right)).cast(pl.Float64),
        temporary_columns=(vwap_minus_min, adv180, corr, rank_left, rank_right),
    )


def alpha062() -> AlphaFactor:
    # Alpha#062: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open)
    #             + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    adv20 = _tmp("alpha062", "adv20")
    sum_adv20 = _tmp("alpha062", "sum_adv20")
    corr = _tmp("alpha062", "corr")
    rank_corr = _tmp("alpha062", "rank_corr")
    rank_open = _tmp("alpha062", "rank_open")
    rank_hl_avg = _tmp("alpha062", "rank_hl_avg")
    rank_high = _tmp("alpha062", "rank_high")
    inner_bool = _tmp("alpha062", "inner_bool")
    rank_inner_bool = _tmp("alpha062", "rank_inner_bool")
    return AlphaFactor(
        name="alpha062",
        stages=(
            (
                ts_mean(schema.VOLUME, 20).alias(adv20),
                rank(schema.OPEN).alias(rank_open),
                rank((pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2).alias(rank_hl_avg),
                rank(schema.HIGH).alias(rank_high),
            ),
            (ts_sum(adv20, 22).alias(sum_adv20),),
            (ts_corr(schema.VWAP, sum_adv20, 10).alias(corr),),
            (rank(corr).alias(rank_corr),),
            (
                (2 * pl.col(rank_open) < (pl.col(rank_hl_avg) + pl.col(rank_high)))
                .cast(pl.Float64)
                .alias(inner_bool),
            ),
            (rank(inner_bool).alias(rank_inner_bool),),
        ),
        expr=-((pl.col(rank_corr) < pl.col(rank_inner_bool)).cast(pl.Float64)),
        temporary_columns=(
            adv20,
            sum_adv20,
            corr,
            rank_corr,
            rank_open,
            rank_hl_avg,
            rank_high,
            inner_bool,
            rank_inner_bool,
        ),
    )


def alpha063() -> AlphaFactor:
    # Alpha#063: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))
    #             - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))),
    #             sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha063 requires IndNeutralize (industry classification)"
    )


def alpha064() -> AlphaFactor:
    # Alpha#064: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),
    #             sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404)
    #             + (vwap * (1 - 0.178404))), 3.69741))) * -1)
    weighted_price_1 = _tmp("alpha064", "weighted_price_1")
    adv120 = _tmp("alpha064", "adv120")
    sum_wp1 = _tmp("alpha064", "sum_wp1")
    sum_adv120 = _tmp("alpha064", "sum_adv120")
    corr = _tmp("alpha064", "corr")
    rank_corr = _tmp("alpha064", "rank_corr")
    weighted_price_2 = _tmp("alpha064", "weighted_price_2")
    delta_wp2 = _tmp("alpha064", "delta_wp2")
    rank_delta = _tmp("alpha064", "rank_delta")
    return AlphaFactor(
        name="alpha064",
        stages=(
            (
                (
                    pl.col(schema.OPEN) * 0.178404 + pl.col(schema.LOW) * (1 - 0.178404)
                ).alias(weighted_price_1),
                ts_mean(schema.VOLUME, 120).alias(adv120),
                (
                    ((pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2) * 0.178404
                    + pl.col(schema.VWAP) * (1 - 0.178404)
                ).alias(weighted_price_2),
            ),
            (
                ts_sum(weighted_price_1, 13).alias(sum_wp1),
                ts_sum(adv120, 13).alias(sum_adv120),
                delta(weighted_price_2, 4).alias(delta_wp2),
            ),
            (ts_corr(sum_wp1, sum_adv120, 17).alias(corr),),
            (
                rank(corr).alias(rank_corr),
                rank(delta_wp2).alias(rank_delta),
            ),
        ),
        expr=-((pl.col(rank_corr) < pl.col(rank_delta)).cast(pl.Float64)),
        temporary_columns=(
            weighted_price_1,
            adv120,
            sum_wp1,
            sum_adv120,
            corr,
            rank_corr,
            weighted_price_2,
            delta_wp2,
            rank_delta,
        ),
    )


def alpha065() -> AlphaFactor:
    # Alpha#065: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))),
    #             sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    weighted_price = _tmp("alpha065", "weighted_price")
    adv60 = _tmp("alpha065", "adv60")
    sum_adv60 = _tmp("alpha065", "sum_adv60")
    corr = _tmp("alpha065", "corr")
    rank_corr = _tmp("alpha065", "rank_corr")
    open_minus_min = _tmp("alpha065", "open_minus_min")
    rank_open_minus_min = _tmp("alpha065", "rank_open_minus_min")
    return AlphaFactor(
        name="alpha065",
        stages=(
            (
                (
                    pl.col(schema.OPEN) * 0.00817205
                    + pl.col(schema.VWAP) * (1 - 0.00817205)
                ).alias(weighted_price),
                ts_mean(schema.VOLUME, 60).alias(adv60),
                (pl.col(schema.OPEN) - ts_min(schema.OPEN, 14)).alias(open_minus_min),
            ),
            (ts_sum(adv60, 9).alias(sum_adv60),),
            (ts_corr(weighted_price, sum_adv60, 6).alias(corr),),
            (
                rank(corr).alias(rank_corr),
                rank(open_minus_min).alias(rank_open_minus_min),
            ),
        ),
        expr=-((pl.col(rank_corr) < pl.col(rank_open_minus_min)).cast(pl.Float64)),
        temporary_columns=(
            weighted_price,
            adv60,
            sum_adv60,
            corr,
            rank_corr,
            open_minus_min,
            rank_open_minus_min,
        ),
    )


def alpha066() -> AlphaFactor:
    # Alpha#066: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052))
    #             + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap)
    #             / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    decayed_vwap_delta = _tmp("alpha066", "decayed_vwap_delta")
    rank_decayed = _tmp("alpha066", "rank_decayed")
    ratio = _tmp("alpha066", "ratio")
    decayed_ratio = _tmp("alpha066", "decayed_ratio")
    ts_rank_ratio = _tmp("alpha066", "ts_rank_ratio")
    return AlphaFactor(
        name="alpha066",
        stages=(
            (
                decay_linear(delta(schema.VWAP, 4), 7).alias(decayed_vwap_delta),
                (
                    (pl.col(schema.LOW) - pl.col(schema.VWAP))
                    / (
                        pl.col(schema.OPEN)
                        - (pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2
                    )
                ).alias(ratio),
            ),
            (
                rank(decayed_vwap_delta).alias(rank_decayed),
                decay_linear(
                    pl.when(pl.col(ratio).is_finite())
                    .then(pl.col(ratio))
                    .otherwise(None),
                    11,
                ).alias(decayed_ratio),
            ),
            (ts_rank(decayed_ratio, 7).alias(ts_rank_ratio),),
        ),
        expr=-(pl.col(rank_decayed) + pl.col(ts_rank_ratio)),
        temporary_columns=(
            decayed_vwap_delta,
            rank_decayed,
            ratio,
            decayed_ratio,
            ts_rank_ratio,
        ),
    )


def alpha067() -> AlphaFactor:
    # Alpha#067: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(
    #             IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry),
    #             6.02936))) * -1)
    # Requires IndNeutralize (sector/subindustry classification); not implemented.
    raise NotImplementedError(
        "alpha067 requires IndNeutralize (sector/subindustry classification)"
    )


def alpha068() -> AlphaFactor:
    # Alpha#068: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333)
    #             < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    adv15 = _tmp("alpha068", "adv15")
    ranked_high = _tmp("alpha068", "rank_high")
    ranked_adv15 = _tmp("alpha068", "rank_adv15")
    corr = _tmp("alpha068", "corr")
    ts_rank_corr = _tmp("alpha068", "ts_rank_corr")
    weighted_price = _tmp("alpha068", "weighted_price")
    delta_wp = _tmp("alpha068", "delta_wp")
    rank_delta = _tmp("alpha068", "rank_delta")
    return AlphaFactor(
        name="alpha068",
        stages=(
            (
                ts_mean(schema.VOLUME, 15).alias(adv15),
                rank(schema.HIGH).alias(ranked_high),
                (
                    pl.col(schema.CLOSE) * 0.518371
                    + pl.col(schema.LOW) * (1 - 0.518371)
                ).alias(weighted_price),
            ),
            (
                rank(adv15).alias(ranked_adv15),
                delta(weighted_price, 1).alias(delta_wp),
            ),
            (ts_corr(ranked_high, ranked_adv15, 9).alias(corr),),
            (ts_rank(corr, 14).alias(ts_rank_corr),),
            (rank(delta_wp).alias(rank_delta),),
        ),
        expr=-((pl.col(ts_rank_corr) < pl.col(rank_delta)).cast(pl.Float64)),
        temporary_columns=(
            adv15,
            ranked_high,
            ranked_adv15,
            corr,
            ts_rank_corr,
            weighted_price,
            delta_wp,
            rank_delta,
        ),
    )


def alpha069() -> AlphaFactor:
    # Alpha#069: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))
    #             ^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))),
    #             adv20, 4.92416), 9.0615)) * -1)
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha069 requires IndNeutralize (industry classification)"
    )


def alpha070() -> AlphaFactor:
    # Alpha#070: ((rank(delta(vwap, 1.29456))
    #             ^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry),
    #             adv50, 17.8256), 17.9171)) * -1)
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha070 requires IndNeutralize (industry classification)"
    )


def alpha071() -> AlphaFactor:
    # Alpha#071: (max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976),
    #             Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948),
    #             Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388)))
    ts_rank_close = _tmp("alpha071", "ts_rank_close")
    adv180 = _tmp("alpha071", "adv180")
    ts_rank_adv180 = _tmp("alpha071", "ts_rank_adv180")
    corr = _tmp("alpha071", "corr")
    decayed_corr = _tmp("alpha071", "decayed_corr")
    left = _tmp("alpha071", "left")
    price_diff = _tmp("alpha071", "price_diff")
    ranked_price_diff = _tmp("alpha071", "ranked_price_diff")
    signed_power_rank = _tmp("alpha071", "signed_power_rank")
    decayed_rank = _tmp("alpha071", "decayed_rank")
    right = _tmp("alpha071", "right")
    return AlphaFactor(
        name="alpha071",
        stages=(
            (
                ts_rank(schema.CLOSE, 4).alias(ts_rank_close),
                ts_mean(schema.VOLUME, 180).alias(adv180),
                (
                    (pl.col(schema.LOW) + pl.col(schema.OPEN))
                    - (pl.col(schema.VWAP) + pl.col(schema.VWAP))
                ).alias(price_diff),
            ),
            (
                ts_rank(adv180, 12).alias(ts_rank_adv180),
                rank(price_diff).alias(ranked_price_diff),
            ),
            (
                ts_corr(ts_rank_close, ts_rank_adv180, 18).alias(corr),
                signed_power(ranked_price_diff, 2).alias(signed_power_rank),
            ),
            (
                decay_linear(corr, 4).alias(decayed_corr),
                decay_linear(signed_power_rank, 16).alias(decayed_rank),
            ),
            (
                ts_rank(decayed_corr, 16).alias(left),
                ts_rank(decayed_rank, 4).alias(right),
            ),
        ),
        expr=pl.max_horizontal(pl.col(left), pl.col(right)),
        temporary_columns=(
            ts_rank_close,
            adv180,
            ts_rank_adv180,
            corr,
            decayed_corr,
            left,
            price_diff,
            ranked_price_diff,
            signed_power_rank,
            decayed_rank,
            right,
        ),
    )


def alpha072() -> AlphaFactor:
    # Alpha#072: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519))
    #             / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469),
    #             Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
    hl_avg = _tmp("alpha072", "hl_avg")
    adv40 = _tmp("alpha072", "adv40")
    corr_hl = _tmp("alpha072", "corr_hl")
    decayed_corr_hl = _tmp("alpha072", "decayed_corr_hl")
    ranked_decayed_hl = _tmp("alpha072", "ranked_decayed_hl")
    ts_rank_vwap = _tmp("alpha072", "ts_rank_vwap")
    ts_rank_volume = _tmp("alpha072", "ts_rank_volume")
    corr_ranked = _tmp("alpha072", "corr_ranked")
    decayed_corr_ranked = _tmp("alpha072", "decayed_corr_ranked")
    ranked_decayed_ranked = _tmp("alpha072", "ranked_decayed_ranked")
    return AlphaFactor(
        name="alpha072",
        stages=(
            (
                ((pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2).alias(hl_avg),
                ts_mean(schema.VOLUME, 40).alias(adv40),
                ts_rank(schema.VWAP, 4).alias(ts_rank_vwap),
                ts_rank(schema.VOLUME, 19).alias(ts_rank_volume),
            ),
            (
                ts_corr(hl_avg, adv40, 9).alias(corr_hl),
                ts_corr(ts_rank_vwap, ts_rank_volume, 7).alias(corr_ranked),
            ),
            (
                decay_linear(corr_hl, 10).alias(decayed_corr_hl),
                decay_linear(corr_ranked, 3).alias(decayed_corr_ranked),
            ),
            (
                rank(decayed_corr_hl).alias(ranked_decayed_hl),
                rank(decayed_corr_ranked).alias(ranked_decayed_ranked),
            ),
        ),
        expr=pl.col(ranked_decayed_hl) / pl.col(ranked_decayed_ranked),
        temporary_columns=(
            hl_avg,
            adv40,
            corr_hl,
            decayed_corr_hl,
            ranked_decayed_hl,
            ts_rank_vwap,
            ts_rank_volume,
            corr_ranked,
            decayed_corr_ranked,
            ranked_decayed_ranked,
        ),
    )


def alpha073() -> AlphaFactor:
    # Alpha#073: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
    #             Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608)
    #             / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    decayed_vwap_delta = _tmp("alpha073", "decayed_vwap_delta")
    ranked_decayed = _tmp("alpha073", "ranked_decayed")
    wp = _tmp("alpha073", "wp")
    delta_wp = _tmp("alpha073", "delta_wp")
    neg_ratio = _tmp("alpha073", "neg_ratio")
    decayed_ratio = _tmp("alpha073", "decayed_ratio")
    right = _tmp("alpha073", "right")
    return AlphaFactor(
        name="alpha073",
        stages=(
            (
                decay_linear(delta(schema.VWAP, 5), 3).alias(decayed_vwap_delta),
                (
                    pl.col(schema.OPEN) * 0.147155 + pl.col(schema.LOW) * (1 - 0.147155)
                ).alias(wp),
            ),
            (
                rank(decayed_vwap_delta).alias(ranked_decayed),
                delta(wp, 2).alias(delta_wp),
            ),
            (
                pl.when(pl.col(wp).abs() > 0)
                .then(-pl.col(delta_wp) / pl.col(wp))
                .otherwise(None)
                .alias(neg_ratio),
            ),
            (decay_linear(neg_ratio, 3).alias(decayed_ratio),),
            (ts_rank(decayed_ratio, 17).alias(right),),
        ),
        expr=-(pl.max_horizontal(pl.col(ranked_decayed), pl.col(right))),
        temporary_columns=(
            decayed_vwap_delta,
            ranked_decayed,
            wp,
            delta_wp,
            neg_ratio,
            decayed_ratio,
            right,
        ),
    )


def alpha074() -> AlphaFactor:
    # Alpha#074: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365))
    #             < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))),
    #             rank(volume), 11.4791))) * -1)
    adv30 = _tmp("alpha074", "adv30")
    sum_adv30 = _tmp("alpha074", "sum_adv30")
    corr_close = _tmp("alpha074", "corr_close")
    rank_corr_close = _tmp("alpha074", "rank_corr_close")
    weighted_price = _tmp("alpha074", "weighted_price")
    ranked_weighted = _tmp("alpha074", "ranked_weighted")
    ranked_volume = _tmp("alpha074", "ranked_volume")
    corr_ranked = _tmp("alpha074", "corr_ranked")
    rank_corr_ranked = _tmp("alpha074", "rank_corr_ranked")
    return AlphaFactor(
        name="alpha074",
        stages=(
            (
                ts_mean(schema.VOLUME, 30).alias(adv30),
                (
                    pl.col(schema.HIGH) * 0.0261661
                    + pl.col(schema.VWAP) * (1 - 0.0261661)
                ).alias(weighted_price),
            ),
            (
                ts_sum(adv30, 37).alias(sum_adv30),
                rank(weighted_price).alias(ranked_weighted),
                rank(schema.VOLUME).alias(ranked_volume),
            ),
            (
                ts_corr(schema.CLOSE, sum_adv30, 15).alias(corr_close),
                ts_corr(ranked_weighted, ranked_volume, 11).alias(corr_ranked),
            ),
            (
                rank(corr_close).alias(rank_corr_close),
                rank(corr_ranked).alias(rank_corr_ranked),
            ),
        ),
        expr=-((pl.col(rank_corr_close) < pl.col(rank_corr_ranked)).cast(pl.Float64)),
        temporary_columns=(
            adv30,
            sum_adv30,
            corr_close,
            rank_corr_close,
            weighted_price,
            ranked_weighted,
            ranked_volume,
            corr_ranked,
            rank_corr_ranked,
        ),
    )


def alpha075() -> AlphaFactor:
    # Alpha#075: (rank(correlation(vwap, volume, 4.24304))
    #             < rank(correlation(rank(low), rank(adv50), 12.4413)))
    corr_vwap = _tmp("alpha075", "corr_vwap")
    rank_corr_vwap = _tmp("alpha075", "rank_corr_vwap")
    adv50 = _tmp("alpha075", "adv50")
    ranked_low = _tmp("alpha075", "ranked_low")
    ranked_adv50 = _tmp("alpha075", "ranked_adv50")
    corr_ranked = _tmp("alpha075", "corr_ranked")
    rank_corr_ranked = _tmp("alpha075", "rank_corr_ranked")
    return AlphaFactor(
        name="alpha075",
        stages=(
            (
                ts_corr(schema.VWAP, schema.VOLUME, 4).alias(corr_vwap),
                ts_mean(schema.VOLUME, 50).alias(adv50),
            ),
            (
                rank(corr_vwap).alias(rank_corr_vwap),
                rank(schema.LOW).alias(ranked_low),
                rank(adv50).alias(ranked_adv50),
            ),
            (ts_corr(ranked_low, ranked_adv50, 12).alias(corr_ranked),),
            (rank(corr_ranked).alias(rank_corr_ranked),),
        ),
        expr=(pl.col(rank_corr_vwap) < pl.col(rank_corr_ranked)).cast(pl.Float64),
        temporary_columns=(
            corr_vwap,
            rank_corr_vwap,
            adv50,
            ranked_low,
            ranked_adv50,
            corr_ranked,
            rank_corr_ranked,
        ),
    )


def alpha076() -> AlphaFactor:
    # Alpha#076: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
    #             Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector),
    #             adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
    # Requires IndNeutralize (sector classification); not implemented.
    raise NotImplementedError("alpha076 requires IndNeutralize (sector classification)")


def alpha077() -> AlphaFactor:
    # Alpha#077: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
    #              rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    hl_avg = _tmp("alpha077", "hl_avg")
    price_diff = _tmp("alpha077", "price_diff")
    adv40 = _tmp("alpha077", "adv40")
    decayed_price = _tmp("alpha077", "decayed_price")
    ranked_decayed_price = _tmp("alpha077", "ranked_decayed_price")
    corr = _tmp("alpha077", "corr")
    decayed_corr = _tmp("alpha077", "decayed_corr")
    ranked_decayed_corr = _tmp("alpha077", "ranked_decayed_corr")
    return AlphaFactor(
        name="alpha077",
        stages=(
            (
                ((pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2).alias(hl_avg),
                ts_mean(schema.VOLUME, 40).alias(adv40),
            ),
            (
                (
                    pl.col(hl_avg)
                    + pl.col(schema.HIGH)
                    - (pl.col(schema.VWAP) + pl.col(schema.HIGH))
                ).alias(price_diff),
                ts_corr(hl_avg, adv40, 3).alias(corr),
            ),
            (
                decay_linear(price_diff, 20).alias(decayed_price),
                decay_linear(corr, 6).alias(decayed_corr),
            ),
            (
                rank(decayed_price).alias(ranked_decayed_price),
                rank(decayed_corr).alias(ranked_decayed_corr),
            ),
        ),
        expr=pl.min_horizontal(
            pl.col(ranked_decayed_price), pl.col(ranked_decayed_corr)
        ),
        temporary_columns=(
            hl_avg,
            adv40,
            price_diff,
            decayed_price,
            ranked_decayed_price,
            corr,
            decayed_corr,
            ranked_decayed_corr,
        ),
    )


def alpha078() -> AlphaFactor:
    # Alpha#078: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
    #             sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    weighted_price = _tmp("alpha078", "weighted_price")
    adv40 = _tmp("alpha078", "adv40")
    sum_wp = _tmp("alpha078", "sum_wp")
    sum_adv40 = _tmp("alpha078", "sum_adv40")
    corr_left = _tmp("alpha078", "corr_left")
    ranked_corr_left = _tmp("alpha078", "ranked_corr_left")
    ranked_vwap = _tmp("alpha078", "ranked_vwap")
    ranked_volume = _tmp("alpha078", "ranked_volume")
    corr_right = _tmp("alpha078", "corr_right")
    ranked_corr_right = _tmp("alpha078", "ranked_corr_right")
    return AlphaFactor(
        name="alpha078",
        stages=(
            (
                (
                    pl.col(schema.LOW) * 0.352233 + pl.col(schema.VWAP) * (1 - 0.352233)
                ).alias(weighted_price),
                ts_mean(schema.VOLUME, 40).alias(adv40),
                rank(schema.VWAP).alias(ranked_vwap),
                rank(schema.VOLUME).alias(ranked_volume),
            ),
            (
                ts_sum(weighted_price, 20).alias(sum_wp),
                ts_sum(adv40, 20).alias(sum_adv40),
                ts_corr(ranked_vwap, ranked_volume, 6).alias(corr_right),
            ),
            (
                ts_corr(sum_wp, sum_adv40, 7).alias(corr_left),
                rank(corr_right).alias(ranked_corr_right),
            ),
            (rank(corr_left).alias(ranked_corr_left),),
        ),
        expr=signed_power(ranked_corr_left, ranked_corr_right),
        temporary_columns=(
            weighted_price,
            adv40,
            sum_wp,
            sum_adv40,
            corr_left,
            ranked_corr_left,
            ranked_vwap,
            ranked_volume,
            corr_right,
            ranked_corr_right,
        ),
    )


def alpha079() -> AlphaFactor:
    # Alpha#079: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),
    #             IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973),
    #             Ts_Rank(adv150, 9.18637), 14.6644)))
    # Requires IndNeutralize (sector classification); not implemented.
    raise NotImplementedError("alpha079 requires IndNeutralize (sector classification)")


def alpha080() -> AlphaFactor:
    # Alpha#080: ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),
    #             IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456),
    #             5.53756)) * -1)
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha080 requires IndNeutralize (industry classification)"
    )


def alpha081() -> AlphaFactor:
    # Alpha#081: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)),
    #             14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    adv10 = _tmp("alpha081", "adv10")
    sum_adv10 = _tmp("alpha081", "sum_adv10")
    corr = _tmp("alpha081", "corr")
    ranked_corr = _tmp("alpha081", "ranked_corr")
    signed_power_rank = _tmp("alpha081", "signed_power_rank")
    rank_of_power = _tmp("alpha081", "rank_of_power")
    product_val = _tmp("alpha081", "product_val")
    log_product = _tmp("alpha081", "log_product")
    rank_log = _tmp("alpha081", "rank_log")
    ranked_vwap = _tmp("alpha081", "ranked_vwap")
    ranked_volume = _tmp("alpha081", "ranked_volume")
    corr_right = _tmp("alpha081", "corr_right")
    rank_corr_right = _tmp("alpha081", "rank_corr_right")
    return AlphaFactor(
        name="alpha081",
        stages=(
            (
                ts_mean(schema.VOLUME, 10).alias(adv10),
                rank(schema.VWAP).alias(ranked_vwap),
                rank(schema.VOLUME).alias(ranked_volume),
            ),
            (
                ts_sum(adv10, 50).alias(sum_adv10),
                ts_corr(ranked_vwap, ranked_volume, 5).alias(corr_right),
            ),
            (ts_corr(schema.VWAP, sum_adv10, 8).alias(corr),),
            (
                rank(corr).alias(ranked_corr),
                rank(corr_right).alias(rank_corr_right),
            ),
            (signed_power(ranked_corr, 4).alias(signed_power_rank),),
            (rank(signed_power_rank).alias(rank_of_power),),
            (ts_product(rank_of_power, 15).alias(product_val),),
            (
                pl.when(pl.col(product_val) > 0)
                .then(pl.col(product_val).log())
                .otherwise(None)
                .alias(log_product),
            ),
            (rank(log_product).alias(rank_log),),
        ),
        expr=-((pl.col(rank_log) < pl.col(rank_corr_right)).cast(pl.Float64)),
        temporary_columns=(
            adv10,
            sum_adv10,
            corr,
            ranked_corr,
            signed_power_rank,
            rank_of_power,
            product_val,
            log_product,
            rank_log,
            ranked_vwap,
            ranked_volume,
            corr_right,
            rank_corr_right,
        ),
    )


def alpha082() -> AlphaFactor:
    # Alpha#082: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),
    #             Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector),
    #             ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
    # Requires IndNeutralize (sector classification); not implemented.
    raise NotImplementedError("alpha082 requires IndNeutralize (sector classification)")


def alpha083() -> AlphaFactor:
    # Alpha#083: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume)))
    #             / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
    hl_range = _tmp("alpha083", "hl_range")
    mean_close_5 = _tmp("alpha083", "mean_close_5")
    range_over_mean = _tmp("alpha083", "range_over_mean")
    delayed_rom = _tmp("alpha083", "delayed_rom")
    ranked_delayed = _tmp("alpha083", "ranked_delayed")
    ranked_volume = _tmp("alpha083", "ranked_volume")
    ranked_ranked_volume = _tmp("alpha083", "ranked_ranked_volume")
    numerator = _tmp("alpha083", "numerator")
    denominator = _tmp("alpha083", "denominator")
    return AlphaFactor(
        name="alpha083",
        stages=(
            (
                (pl.col(schema.HIGH) - pl.col(schema.LOW)).alias(hl_range),
                (ts_sum(schema.CLOSE, 5) / 5).alias(mean_close_5),
            ),
            (
                pl.when(pl.col(mean_close_5).abs() > 0)
                .then(pl.col(hl_range) / pl.col(mean_close_5))
                .otherwise(None)
                .alias(range_over_mean),
                rank(schema.VOLUME).alias(ranked_volume),
            ),
            (
                delay(range_over_mean, 2).alias(delayed_rom),
                rank(ranked_volume).alias(ranked_ranked_volume),
            ),
            (
                rank(delayed_rom).alias(ranked_delayed),
                pl.when((pl.col(schema.VWAP) - pl.col(schema.CLOSE)).abs() > 0)
                .then(
                    pl.col(range_over_mean)
                    / (pl.col(schema.VWAP) - pl.col(schema.CLOSE))
                )
                .otherwise(None)
                .alias(denominator),
            ),
            ((pl.col(ranked_delayed) * pl.col(ranked_ranked_volume)).alias(numerator),),
        ),
        expr=pl.when(pl.col(denominator).is_finite() & pl.col(denominator).abs().gt(0))
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None),
        temporary_columns=(
            hl_range,
            mean_close_5,
            range_over_mean,
            delayed_rom,
            ranked_delayed,
            ranked_volume,
            ranked_ranked_volume,
            numerator,
            denominator,
        ),
    )


def alpha084() -> AlphaFactor:
    # Alpha#084: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
    vwap_max = _tmp("alpha084", "vwap_max")
    vwap_diff = _tmp("alpha084", "vwap_diff")
    ts_ranked = _tmp("alpha084", "ts_ranked")
    close_delta = _tmp("alpha084", "close_delta")
    return AlphaFactor(
        name="alpha084",
        stages=(
            (
                ts_max(schema.VWAP, 15).alias(vwap_max),
                delta(schema.CLOSE, 5).alias(close_delta),
            ),
            ((pl.col(schema.VWAP) - pl.col(vwap_max)).alias(vwap_diff),),
            (ts_rank(vwap_diff, 20).alias(ts_ranked),),
        ),
        expr=(
            pl.col(ts_ranked).sign() * pl.col(ts_ranked).abs().pow(pl.col(close_delta))
        ),
        temporary_columns=(vwap_max, vwap_diff, ts_ranked, close_delta),
    )


def alpha085() -> AlphaFactor:
    # Alpha#085: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))
    #             ^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596),
    #             Ts_Rank(volume, 10.1595), 7.11408)))
    weighted_price = _tmp("alpha085", "weighted_price")
    adv30 = _tmp("alpha085", "adv30")
    corr1 = _tmp("alpha085", "corr1")
    rank_corr1 = _tmp("alpha085", "rank_corr1")
    hl_avg = _tmp("alpha085", "hl_avg")
    ts_rank_hl = _tmp("alpha085", "ts_rank_hl")
    ts_rank_vol = _tmp("alpha085", "ts_rank_vol")
    corr2 = _tmp("alpha085", "corr2")
    rank_corr2 = _tmp("alpha085", "rank_corr2")
    return AlphaFactor(
        name="alpha085",
        stages=(
            (
                (
                    pl.col(schema.HIGH) * 0.876703
                    + pl.col(schema.CLOSE) * (1 - 0.876703)
                ).alias(weighted_price),
                ts_mean(schema.VOLUME, 30).alias(adv30),
                ((pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2).alias(hl_avg),
            ),
            (
                ts_corr(weighted_price, adv30, 10).alias(corr1),
                ts_rank(hl_avg, 4).alias(ts_rank_hl),
                ts_rank(schema.VOLUME, 10).alias(ts_rank_vol),
            ),
            (
                rank(corr1).alias(rank_corr1),
                ts_corr(ts_rank_hl, ts_rank_vol, 7).alias(corr2),
            ),
            (rank(corr2).alias(rank_corr2),),
        ),
        expr=(
            pl.col(rank_corr1).sign() * pl.col(rank_corr1).abs().pow(pl.col(rank_corr2))
        ),
        temporary_columns=(
            weighted_price,
            adv30,
            corr1,
            rank_corr1,
            hl_avg,
            ts_rank_hl,
            ts_rank_vol,
            corr2,
            rank_corr2,
        ),
    )


def alpha086() -> AlphaFactor:
    # Alpha#086: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195)
    #             < rank(((open + close) - (vwap + open)))) * -1)
    adv20 = _tmp("alpha086", "adv20")
    sum_adv20 = _tmp("alpha086", "sum_adv20")
    corr = _tmp("alpha086", "corr")
    ts_rank_corr = _tmp("alpha086", "ts_rank_corr")
    close_minus_vwap = _tmp("alpha086", "close_minus_vwap")
    ranked_diff = _tmp("alpha086", "ranked_diff")
    return AlphaFactor(
        name="alpha086",
        stages=(
            (
                ts_mean(schema.VOLUME, 20).alias(adv20),
                (pl.col(schema.CLOSE) - pl.col(schema.VWAP)).alias(close_minus_vwap),
            ),
            (ts_sum(adv20, 15).alias(sum_adv20),),
            (ts_corr(schema.CLOSE, sum_adv20, 6).alias(corr),),
            (
                ts_rank(corr, 20).alias(ts_rank_corr),
                rank(close_minus_vwap).alias(ranked_diff),
            ),
        ),
        expr=-((pl.col(ts_rank_corr) < pl.col(ranked_diff)).cast(pl.Float64)),
        temporary_columns=(
            adv20,
            sum_adv20,
            corr,
            ts_rank_corr,
            close_minus_vwap,
            ranked_diff,
        ),
    )


def alpha087() -> AlphaFactor:
    # Alpha#087: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),
    #             1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(
    #             IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha087 requires IndNeutralize (industry classification)"
    )


def alpha088() -> AlphaFactor:
    # Alpha#088: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)),
    #              Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966),
    #              8.01266), 6.65053), 2.61957))
    ranked_open = _tmp("alpha088", "ranked_open")
    ranked_low = _tmp("alpha088", "ranked_low")
    ranked_high = _tmp("alpha088", "ranked_high")
    ranked_close = _tmp("alpha088", "ranked_close")
    rank_diff = _tmp("alpha088", "rank_diff")
    decayed_rank_diff = _tmp("alpha088", "decayed_rank_diff")
    left = _tmp("alpha088", "left")
    adv60 = _tmp("alpha088", "adv60")
    ts_rank_close = _tmp("alpha088", "ts_rank_close")
    ts_rank_adv60 = _tmp("alpha088", "ts_rank_adv60")
    corr = _tmp("alpha088", "corr")
    decayed_corr = _tmp("alpha088", "decayed_corr")
    right = _tmp("alpha088", "right")
    return AlphaFactor(
        name="alpha088",
        stages=(
            (
                rank(schema.OPEN).alias(ranked_open),
                rank(schema.LOW).alias(ranked_low),
                rank(schema.HIGH).alias(ranked_high),
                rank(schema.CLOSE).alias(ranked_close),
                ts_mean(schema.VOLUME, 60).alias(adv60),
            ),
            (
                (
                    (pl.col(ranked_open) + pl.col(ranked_low))
                    - (pl.col(ranked_high) + pl.col(ranked_close))
                ).alias(rank_diff),
                ts_rank(schema.CLOSE, 8).alias(ts_rank_close),
            ),
            (
                decay_linear(rank_diff, 8).alias(decayed_rank_diff),
                ts_rank(adv60, 21).alias(ts_rank_adv60),
            ),
            (
                rank(decayed_rank_diff).alias(left),
                ts_corr(ts_rank_close, ts_rank_adv60, 8).alias(corr),
            ),
            (decay_linear(corr, 7).alias(decayed_corr),),
            (ts_rank(decayed_corr, 3).alias(right),),
        ),
        expr=pl.min_horizontal(pl.col(left), pl.col(right)),
        temporary_columns=(
            ranked_open,
            ranked_low,
            ranked_high,
            ranked_close,
            rank_diff,
            decayed_rank_diff,
            left,
            adv60,
            ts_rank_close,
            ts_rank_adv60,
            corr,
            decayed_corr,
            right,
        ),
    )


def alpha089() -> AlphaFactor:
    # Alpha#089: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))),
    #             adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(
    #             delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha089 requires IndNeutralize (industry classification)"
    )


def alpha090() -> AlphaFactor:
    # Alpha#090: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(
    #             IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
    # Requires IndNeutralize (subindustry classification); not implemented.
    raise NotImplementedError(
        "alpha090 requires IndNeutralize (subindustry classification)"
    )


def alpha091() -> AlphaFactor:
    # Alpha#091: ((Ts_Rank(decay_linear(decay_linear(correlation(
    #             IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667)
    #             - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha091 requires IndNeutralize (industry classification)"
    )


def alpha092() -> AlphaFactor:
    # Alpha#092: (min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),
    #             18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555),
    #             6.94024), 6.80584)))
    hl_avg = _tmp("alpha092", "hl_avg")
    bool_val = _tmp("alpha092", "bool_val")
    adv30 = _tmp("alpha092", "adv30")
    ranked_low = _tmp("alpha092", "ranked_low")
    ranked_adv30 = _tmp("alpha092", "ranked_adv30")
    decayed_bool = _tmp("alpha092", "decayed_bool")
    left = _tmp("alpha092", "left")
    corr = _tmp("alpha092", "corr")
    decayed_corr = _tmp("alpha092", "decayed_corr")
    right = _tmp("alpha092", "right")
    return AlphaFactor(
        name="alpha092",
        stages=(
            (
                ((pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2).alias(hl_avg),
                ts_mean(schema.VOLUME, 30).alias(adv30),
                rank(schema.LOW).alias(ranked_low),
            ),
            (
                (
                    (pl.col(hl_avg) + pl.col(schema.CLOSE))
                    < (pl.col(schema.LOW) + pl.col(schema.OPEN))
                )
                .cast(pl.Float64)
                .alias(bool_val),
                rank(adv30).alias(ranked_adv30),
            ),
            (
                decay_linear(bool_val, 15).alias(decayed_bool),
                ts_corr(ranked_low, ranked_adv30, 8).alias(corr),
            ),
            (
                ts_rank(decayed_bool, 19).alias(left),
                decay_linear(corr, 7).alias(decayed_corr),
            ),
            (ts_rank(decayed_corr, 7).alias(right),),
        ),
        expr=pl.min_horizontal(pl.col(left), pl.col(right)),
        temporary_columns=(
            hl_avg,
            bool_val,
            adv30,
            ranked_low,
            ranked_adv30,
            decayed_bool,
            left,
            corr,
            decayed_corr,
            right,
        ),
    )


def alpha093() -> AlphaFactor:
    # Alpha#093: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry),
    #             adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(
    #             delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha093 requires IndNeutralize (industry classification)"
    )


def alpha094() -> AlphaFactor:
    # Alpha#094: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(
    #             Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    vwap_min = _tmp("alpha094", "vwap_min")
    vwap_diff = _tmp("alpha094", "vwap_diff")
    ranked_diff = _tmp("alpha094", "ranked_diff")
    adv60 = _tmp("alpha094", "adv60")
    ts_rank_vwap = _tmp("alpha094", "ts_rank_vwap")
    ts_rank_adv60 = _tmp("alpha094", "ts_rank_adv60")
    corr = _tmp("alpha094", "corr")
    ts_rank_corr = _tmp("alpha094", "ts_rank_corr")
    return AlphaFactor(
        name="alpha094",
        stages=(
            (
                ts_min(schema.VWAP, 12).alias(vwap_min),
                ts_rank(schema.VWAP, 20).alias(ts_rank_vwap),
                ts_mean(schema.VOLUME, 60).alias(adv60),
            ),
            (
                (pl.col(schema.VWAP) - pl.col(vwap_min)).alias(vwap_diff),
                ts_rank(adv60, 4).alias(ts_rank_adv60),
            ),
            (
                rank(vwap_diff).alias(ranked_diff),
                ts_corr(ts_rank_vwap, ts_rank_adv60, 18).alias(corr),
            ),
            (ts_rank(corr, 3).alias(ts_rank_corr),),
        ),
        expr=-(
            pl.col(ranked_diff).sign()
            * pl.col(ranked_diff).abs().pow(pl.col(ts_rank_corr))
        ),
        temporary_columns=(
            vwap_min,
            vwap_diff,
            ranked_diff,
            adv60,
            ts_rank_vwap,
            ts_rank_adv60,
            corr,
            ts_rank_corr,
        ),
    )


def alpha095() -> AlphaFactor:
    # Alpha#095: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(
    #             sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    open_min = _tmp("alpha095", "open_min")
    open_diff = _tmp("alpha095", "open_diff")
    ranked_open_diff = _tmp("alpha095", "ranked_open_diff")
    hl_avg = _tmp("alpha095", "hl_avg")
    adv40 = _tmp("alpha095", "adv40")
    sum_hl = _tmp("alpha095", "sum_hl")
    sum_adv40 = _tmp("alpha095", "sum_adv40")
    corr = _tmp("alpha095", "corr")
    ranked_corr = _tmp("alpha095", "ranked_corr")
    signed_power_corr = _tmp("alpha095", "signed_power_corr")
    ts_rank_power = _tmp("alpha095", "ts_rank_power")
    return AlphaFactor(
        name="alpha095",
        stages=(
            (
                ts_min(schema.OPEN, 12).alias(open_min),
                ((pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2).alias(hl_avg),
                ts_mean(schema.VOLUME, 40).alias(adv40),
            ),
            (
                (pl.col(schema.OPEN) - pl.col(open_min)).alias(open_diff),
                ts_sum(hl_avg, 20).alias(sum_hl),
                ts_sum(adv40, 20).alias(sum_adv40),
            ),
            (
                rank(open_diff).alias(ranked_open_diff),
                ts_corr(sum_hl, sum_adv40, 13).alias(corr),
            ),
            (rank(corr).alias(ranked_corr),),
            (signed_power(ranked_corr, 5).alias(signed_power_corr),),
            (ts_rank(signed_power_corr, 12).alias(ts_rank_power),),
        ),
        expr=(pl.col(ranked_open_diff) < pl.col(ts_rank_power)).cast(pl.Float64),
        temporary_columns=(
            open_min,
            open_diff,
            ranked_open_diff,
            hl_avg,
            adv40,
            sum_hl,
            sum_adv40,
            corr,
            ranked_corr,
            signed_power_corr,
            ts_rank_power,
        ),
    )


def alpha096() -> AlphaFactor:
    # Alpha#096: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783),
    #             8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),
    #             Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    ranked_vwap = _tmp("alpha096", "ranked_vwap")
    ranked_volume = _tmp("alpha096", "ranked_volume")
    corr1 = _tmp("alpha096", "corr1")
    decayed_corr1 = _tmp("alpha096", "decayed_corr1")
    left = _tmp("alpha096", "left")
    adv60 = _tmp("alpha096", "adv60")
    ts_rank_close = _tmp("alpha096", "ts_rank_close")
    ts_rank_adv60 = _tmp("alpha096", "ts_rank_adv60")
    corr2 = _tmp("alpha096", "corr2")
    argmax_corr2 = _tmp("alpha096", "argmax_corr2")
    decayed_argmax = _tmp("alpha096", "decayed_argmax")
    right = _tmp("alpha096", "right")
    return AlphaFactor(
        name="alpha096",
        stages=(
            (
                rank(schema.VWAP).alias(ranked_vwap),
                rank(schema.VOLUME).alias(ranked_volume),
                ts_rank(schema.CLOSE, 7).alias(ts_rank_close),
                ts_mean(schema.VOLUME, 60).alias(adv60),
            ),
            (
                ts_corr(ranked_vwap, ranked_volume, 4).alias(corr1),
                ts_rank(adv60, 4).alias(ts_rank_adv60),
            ),
            (
                decay_linear(corr1, 4).alias(decayed_corr1),
                ts_corr(ts_rank_close, ts_rank_adv60, 4).alias(corr2),
            ),
            (
                ts_rank(decayed_corr1, 8).alias(left),
                ts_arg_max(corr2, 13).alias(argmax_corr2),
            ),
            (decay_linear(argmax_corr2, 14).alias(decayed_argmax),),
            (ts_rank(decayed_argmax, 13).alias(right),),
        ),
        expr=-(pl.max_horizontal(pl.col(left), pl.col(right))),
        temporary_columns=(
            ranked_vwap,
            ranked_volume,
            corr1,
            decayed_corr1,
            left,
            adv60,
            ts_rank_close,
            ts_rank_adv60,
            corr2,
            argmax_corr2,
            decayed_argmax,
            right,
        ),
    )


def alpha097() -> AlphaFactor:
    # Alpha#097: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),
    #             IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(
    #             correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925),
    #             15.7152), 6.71659)) * -1)
    # Requires IndNeutralize (industry classification); not implemented.
    raise NotImplementedError(
        "alpha097 requires IndNeutralize (industry classification)"
    )


def alpha098() -> AlphaFactor:
    # Alpha#098: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088))
    #             - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15),
    #             20.8187), 8.62571), 6.95668), 8.07206)))
    adv5 = _tmp("alpha098", "adv5")
    sum_adv5 = _tmp("alpha098", "sum_adv5")
    corr1 = _tmp("alpha098", "corr1")
    decayed_corr1 = _tmp("alpha098", "decayed_corr1")
    left = _tmp("alpha098", "left")
    adv15 = _tmp("alpha098", "adv15")
    ranked_open = _tmp("alpha098", "ranked_open")
    ranked_adv15 = _tmp("alpha098", "ranked_adv15")
    corr2 = _tmp("alpha098", "corr2")
    argmin_corr2 = _tmp("alpha098", "argmin_corr2")
    ts_rank_argmin = _tmp("alpha098", "ts_rank_argmin")
    decayed_rank = _tmp("alpha098", "decayed_rank")
    right = _tmp("alpha098", "right")
    return AlphaFactor(
        name="alpha098",
        stages=(
            (
                ts_mean(schema.VOLUME, 5).alias(adv5),
                ts_mean(schema.VOLUME, 15).alias(adv15),
                rank(schema.OPEN).alias(ranked_open),
            ),
            (
                ts_sum(adv5, 26).alias(sum_adv5),
                rank(adv15).alias(ranked_adv15),
            ),
            (
                ts_corr(schema.VWAP, sum_adv5, 5).alias(corr1),
                ts_corr(ranked_open, ranked_adv15, 21).alias(corr2),
            ),
            (
                decay_linear(corr1, 7).alias(decayed_corr1),
                ts_arg_max(-pl.col(corr2), 9).alias(argmin_corr2),
            ),
            (
                rank(decayed_corr1).alias(left),
                ts_rank(argmin_corr2, 7).alias(ts_rank_argmin),
            ),
            (decay_linear(ts_rank_argmin, 8).alias(decayed_rank),),
            (rank(decayed_rank).alias(right),),
        ),
        expr=pl.col(left) - pl.col(right),
        temporary_columns=(
            adv5,
            sum_adv5,
            corr1,
            decayed_corr1,
            left,
            adv15,
            ranked_open,
            ranked_adv15,
            corr2,
            argmin_corr2,
            ts_rank_argmin,
            decayed_rank,
            right,
        ),
    )


def alpha099() -> AlphaFactor:
    # Alpha#099: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136))
    #             < rank(correlation(low, volume, 6.28259))) * -1)
    hl_avg = _tmp("alpha099", "hl_avg")
    adv60 = _tmp("alpha099", "adv60")
    sum_hl = _tmp("alpha099", "sum_hl")
    sum_adv60 = _tmp("alpha099", "sum_adv60")
    corr1 = _tmp("alpha099", "corr1")
    rank_corr1 = _tmp("alpha099", "rank_corr1")
    corr2 = _tmp("alpha099", "corr2")
    rank_corr2 = _tmp("alpha099", "rank_corr2")
    return AlphaFactor(
        name="alpha099",
        stages=(
            (
                ((pl.col(schema.HIGH) + pl.col(schema.LOW)) / 2).alias(hl_avg),
                ts_mean(schema.VOLUME, 60).alias(adv60),
            ),
            (
                ts_sum(hl_avg, 20).alias(sum_hl),
                ts_sum(adv60, 20).alias(sum_adv60),
                ts_corr(schema.LOW, schema.VOLUME, 6).alias(corr2),
            ),
            (
                ts_corr(sum_hl, sum_adv60, 9).alias(corr1),
                rank(corr2).alias(rank_corr2),
            ),
            (rank(corr1).alias(rank_corr1),),
        ),
        expr=-((pl.col(rank_corr1) < pl.col(rank_corr2)).cast(pl.Float64)),
        temporary_columns=(
            hl_avg,
            adv60,
            sum_hl,
            sum_adv60,
            corr1,
            rank_corr1,
            corr2,
            rank_corr2,
        ),
    )


def alpha100() -> AlphaFactor:
    # Alpha#100: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(
    #             (((close - low) - (high - close)) / (high - low)) * volume)),
    #             IndClass.subindustry), IndClass.subindustry))) * scale(indneutralize(
    #             (correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),
    #             IndClass.subindustry))) * (volume / adv20))))
    # Requires IndNeutralize (subindustry classification); not implemented.
    raise NotImplementedError(
        "alpha100 requires IndNeutralize (subindustry classification)"
    )


def alpha101() -> AlphaFactor:
    # Alpha#101: ((close - open) / ((high - low) + .001))
    return AlphaFactor(
        name="alpha101",
        stages=(),
        expr=(pl.col(schema.CLOSE) - pl.col(schema.OPEN))
        / (pl.col(schema.HIGH) - pl.col(schema.LOW) + 0.001),
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
    "alpha021": alpha021,
    "alpha022": alpha022,
    "alpha023": alpha023,
    "alpha024": alpha024,
    "alpha025": alpha025,
    "alpha026": alpha026,
    "alpha027": alpha027,
    "alpha028": alpha028,
    "alpha029": alpha029,
    "alpha030": alpha030,
    "alpha031": alpha031,
    "alpha032": alpha032,
    "alpha033": alpha033,
    "alpha034": alpha034,
    "alpha035": alpha035,
    "alpha036": alpha036,
    "alpha037": alpha037,
    "alpha038": alpha038,
    "alpha039": alpha039,
    "alpha040": alpha040,
    "alpha041": alpha041,
    "alpha042": alpha042,
    "alpha043": alpha043,
    "alpha044": alpha044,
    "alpha045": alpha045,
    "alpha046": alpha046,
    "alpha047": alpha047,
    "alpha049": alpha049,
    "alpha050": alpha050,
    "alpha051": alpha051,
    "alpha052": alpha052,
    "alpha053": alpha053,
    "alpha054": alpha054,
    "alpha055": alpha055,
    "alpha057": alpha057,
    "alpha060": alpha060,
    "alpha061": alpha061,
    "alpha062": alpha062,
    "alpha064": alpha064,
    "alpha065": alpha065,
    "alpha066": alpha066,
    "alpha068": alpha068,
    "alpha071": alpha071,
    "alpha072": alpha072,
    "alpha073": alpha073,
    "alpha074": alpha074,
    "alpha075": alpha075,
    "alpha077": alpha077,
    "alpha078": alpha078,
    "alpha081": alpha081,
    "alpha083": alpha083,
    "alpha084": alpha084,
    "alpha085": alpha085,
    "alpha086": alpha086,
    "alpha088": alpha088,
    "alpha092": alpha092,
    "alpha094": alpha094,
    "alpha095": alpha095,
    "alpha096": alpha096,
    "alpha098": alpha098,
    "alpha099": alpha099,
    "alpha101": alpha101,
}
