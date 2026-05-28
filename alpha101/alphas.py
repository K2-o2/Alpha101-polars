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
    return AlphaFactor(
        name="alpha033",
        stages=(),
        expr=rank(-((1 - (pl.col(schema.OPEN) / pl.col(schema.CLOSE))).pow(1))),
    )


def alpha034() -> AlphaFactor:
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
    return AlphaFactor(
        name="alpha041",
        stages=(),
        expr=(pl.col(schema.HIGH) * pl.col(schema.LOW)).sqrt() - pl.col(schema.VWAP),
    )


def alpha042() -> AlphaFactor:
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
    ranked_volume = _tmp("alpha044", "rank_volume")
    return AlphaFactor(
        name="alpha044",
        stages=((rank(schema.VOLUME).alias(ranked_volume),),),
        expr=-ts_corr(schema.HIGH, ranked_volume, 5),
        temporary_columns=(ranked_volume,),
    )


def alpha045() -> AlphaFactor:
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
                rank(pl.col(schema.HIGH) - pl.col(schema.CLOSE)).alias(ranked_high_close),
                rank(pl.col(schema.VWAP) - pl.col(delayed_vwap)).alias(ranked_vwap_delta),
            ),
        ),
        expr=(
            (
                (pl.col(ranked_inverse_close) * pl.col(schema.VOLUME) / pl.col(adv20))
                * ((pl.col(schema.HIGH) * pl.col(ranked_high_close)) / pl.col(high_mean_5))
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
    # Requires industry neutralization metadata; not registered until the schema supports it.
    close_delta = _tmp("alpha048", "delta_close")
    delayed_close = _tmp("alpha048", "delay_close")
    delayed_close_delta = _tmp("alpha048", "delta_delay_close")
    corr = _tmp("alpha048", "corr")
    denominator = _tmp("alpha048", "denominator")
    return AlphaFactor(
        name="alpha048",
        stages=(
            (
                delta(schema.CLOSE, 1).alias(close_delta),
                delay(schema.CLOSE, 1).alias(delayed_close),
            ),
            (delta(delayed_close, 1).alias(delayed_close_delta),),
            (
                ts_corr(close_delta, delayed_close_delta, 250).alias(corr),
                ts_sum((pl.col(close_delta) / pl.col(delayed_close)).pow(2), 250).alias(denominator),
            ),
        ),
        expr=((pl.col(corr) * pl.col(close_delta)) / pl.col(schema.CLOSE)) / pl.col(denominator),
        temporary_columns=(close_delta, delayed_close, delayed_close_delta, corr, denominator),
    )


def alpha049() -> AlphaFactor:
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
    ranked_volume = _tmp("alpha050", "rank_volume")
    ranked_vwap = _tmp("alpha050", "rank_vwap")
    corr = _tmp("alpha050", "corr")
    ranked_corr = _tmp("alpha050", "rank_corr")
    max_ranked_corr = _tmp("alpha050", "max_rank_corr")
    return AlphaFactor(
        name="alpha050",
        stages=(
            (rank(schema.VOLUME).alias(ranked_volume), rank(schema.VWAP).alias(ranked_vwap)),
            (ts_corr(ranked_volume, ranked_vwap, 5).alias(corr),),
            (rank(corr).alias(ranked_corr),),
            (ts_max(ranked_corr, 5).alias(max_ranked_corr),),
        ),
        expr=-pl.col(max_ranked_corr),
        temporary_columns=(ranked_volume, ranked_vwap, corr, ranked_corr, max_ranked_corr),
    )


def alpha051() -> AlphaFactor:
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
                ((pl.col(returns_sum_240) - pl.col(returns_sum_20)) / 220).alias(returns_diff_mean),
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
    value = _tmp("alpha053", "value")
    raw_value = (
        ((pl.col(schema.CLOSE) - pl.col(schema.LOW)) - (pl.col(schema.HIGH) - pl.col(schema.CLOSE)))
        / (pl.col(schema.CLOSE) - pl.col(schema.LOW))
    )
    return AlphaFactor(
        name="alpha053",
        stages=(
            (
                pl.when(raw_value.is_finite()).then(raw_value).otherwise(None).alias(value),
            ),
        ),
        expr=-delta(value, 9),
        temporary_columns=(value,),
    )


def alpha054() -> AlphaFactor:
    raw_value = (
        (-(pl.col(schema.LOW) - pl.col(schema.CLOSE)) * pl.col(schema.OPEN).pow(5))
        / ((pl.col(schema.LOW) - pl.col(schema.HIGH)) * pl.col(schema.CLOSE).pow(5))
    )
    return AlphaFactor(
        name="alpha054",
        stages=(),
        expr=pl.when(raw_value.is_finite()).then(raw_value).otherwise(None),
    )


def alpha055() -> AlphaFactor:
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
        temporary_columns=(low_min, high_max, price_position, ranked_price_position, ranked_volume),
    )


def alpha056() -> AlphaFactor:
    # Requires market-cap data; not registered until the schema supports it.
    returns_sum_2 = _tmp("alpha056", "sum_returns_2")
    nested_returns_sum = _tmp("alpha056", "sum_sum_returns")
    returns_sum_10 = _tmp("alpha056", "sum_returns_10")
    returns_ratio = _tmp("alpha056", "returns_ratio")
    ranked_returns_ratio = _tmp("alpha056", "rank_returns_ratio")
    cap_value = _tmp("alpha056", "cap_value")
    ranked_cap_value = _tmp("alpha056", "rank_cap_value")
    return AlphaFactor(
        name="alpha056",
        stages=(
            (
                ts_sum(schema.RETURNS, 2).alias(returns_sum_2),
                ts_sum(schema.RETURNS, 10).alias(returns_sum_10),
                (pl.col(schema.RETURNS) * pl.col("cap")).alias(cap_value),
            ),
            (ts_sum(returns_sum_2, 3).alias(nested_returns_sum),),
            ((pl.col(returns_sum_10) / pl.col(nested_returns_sum)).alias(returns_ratio),),
            (
                rank(returns_ratio).alias(ranked_returns_ratio),
                rank(cap_value).alias(ranked_cap_value),
            ),
        ),
        expr=-(pl.col(ranked_returns_ratio) * pl.col(ranked_cap_value)),
        temporary_columns=(
            returns_sum_2,
            nested_returns_sum,
            returns_sum_10,
            returns_ratio,
            ranked_returns_ratio,
            cap_value,
            ranked_cap_value,
        ),
    )


def alpha057() -> AlphaFactor:
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
    # Requires sector neutralization metadata; not registered until the schema supports it.
    corr = _tmp("alpha058", "corr")
    decayed_corr = _tmp("alpha058", "decay_corr")
    ranked_decayed_corr = _tmp("alpha058", "ts_rank_decay_corr")
    return AlphaFactor(
        name="alpha058",
        stages=(
            (ts_corr(schema.VWAP, schema.VOLUME, 4).alias(corr),),
            (decay_linear(corr, 8).alias(decayed_corr),),
            (ts_rank(decayed_corr, 6).alias(ranked_decayed_corr),),
        ),
        expr=-pl.col(ranked_decayed_corr),
        temporary_columns=(corr, decayed_corr, ranked_decayed_corr),
    )


def alpha059() -> AlphaFactor:
    # Requires industry neutralization metadata; not registered until the schema supports it.
    corr = _tmp("alpha059", "corr")
    decayed_corr = _tmp("alpha059", "decay_corr")
    ranked_decayed_corr = _tmp("alpha059", "ts_rank_decay_corr")
    return AlphaFactor(
        name="alpha059",
        stages=(
            (ts_corr(schema.VWAP, schema.VOLUME, 4).alias(corr),),
            (decay_linear(corr, 16).alias(decayed_corr),),
            (ts_rank(decayed_corr, 8).alias(ranked_decayed_corr),),
        ),
        expr=-pl.col(ranked_decayed_corr),
        temporary_columns=(corr, decayed_corr, ranked_decayed_corr),
    )


def alpha060() -> AlphaFactor:
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
                        ((pl.col(schema.CLOSE) - pl.col(schema.LOW)) - (pl.col(schema.HIGH) - pl.col(schema.CLOSE)))
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
}
