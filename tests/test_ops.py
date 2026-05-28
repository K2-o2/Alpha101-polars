from datetime import date, timedelta

import polars as pl

from alpha101.ops import delta, rank, ts_corr


def test_delta_is_computed_per_symbol() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["a", "a", "b", "b"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
            "close": [1.0, 3.0, 10.0, 13.0],
        }
    ).sort(["symbol", "date"])

    result = frame.with_columns(delta("close", 1).alias("delta"))

    assert result["delta"].to_list() == [None, 2.0, None, 3.0]


def test_rank_is_computed_per_date() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["a", "b", "c"],
            "date": [date(2024, 1, 1)] * 3,
            "close": [3.0, 1.0, 2.0],
        }
    )

    result = frame.with_columns(rank("close").alias("rank")).sort("symbol")

    assert result["rank"].to_list() == [1.0, 1 / 3, 2 / 3]


def test_rank_alias_is_cross_sectional() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["a", "b", "a", "b"],
            "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 2)],
            "close": [1.0, 2.0, 4.0, 3.0],
        }
    )

    result = frame.with_columns(rank("close").alias("rank")).sort(["date", "symbol"])

    assert result["rank"].to_list() == [0.5, 1.0, 1.0, 0.5]


def test_ts_corr_returns_one_for_identical_series() -> None:
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    frame = pl.DataFrame(
        {
            "symbol": ["a"] * 6,
            "date": days,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    result = frame.with_columns(ts_corr("x", "y", 3).alias("corr"))

    assert result["corr"].drop_nulls().round(10).to_list() == [1.0, 1.0, 1.0, 1.0]
