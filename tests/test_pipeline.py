from datetime import date, timedelta

import polars as pl
import pytest

from alpha101 import alpha_names, compute_alphas


def sample_frame() -> pl.DataFrame:
    rows = []
    for symbol_idx, symbol in enumerate(["a", "b", "c"]):
        for day in range(40):
            wave = ((day + symbol_idx) % 7 - 3) * 0.03
            close = 10.0 + symbol_idx * 0.4 + day * 0.07 + wave
            open_ = close - 0.08 + ((day * symbol_idx) % 5) * 0.015
            rows.append(
                {
                    "date": date(2024, 1, 1) + timedelta(days=day),
                    "symbol": symbol,
                    "open": open_,
                    "high": max(open_, close) + 0.2,
                    "low": min(open_, close) - 0.3,
                    "close": close,
                    "volume": 1000.0 + symbol_idx * 100 + day * (8 + symbol_idx) + (day % 3) * 17,
                    "vwap": (open_ + close) / 2,
                    "returns": None if day == 0 else 0.005 + symbol_idx * 0.001 + wave * 0.01,
                }
            )
    return pl.DataFrame(rows)


def test_compute_selected_alphas_adds_columns() -> None:
    result = compute_alphas(sample_frame(), names=["alpha002", "alpha006"])

    assert {"alpha002", "alpha006"}.issubset(result.columns)
    assert result.height == 120


def test_compute_all_registered_alphas() -> None:
    result = compute_alphas(sample_frame())

    assert set(alpha_names()).issubset(result.columns)


def test_compute_alphas_does_not_leak_temporary_columns() -> None:
    result = compute_alphas(sample_frame(), names=["alpha002", "alpha020"])

    assert not any(column.startswith("__alpha") for column in result.columns)


def test_missing_required_columns_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        compute_alphas(pl.DataFrame({"date": [date(2024, 1, 1)]}))
