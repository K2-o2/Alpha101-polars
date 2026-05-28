"""AkShare data loading utilities for the sample A-share universe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from alpha101 import schema


@dataclass(frozen=True)
class Stock:
    symbol: str
    name: str


SAMPLE_STOCKS: tuple[Stock, ...] = (
    Stock("sh600519", "贵州茅台"),
    Stock("sh600036", "招商银行"),
    Stock("sh601318", "中国平安"),
    Stock("sh600030", "中信证券"),
    Stock("sh600276", "恒瑞医药"),
    Stock("sz000858", "五粮液"),
    Stock("sz000333", "美的集团"),
    Stock("sz000651", "格力电器"),
    Stock("sz300750", "宁德时代"),
    Stock("sz002475", "立讯精密"),
)


def sample_symbols() -> list[str]:
    return [stock.symbol for stock in SAMPLE_STOCKS]


def _download_one(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
) -> pl.DataFrame:
    import akshare as ak

    raw = ak.stock_zh_a_daily(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    frame = pl.from_pandas(raw)
    lower_names = {name: str(name).lower() for name in frame.columns}
    frame = frame.rename(lower_names)

    if "date" not in frame.columns:
        raise ValueError(f"AkShare response for {symbol} does not include a date column")

    frame = frame.with_columns(
        pl.lit(symbol).alias(schema.SYMBOL),
        pl.col("date").cast(pl.Date).alias(schema.DATE),
    )

    if "vwap" not in frame.columns:
        if "amount" not in frame.columns:
            raise ValueError(f"AkShare response for {symbol} does not include amount for vwap")
        frame = frame.with_columns(
            pl.when(pl.col(schema.VOLUME) > 0)
            .then(pl.col("amount") / pl.col(schema.VOLUME))
            .otherwise(None)
            .alias(schema.VWAP)
        )

    return (
        frame.select(
            schema.DATE,
            schema.SYMBOL,
            schema.OPEN,
            schema.HIGH,
            schema.LOW,
            schema.CLOSE,
            schema.VOLUME,
            schema.VWAP,
        )
        .with_columns(
            pl.all().exclude(schema.DATE, schema.SYMBOL).cast(pl.Float64),
        )
        .sort(schema.DATE)
        .with_columns(
            pl.col(schema.CLOSE).pct_change().alias(schema.RETURNS),
        )
    )


def load_akshare_daily(
    symbols: list[str] | None = None,
    start_date: str = "20200101",
    end_date: str = "20251231",
    adjust: str = "hfq",
) -> pl.DataFrame:
    selected = symbols or sample_symbols()
    frames = [_download_one(symbol, start_date, end_date, adjust) for symbol in selected]
    return pl.concat(frames, how="vertical").sort([schema.SYMBOL, schema.DATE])


def save_sample_daily(
    output_path: str | Path = "data/raw/akshare_a_daily_hfq_2020_2025.parquet",
    symbols: list[str] | None = None,
    start_date: str = "20200101",
    end_date: str = "20251231",
    adjust: str = "hfq",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = load_akshare_daily(symbols=symbols, start_date=start_date, end_date=end_date, adjust=adjust)
    frame.write_parquet(path)
    return path
