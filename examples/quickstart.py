"""Compute the first batch of Alpha101 factors on saved sample data."""

from pathlib import Path
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha101 import compute_alphas


if __name__ == "__main__":
    bars = pl.read_parquet("data/raw/akshare_a_daily_hfq_2020_2025.parquet")
    features = compute_alphas(bars)
    features.write_parquet("data/processed/alpha101_sample_features.parquet")
    print(features.select("date", "symbol", "alpha001", "alpha002", "alpha003"))
