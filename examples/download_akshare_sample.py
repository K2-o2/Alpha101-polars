"""Download the 10-stock A-share sample universe from AkShare."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha101.data import save_sample_daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/raw/akshare_a_daily_hfq_2020_2025.parquet",
        help="Output parquet path.",
    )
    parser.add_argument("--start-date", default="20200101", help="Start date in YYYYMMDD format.")
    parser.add_argument("--end-date", default="20251231", help="End date in YYYYMMDD format.")
    parser.add_argument("--adjust", default="hfq", choices=["", "qfq", "hfq"], help="AkShare adjust mode.")
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Stock symbol such as sh600519. Repeat to download a custom subset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = save_sample_daily(
        output_path=args.output,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=args.adjust,
    )
    print(f"Saved sample data to {path}")
