"""Column-name conventions used by alpha101-polars."""

DATE = "date"
SYMBOL = "symbol"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"
VWAP = "vwap"
RETURNS = "returns"

PRICE_COLUMNS = (OPEN, HIGH, LOW, CLOSE, VWAP)
REQUIRED_COLUMNS = (DATE, SYMBOL, OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP, RETURNS)
