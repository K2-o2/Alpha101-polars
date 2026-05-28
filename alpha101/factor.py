"""Factor definitions that may need staged Polars evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class AlphaFactor:
    name: str
    stages: tuple[tuple[pl.Expr, ...], ...]
    expr: pl.Expr
    temporary_columns: tuple[str, ...] = ()

    def expressions(self) -> list[pl.Expr]:
        return [self.expr.alias(self.name)]
