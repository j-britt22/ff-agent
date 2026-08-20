"""The two bucketed D/ST tables (§1).

Continuous points-allowed and yards-allowed are OFF in this league, so only the
bucket matters. Buckets carrying a ``None`` rule score **zero** — ESPN omits
zero-valued rules from its payload entirely, and §1 records all three.
"""

from __future__ import annotations

import polars as pl

# (low, high, rule_abbr | None). Bounds inclusive; None = open-ended.
POINTS_ALLOWED_BUCKETS: tuple[tuple[int | None, int | None, str | None], ...] = (
    (None, 0, "PA0"),      # shutout
    (1, 6, "PA1"),
    (7, 13, "PA7"),
    (14, 17, "PA14"),
    (18, 21, None),        # 0 — verified absent from ESPN's rules
    (22, 27, None),        # 0 — verified: TB allowed exactly 27 in wk3 2025, scored 0
    (28, 34, "PA28"),
    (35, 45, "PA35"),
    (46, None, "PA46"),
)

YARDS_ALLOWED_BUCKETS: tuple[tuple[int | None, int | None, str | None], ...] = (
    (None, 99, "YA100"),
    (100, 199, "YA199"),
    (200, 299, "YA299"),
    (300, 349, None),      # 0 — verified absent
    (350, 399, "YA399"),
    (400, 449, "YA449"),
    (450, 499, "YA499"),
    (500, 549, "YA549"),
    (550, None, "YA550"),
)


def bucket_points_expr(
    value_col: str,
    buckets: tuple[tuple[int | None, int | None, str | None], ...],
    rules: dict[str, float],
) -> pl.Expr:
    """Polars expression mapping a value to its bucket's point award."""
    expr = None
    for lo, hi, abbr in buckets:
        cond = pl.lit(True)
        if lo is not None:
            cond = cond & (pl.col(value_col) >= lo)
        if hi is not None:
            cond = cond & (pl.col(value_col) <= hi)
        pts = pl.lit(0.0 if abbr is None else float(rules.get(abbr, 0.0)))
        expr = pl.when(cond).then(pts) if expr is None else expr.when(cond).then(pts)
    return expr.otherwise(pl.lit(0.0))


def bucket_label_expr(
    value_col: str,
    buckets: tuple[tuple[int | None, int | None, str | None], ...],
) -> pl.Expr:
    """Which bucket a value fell in — for diagnostics and the board."""
    expr = None
    for lo, hi, abbr in buckets:
        cond = pl.lit(True)
        if lo is not None:
            cond = cond & (pl.col(value_col) >= lo)
        if hi is not None:
            cond = cond & (pl.col(value_col) <= hi)
        lbl = pl.lit(abbr if abbr else f"ZERO_{lo}_{hi}")
        expr = pl.when(cond).then(lbl) if expr is None else expr.when(cond).then(lbl)
    return expr.otherwise(pl.lit("UNKNOWN"))
