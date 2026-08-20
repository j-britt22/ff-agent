"""Rank -> points calibration.

ECR is a *rank*. §7.2 step 2 says to recompute the consensus baseline into this
league's scoring, and you cannot recompute a rank — so the rank has to be mapped
onto a points scale first.

The curve is fitted on realised outcomes **recomputed under §1 scoring**, which
is what makes it league-specific: the QB curve here reflects 6-point passing TDs
and -1 per sack, so QB18 sits where it actually sits in *this* league rather than
where it sits in a PPR 1-QB league.

Recency-weighted, because the passing environment moved over 2016-2025
(CLAUDE.md) and 2020 is flagged contaminated.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import CONTAMINATED_SEASONS, HISTORY_SEASONS
from ff_agent.projections import actuals as A

DEFAULT_HALFLIFE = 3.0
"""Seasons. A 3-season half-life keeps ~10 years of shape while letting the
recent scoring environment dominate the level."""

CONTAMINATED_WEIGHT = 0.5
"""2020 had no preseason, empty stadiums and mass absences. Down-weighted rather
than dropped — it is still 32 teams playing football."""


def season_weights(
    seasons: list[int], halflife: float = DEFAULT_HALFLIFE
) -> dict[int, float]:
    newest = max(seasons)
    out: dict[int, float] = {}
    for s in seasons:
        w = 0.5 ** ((newest - s) / halflife)
        if s in CONTAMINATED_SEASONS:
            w *= CONTAMINATED_WEIGHT
        out[s] = w
    return out


def rank_points_curve(
    seasons: list[int] | None = None,
    halflife: float = DEFAULT_HALFLIFE,
    value: str = "points",
) -> pl.DataFrame:
    """Expected §1 points at each positional rank.

    Read it as: "the RB who finishes 22nd at his position scores about X in this
    league." That is exactly the quantity §7.3's replacement level needs.
    """
    seasons = seasons or HISTORY_SEASONS
    w = season_weights(seasons, halflife)

    frames = []
    for s in seasons:
        a = A.player_season_actuals(s)
        d = A.dst_season_actuals(s)
        both = pl.concat([
            a.select("season", "canonical_id", "position", value),
            d.select("season", "canonical_id", "position", value),
        ])
        frames.append(
            A.positional_ranks(both, value=value)
            .with_columns(pl.lit(w[s]).alias("weight"))
        )
    stacked = pl.concat(frames)

    return (
        stacked.group_by("position", "pos_rank")
        .agg(
            ((pl.col(value) * pl.col("weight")).sum() / pl.col("weight").sum())
            .round(2).alias("expected_points"),
            pl.col(value).median().round(2).alias("median_points"),
            pl.col(value).std().round(2).alias("sd_points"),
            pl.len().alias("n_seasons"),
        )
        .sort(["position", "pos_rank"])
    )


def smooth_curve(curve: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Rolling mean within position — single ranks are noisy, the shape is not."""
    return curve.sort(["position", "pos_rank"]).with_columns(
        pl.col("expected_points")
        .rolling_mean(window_size=window, min_samples=1, center=True)
        .over("position")
        .round(2)
        .alias("expected_points_smooth")
    )


def consensus_to_points(
    ecr: pl.DataFrame, curve: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Turn an ECR snapshot into projected §1 points.

    The overall superflex rank is first converted to a rank WITHIN position,
    then read off the calibrated curve.
    """
    curve = smooth_curve(rank_points_curve()) if curve is None else curve
    if "expected_points_smooth" not in curve.columns:
        curve = smooth_curve(curve)

    ranked = ecr.with_columns(
        pl.col("ecr").rank("ordinal").over("position").cast(pl.UInt32).alias("pos_rank")
    )
    out = ranked.join(
        curve.select(
            "position",
            pl.col("pos_rank").cast(pl.UInt32),
            pl.col("expected_points_smooth").alias("consensus_points"),
            pl.col("sd_points").alias("consensus_points_sd"),
        ),
        on=["position", "pos_rank"], how="left",
    )
    # ranks beyond the fitted curve fall to the curve's tail value
    tail = (
        curve.sort("pos_rank").group_by("position")
        .agg(pl.col("expected_points_smooth").last().alias("_tail"))
    )
    return (
        out.join(tail, on="position", how="left")
        .with_columns(
            pl.coalesce("consensus_points", "_tail").alias("consensus_points")
        )
        .drop("_tail")
        .sort("ecr")
    )
