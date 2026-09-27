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

MIN_EXPECTED_POINTS = 0.0
"""Floor on the curve, applied where the curve is built rather than where a
board is printed. A negative *expected* season is impossible under §1: every
negative category (INT −2, sack taken −1, fumble lost −2, missed FG −1) is only
reachable by a player who is simultaneously accruing 0.05/carry, 0.1/yd and
0.5/reception. A player expected to do nothing is expected to score 0, not less.

The fitted curve goes negative anyway, and measurement on 2016-2025 says exactly
where — past the deepest rank that all ten seasons populate, at every position:

    QB   all 10 seasons reach rank 71,  first negative 73   (min −4.73 at 83)
    RB   all 10 seasons reach rank 143, first negative 156  (min −0.50 at 165)
    WR   all 10 seasons reach rank 210, first negative 222  (min −2.60 at 256)
    TE   all 10 seasons reach rank 123, first negative 129  (min −0.07 at 137)

So the negative tail is a selection effect, not signal. A season only produces a
rank that deep if some marginal player recorded a stat line at all, and the
players who get there are the ones whose only counted event was a lost fumble —
2025's RB150 and RB151 are Nyheim Hines at −0.10 and Travis Homer at −0.15 over
4 and 8 games. Averaging one or two such seasons estimates that fluke, not an
expectation, and reading it as a forecast says a 2026 RB will COST his manager
a third of a point when the honest statement is that he will score about none.

K and D/ST never go negative — their pools are only 32-47 deep and are fully
supported throughout — so this touches skill positions only.

Deliberately left visible: ``expected_points`` keeps the unfloored weighted
mean, so the tail is still inspectable and this finding is still checkable in
the data. Only ``expected_points_smooth``, the column every consumer reads, is
floored."""


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
    """Rolling mean within position — single ranks are noisy, the shape is not.

    Floored at ``MIN_EXPECTED_POINTS``; see that constant for what the deep tail
    was doing and why the floor belongs here rather than at the end of the
    pipeline. Smoothing alone does not fix it — a 5-rank window centred in a
    region where every neighbour is a one-season fluke just averages flukes.
    """
    return curve.sort(["position", "pos_rank"]).with_columns(
        pl.col("expected_points")
        .rolling_mean(window_size=window, min_samples=1, center=True)
        .over("position")
        .clip(lower_bound=MIN_EXPECTED_POINTS)
        .round(2)
        .alias("expected_points_smooth")
    )


def curve_tail(curve: pl.DataFrame) -> pl.DataFrame:
    """The value used for ranks DEEPER than the curve was fitted on.

    Split out and named because it is an extrapolation, and extrapolating off
    the end of a fitted curve is how a projection acquires a number that nothing
    measured. Two properties, both deliberate:

    * **Flat, not extended.** Hold the last fitted value rather than continuing
      the local slope. The slope at the end of the curve is fitted on one or two
      seasons (see ``MIN_EXPECTED_POINTS``), so extending it would project ever
      more negative points for players expected to score none.
    * **Floored.** The last fitted rank is the single thinnest point on the whole
      curve, so before the floor existed this rule propagated the noisiest value
      available to every player past the end. That is how RB166 (Ja'Quinden
      Jackson, ECR 386) arrived at −0.50 in the 2026 board.
    """
    return (
        curve.sort("pos_rank").group_by("position")
        .agg(
            pl.col("expected_points_smooth").last().alias("_tail"),
            pl.col("pos_rank").last().alias("_tail_rank"),
        )
        .with_columns(pl.col("_tail").clip(lower_bound=MIN_EXPECTED_POINTS))
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
    # Ranks beyond the fitted curve fall to its tail (see ``curve_tail``). The
    # fall-through is flagged rather than silent: a projection that came from
    # extrapolation should be identifiable as one.
    return (
        out.join(curve_tail(curve), on="position", how="left")
        .with_columns(
            (pl.col("consensus_points").is_null() & pl.col("_tail").is_not_null())
            .alias("consensus_points_extrapolated"),
            pl.coalesce("consensus_points", "_tail").alias("consensus_points"),
        )
        .drop("_tail", "_tail_rank")
        .sort("ecr")
    )
