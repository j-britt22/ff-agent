"""Assembled projections — everything Milestone 4's board will consume.

§7.2 steps 5 and 6: attach an uncertainty band per player, and carry the weeks
15-17 matchup strength and the week 5/14 bye flag as SEPARATE fields rather than
folding them into the point projection. Keeping them separate matters — §2.1's
bye adjustment and §2.4's playoff-ceiling preference are decisions the board
makes, not facts about the player.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import FREE_BYE_WEEKS, SEASON
from ff_agent.data import byes as byes_mod
from ff_agent.data import nflverse as nv
from ff_agent.projections import actuals as A
from ff_agent.projections import calibration as CAL
from ff_agent.projections import consensus as C
from ff_agent.projections import model as M
from ff_agent.projections.backtest import DEFAULT_WEIGHT

PLAYOFF_WEEKS = (15, 16, 17)


def playoff_schedule_strength(season: int) -> pl.DataFrame:
    """Per-team difficulty of the weeks 15-17 slate (§2.4).

    Opponent quality is proxied by the points that opponent ALLOWED per game in
    the prior season. Crude but directionally sound, and deliberately kept as a
    separate field so Milestone 4 can decide how much it is worth rather than
    having it silently baked into a projection.
    """
    sched = nv.schedules(offline=True).filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
        & pl.col("week").is_in(list(PLAYOFF_WEEKS))
    )
    pairs = pl.concat([
        sched.select("week", pl.col("home_team").alias("team"),
                     pl.col("away_team").alias("opp")),
        sched.select("week", pl.col("away_team").alias("team"),
                     pl.col("home_team").alias("opp")),
    ])
    prior = nv.schedules(offline=True).filter(
        (pl.col("season") == season - 1) & (pl.col("game_type") == "REG")
    )
    allowed = pl.concat([
        prior.select(pl.col("home_team").alias("opp"),
                     pl.col("away_score").alias("pa")),
        prior.select(pl.col("away_team").alias("opp"),
                     pl.col("home_score").alias("pa")),
    ]).group_by("opp").agg(pl.col("pa").mean().alias("opp_pts_allowed_pg"))

    return (
        pairs.join(allowed, on="opp", how="left")
        .group_by("team")
        .agg(
            pl.col("opp_pts_allowed_pg").mean().round(2).alias("playoff_sos"),
            pl.col("opp").sort().alias("playoff_opponents"),
        )
        .sort("playoff_sos", descending=True)
    )


def build(season: int = SEASON, weight: float = DEFAULT_WEIGHT) -> pl.DataFrame:
    """The projection table Milestone 4 turns into a board."""
    ecr = C.ecr(season).filter(pl.col("canonical_id").is_not_null())
    curve = CAL.smooth_curve(
        CAL.rank_points_curve(seasons=[s for s in range(2016, season)])
    )
    cons = CAL.consensus_to_points(ecr, curve)
    proj = M.project(season)
    blended = M.blend(cons, proj, weight=weight)

    byes = byes_mod.bye_weeks(season).select(
        "team", "bye_week", pl.col("free_bye").alias("free_bye_week_5_or_14")
    )
    sos = playoff_schedule_strength(season).select("team", "playoff_sos")

    out = (
        blended.join(byes, on="team", how="left")
        .join(sos, on="team", how="left")
        .with_columns(
            # §7.2 step 5 — uncertainty. ECR's own dispersion, converted to the
            # points scale via the calibrated curve's spread at that rank.
            (pl.col("consensus_points_sd").fill_null(0.0)).alias("points_sd"),
            (pl.col("worst") - pl.col("best")).alias("ecr_rank_spread"),
            pl.lit(sorted(FREE_BYE_WEEKS)).alias("my_bye_weeks"),
            pl.lit(season).alias("season"),
        )
        .with_columns(
            (pl.col("blended_points") - pl.col("points_sd")).round(2).alias("floor_points"),
            (pl.col("blended_points") + pl.col("points_sd")).round(2).alias("ceiling_points"),
        )
        .sort("blended_points", descending=True, nulls_last=True)
    )
    return out.select(
        "season", "canonical_id", "name", "position", "team",
        "ecr", "pos_rank", "consensus_points", "model_points", "blended_points",
        "points_sd", "floor_points", "ceiling_points", "ecr_rank_spread",
        "bye_week", "free_bye_week_5_or_14", "playoff_sos",
    )
