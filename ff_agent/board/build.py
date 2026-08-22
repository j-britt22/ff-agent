"""board.json (§4, §7.3, §7.4) — slot-independent, everything the draft needs.

VOR is computed from PROJECTED points, never realised ones. Milestone 3 found the
realised "elite" figure is upward-biased because the top scorer is a maximum over
players and so banks luck; replacement levels are rank-based averages and stay
robust. Using realised elites would import that bias straight into round one.
"""

from __future__ import annotations

import json

import polars as pl

from ff_agent.config import ARTIFACTS_DIR, FREE_BYE_WEEKS, SEASON
from ff_agent.board import replacement as R
from ff_agent.board import tiers as T
from ff_agent.projections import board_inputs as BI
from ff_agent.projections import calibration as CAL
from ff_agent.projections import xfp as X

GAMES = 17
"""Weeks in an NFL regular season — the divisor turning season VOR into the
one-week value of a free bye."""

SOE_PERSISTENCE = 0.434
"""Measured year-over-year carry-forward of sacks-over-expected (M3b).

This number is why the sack term is an ADJUSTMENT rather than a warning label.
Consensus does not price sacks at all — §3.3 says effectively no other league
scores them — and our own opportunity model DROPPED sack_rate for falling below
the 0.45 stability cutoff. So without an explicit term the edge §3.3 calls
"cheap and durable" is left entirely on the table.

A quarterback who took N sacks more than expected last season is projected to
take ``0.434 x N`` more than expected again, and each one costs exactly 1 point
under §1."""

SACK_RISK_THRESHOLD = 3.0
"""Points of expected sack cost above which the board raises a visible flag."""

TD_REGRESSION_THRESHOLD = 2.5
"""Touchdowns over expected. Applied to WR/TE ONLY — M3b found xTD beats actual
TD for receivers but NOT for RB/QB, where goal-line role is a durable team
assignment. Fading a back for scoring touchdowns would fade the thing you want."""


def sack_adjustment(prior_soe: pl.Expr, position: pl.Expr) -> pl.Expr:
    """§3.3 priced in points, for QBs only.

    Each sack is -1 under §1, and sacks-over-expected persists at 0.434, so the
    expected cost is ``-0.434 x prior_SOE``. A QB who took 14 more sacks than
    expected is projected to give back roughly six points relative to a neutral
    quarterback — before anything else about him is considered.
    """
    return (
        pl.when((position == "QB") & prior_soe.is_not_null())
        .then(-prior_soe * SOE_PERSISTENCE)
        .otherwise(0.0)
    )


def bye_adjustment(vor: pl.Expr, free_bye: pl.Expr) -> pl.Expr:
    """§2.1, priced in points rather than hand-tuned.

    A normal player's NFL bye costs one week of (player - replacement), because
    you must start a replacement that week. If the bye lands in week 5 or 14 it
    costs nothing, so the free bye is worth exactly that avoided loss: one week
    of VOR.

    A free bye is never a penalty: for a player below replacement the value is
    zero, not negative, because you would be starting the replacement anyway.
    """
    return pl.when(free_bye).then((vor / GAMES).clip(lower_bound=0.0)).otherwise(0.0)


def build(
    season: int = SEASON,
    prior_season: int | None = None,
    starter_slots: dict[str, int] | None = None,
    n_teams: int | None = None,
) -> pl.DataFrame:
    """Build the draft board.

    ``starter_slots``/``n_teams`` exist so another league can be boarded: VOR is
    measured against the STARTER replacement, so the league's shape decides
    where replacement sits. Defaulting them reproduces §1 exactly.
    """
    from ff_agent.config import N_TEAMS as _DEFAULT_TEAMS

    n_teams = _DEFAULT_TEAMS if n_teams is None else n_teams
    proj = BI.build(season)
    curve = CAL.smooth_curve(
        CAL.rank_points_curve(seasons=[s for s in range(2016, season)])
    )
    repl = R.replacement_table(curve, starter_slots=starter_slots, n_teams=n_teams)

    prior = prior_season or (season - 1)
    xf = X.xfp_season(prior).select(
        "canonical_id",
        pl.col("sacks_over_expected").alias("prior_sacks_over_expected"),
        pl.col("td_over_expected").alias("prior_td_over_expected"),
        pl.col("fpoe").alias("prior_fpoe"),
    )

    df = (
        proj.join(repl, on="position", how="left")
        .join(xf, on="canonical_id", how="left")
        .with_columns(
            # players with no NFL team (free agents) have no bye and no free bye
            pl.col("free_bye_week_5_or_14").fill_null(False),
        )
        .with_columns(
            (pl.col("blended_points") - pl.col("replacement_points"))
            .round(2).alias("vor_raw")
        )
        .with_columns(
            bye_adjustment(pl.col("vor_raw"), pl.col("free_bye_week_5_or_14"))
            .round(2).alias("bye_adjustment"),
            sack_adjustment(pl.col("prior_sacks_over_expected"), pl.col("position"))
            .round(2).alias("sack_adjustment"),
        )
        .with_columns(
            (pl.col("vor_raw") + pl.col("bye_adjustment")
             + pl.col("sack_adjustment")).round(2).alias("vor")
        )
        .with_columns(
            # §3.3 / ADD-§B.1 — a persistent trait, not noise
            (-pl.col("sack_adjustment") > SACK_RISK_THRESHOLD)
            .fill_null(False).alias("flag_sack_risk"),
            # ADD-§A.1, receivers only
            (pl.col("position").is_in(["WR", "TE"])
             & (pl.col("prior_td_over_expected") > TD_REGRESSION_THRESHOLD))
            .fill_null(False).alias("flag_td_regression"),
            pl.lit(sorted(FREE_BYE_WEEKS)).alias("my_bye_weeks"),
        )
    )

    df = T.assign_tiers(df, value="vor")
    # §0.2 checked BEFORE the join, which is the only place it does any good.
    # tier_stability returns one row per input ROW, so it is the right-hand side
    # here and inherits whatever duplication it was handed: a player arriving
    # twice leaves this join four times, not twice. Asserting the left side
    # first makes the join incapable of multiplying rows at all, instead of
    # catching the squared damage downstream in write_board.
    assert_one_row_per_player(df, stage="the board's tier join")
    stab = T.tier_stability(df, value="vor")
    df = df.join(stab, on="canonical_id", how="left")

    return (
        df.sort("vor", descending=True, nulls_last=True)
        .with_columns(
            pl.col("vor").rank("ordinal", descending=True).cast(pl.Int32)
            .alias("overall_rank"),
            # cast BEFORE subtracting: u32 arithmetic wraps on negatives
            (pl.col("ecr").rank("ordinal").cast(pl.Int32)
             - pl.col("vor").rank("ordinal", descending=True).cast(pl.Int32))
            .alias("rank_vs_ecr"),
        )
        .select(
            "overall_rank", "canonical_id", "name", "position", "team",
            "vor", "vor_raw", "bye_adjustment", "sack_adjustment", "blended_points",
            "consensus_points", "model_points", "replacement_points",
            "tier", "tier_size", "players_remaining_in_tier", "tier_stability",
            "ecr", "rank_vs_ecr", "points_sd", "floor_points", "ceiling_points",
            "bye_week", "free_bye_week_5_or_14", "playoff_sos",
            "flag_sack_risk", "flag_td_regression",
            "prior_sacks_over_expected", "prior_td_over_expected", "prior_fpoe",
            "season",
        )
    )


def assert_one_row_per_player(board: pl.DataFrame, stage: str = "the board") -> None:
    """§0.2, applied to the board itself.

    "Every rostered and drafted player must resolve to exactly one canonical ID.
    Assert it." Resolution is only half of that — a join that fans out breaks it
    just as thoroughly, and silently. A duplicated row means the draft simulator
    can hand the same real person to two different fantasy teams, because it
    marks POOL INDICES taken, not people.

    ``stage`` names where the check ran, because the same failure means
    different things at different points: before the tier join it is upstream
    duplication, after it the join itself multiplied rows.
    """
    dupes = board.group_by("canonical_id").len().filter(pl.col("len") > 1)
    if dupes.height:
        cols = [c for c in ("canonical_id", "name", "position", "team",
                            "blended_points", "vor") if c in board.columns]
        detail = (
            board.filter(pl.col("canonical_id").is_in(dupes["canonical_id"].implode()))
            .select(cols)
            .sort("canonical_id")
        )
        raise ValueError(
            f"{dupes.height} canonical_id(s) appear more than once in {stage} "
            f"({int(dupes['len'].sum())} rows). A join fanned out; find it rather "
            f"than deduplicating here.\n{detail}"
        )


def write_board(season: int = SEASON) -> tuple[pl.DataFrame, str]:
    b = build(season)
    assert_one_row_per_player(b)
    path = ARTIFACTS_DIR / "board.json"
    payload = {
        "season": season,
        "generated_for": "First Down Syndrome (Wildcats League)",
        "replacement_levels": R.replacement_table(
            CAL.smooth_curve(CAL.rank_points_curve(
                seasons=[s for s in range(2016, season)]))
        ).to_dicts(),
        "flex_share_measured": R.flex_share(),
        "free_bye_weeks": sorted(FREE_BYE_WEEKS),
        "players": b.to_dicts(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    b.write_parquet(ARTIFACTS_DIR / "board.parquet")
    return b, str(path)
