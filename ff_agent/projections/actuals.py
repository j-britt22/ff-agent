"""Season outcomes under §1 scoring — the target every projection is judged against.

Built on the Milestone 2 engine, so these are points in THIS league's scoring,
not PPR or standard. That is the whole point: a rank-correlation backtest against
PPR outcomes would be measuring the wrong game.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import FREE_BYE_WEEKS, REGULAR_SEASON_WEEKS, SEASON
from ff_agent.scoring.engine import score_dst, score_players
from ff_agent.scoring.rules import load_rules


def current_rules() -> dict[str, float]:
    """The 2026 ruleset, applied to EVERY historical season.

    This is the deliberate opposite of what Milestone 2 validation does. There we
    score each season under its OWN rules, to reproduce what ESPN recorded. Here
    we recompute history into TODAY's scoring, because §7.2 step 2 asks what a
    player would be worth in the league we are actually playing — and the league
    dropped its completion and incompletion scoring after 2024, so raw historical
    points are not comparable.
    """
    return load_rules(SEASON)

LEAGUE_REGULAR_SEASON = tuple(REGULAR_SEASON_WEEKS)          # 1-14
LEAGUE_PLAYOFF_WEEKS = (15, 16, 17)
MY_ACTIVE_WEEKS = tuple(w for w in REGULAR_SEASON_WEEKS if w not in FREE_BYE_WEEKS)


def player_season_actuals(season: int) -> pl.DataFrame:
    """Per-player season totals plus the splits this league actually cares about.

    ``points`` is the full NFL regular season — the standard basis for a
    rank-correlation backtest. ``points_my_active_weeks`` drops weeks 5 and 14,
    which is what my roster is actually graded on (§2.1), and
    ``points_playoffs`` is weeks 15-17, the §2.4 objective.
    """
    df = score_players(season, rules=current_rules()).filter(pl.col("season_type") == "REG")
    return (
        df.group_by("player_id", "player_display_name", "position")
        .agg(
            pl.col("fantasy_points").sum().round(2).alias("points"),
            pl.col("fantasy_points").mean().round(3).alias("ppg"),
            pl.len().alias("games"),
            pl.col("fantasy_points")
            .filter(pl.col("week").is_in(list(MY_ACTIVE_WEEKS))).sum()
            .round(2).alias("points_my_active_weeks"),
            pl.col("fantasy_points")
            .filter(pl.col("week").is_in(list(LEAGUE_PLAYOFF_WEEKS))).sum()
            .round(2).alias("points_playoffs"),
        )
        .rename({"player_id": "canonical_id", "player_display_name": "name"})
        .with_columns(pl.lit(season).alias("season"))
        .sort("points", descending=True)
    )


def dst_season_actuals(season: int) -> pl.DataFrame:
    df = score_dst(season, rules=current_rules())
    return (
        df.group_by("team")
        .agg(
            pl.col("fantasy_points").sum().round(2).alias("points"),
            pl.col("fantasy_points").mean().round(3).alias("ppg"),
            pl.len().alias("games"),
        )
        .with_columns(
            pl.lit(season).alias("season"),
            pl.lit("DST").alias("position"),
            (pl.lit("DST_") + pl.col("team")).alias("canonical_id"),
        )
        .sort("points", descending=True)
    )


def positional_ranks(actuals: pl.DataFrame, value: str = "points") -> pl.DataFrame:
    """Rank within position by realised points — the 'right answer' curve."""
    return actuals.with_columns(
        pl.col(value).rank("ordinal", descending=True).over("position").alias("pos_rank")
    )
