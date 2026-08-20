"""The league schedule, pulled live rather than reconstructed.

ESPN encodes a bye as a team playing ITSELF, which is the kind of detail that
silently produces a 14-game season for someone who plays 12.

§2.5 is real and now quantified for 2026: with 9 teams over 14 weeks there are 14
bye slots, so **five teams play 12 games and four play 13**. First Down Syndrome
is in the 12-game group. On raw wins that is a structural disadvantage; on win
percentage it disappears — which is exactly why the simulator scores both.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import SEASON, normalize_team_name
from ff_agent.data.cache import cached


def league_schedule(season: int = SEASON, **kw) -> pl.DataFrame:
    """One row per team-week. ``opponent`` is null on a bye."""

    def fetch() -> pl.DataFrame:
        from ff_agent.data.espn import get_league

        lg = get_league(season)
        reg = int(getattr(lg.settings, "reg_season_count", 14))
        rows = []
        for t in lg.teams:
            me = normalize_team_name(t.team_name)
            for wk, opp in enumerate(getattr(t, "schedule", []) or [], start=1):
                if wk > reg:
                    continue
                other = normalize_team_name(getattr(opp, "team_name", None))
                rows.append({
                    "season": season, "week": wk, "team": me,
                    "opponent": None if other == me else other,
                    "team_id": getattr(t, "team_id", None),
                })
        if not rows:
            raise RuntimeError(f"no schedule for {season}")
        return pl.DataFrame(rows)

    return cached("league_schedule", fetch, season=season, source="espn", **kw)


def games_played(season: int = SEASON) -> pl.DataFrame:
    """§2.5's unequal games, made explicit."""
    s = league_schedule(season)
    return (
        s.group_by("team")
        .agg(
            pl.col("opponent").is_not_null().sum().alias("games"),
            pl.col("week").filter(pl.col("opponent").is_null()).sort().alias("bye_weeks"),
        )
        .with_columns(pl.col("bye_weeks").list.len().alias("n_byes"))
        .sort(["games", "team"])
    )


def matchups(season: int = SEASON) -> pl.DataFrame:
    """Deduplicated head-to-head pairs, one row per game."""
    s = league_schedule(season).filter(pl.col("opponent").is_not_null())
    return (
        s.with_columns(
            pl.min_horizontal("team", "opponent").alias("a"),
            pl.max_horizontal("team", "opponent").alias("b"),
        )
        .unique(subset=["week", "a", "b"])
        .select("season", "week", "a", "b")
        .sort(["week", "a"])
    )
