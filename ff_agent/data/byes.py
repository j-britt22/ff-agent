"""NFL bye weeks, and the §2.1 free-bye flag.

My fantasy byes are weeks 5 and 14, so **an NFL player whose bye falls in week 5
or 14 costs me nothing**. Nobody's rankings price this, because it is a property
of my schedule, not the player's.

This module only produces the *flag*. Converting it into points — "roughly one
start's worth of expected points over a replacement-level fill-in" — is
Milestone 4's job, and deliberately not done here.

A bye is derived as absence: the regular-season weeks in which a team has no
game. That is more robust than trusting any bye-week column.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import FREE_BYE_WEEKS, REGULAR_SEASON_WEEKS
from ff_agent.data import nflverse as nv
from ff_agent.data.crosswalk import CURRENT_TEAMS, normalize_team


class ByeWeekError(RuntimeError):
    """Bye derivation produced something structurally impossible."""


def assert_canonical_teams(df: pl.DataFrame, season: int, column: str = "team") -> None:
    """§0.2 applied to team entities — one spelling, asserted, never inferred.

    §0.2 says every player must resolve to exactly one canonical id and that a
    miss is a blocking error, not a warning. A team abbreviation is the D/ST
    corollary of the same rule, and it is the join key that §2.1's bye term and
    §2.4's playoff-SOS term both hang off. So it gets the same treatment.
    """
    unknown = sorted(set(df[column].to_list()) - set(CURRENT_TEAMS))
    if unknown:
        raise ByeWeekError(
            f"Season {season}: team abbreviation(s) {unknown} in column "
            f"'{column}' are not canonical (see crosswalk.CURRENT_TEAMS). Any "
            f"join against ESPN-derived rows drops them SILENTLY, as a null. "
            f"Add the alias to crosswalk.ESPN_TEAM_ALIASES; do not special-case "
            f"it here."
        )


def team_weeks_played(season: int, schedules: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per (team, week) the team actually plays in the regular season.

    Abbreviations are canonicalised here, and this is the only place in the
    module where a team name enters from nflverse — every other function derives
    from this one, so normalising at this single boundary is what makes the
    module speak one spelling.

    **Why it has to happen at all.** nflverse spells the Rams ``LA``; ESPN, the
    crosswalk, and therefore every projection and board row spell them ``LAR``.
    Left raw the mismatch is invisible: a left join yields null, the board fills
    the §2.1 flag with ``False``, and eighteen Rams — Puka Nacua at overall rank
    13 among them — carry no bye week and no playoff SOS. It is only
    *accidentally* harmless in a season like 2026 where LA's bye (week 11) misses
    weeks 5 and 14 anyway. Caught three separate times downstream (M2 scoring,
    M7 draft pool, M4 board) before being fixed at the source.

    Historical relocations fold in on the same path — 2016 ``SD`` becomes
    ``LAC``, 2016-19 ``OAK`` becomes ``LV`` — which keeps backtest seasons
    joinable against the same vocabulary. Verified 2016-2026: exactly 32 distinct
    teams every season, no collisions.
    """
    sched = nv.schedules() if schedules is None else schedules
    reg = sched.filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    ).select("week", "home_team", "away_team")
    home = reg.select(pl.col("home_team").alias("team"), "week")
    away = reg.select(pl.col("away_team").alias("team"), "week")
    return (
        pl.concat([home, away])
        .with_columns(
            pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8)
        )
        .unique()
        .sort(["team", "week"])
    )


BYE_WINDOW = frozenset(range(4, 15))
"""Weeks in which a scheduled bye can fall. Verified 2016-2026: every real bye
lands in 4-14. Used to tell a bye apart from a CANCELLED game."""


def no_game_weeks(season: int, schedules: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every (team, week) in the regular season with no game on the schedule.

    This is a superset of byes: a cancelled game also leaves a hole.
    """
    sched = nv.schedules() if schedules is None else schedules
    reg = sched.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    if reg.is_empty():
        raise ByeWeekError(
            f"No REG games found for season {season}. The schedule for that "
            f"season may not be published yet."
        )
    played = team_weeks_played(season, sched)
    max_week = int(reg["week"].max())
    grid = (
        pl.DataFrame({"team": played["team"].unique().sort()})
        .join(pl.DataFrame({"week": list(range(1, max_week + 1))}), how="cross")
    )
    return (
        grid.join(played.with_columns(pl.lit(True).alias("played")),
                  on=["team", "week"], how="left")
        .filter(pl.col("played").is_null())
        .select("team", "week")
        .sort(["team", "week"])
    )


def _derive(season: int, schedules: pl.DataFrame | None = None
            ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split every no-game hole into (the bye, everything else).

    A team's bye is the week it has no game. Two real historical cases make the
    naive version wrong, and both are in our 2016-2025 window:

      * **2022 BUF/CIN** — the week 17 Bills-Bengals game was cancelled after
        Damar Hamlin's cardiac arrest and never made up. Both teams therefore
        have TWO holes; the real bye is the one inside the bye window, and both
        teams played 16 games instead of 17.
      * **2017 MIA/TB** — their week 1 game was postponed for Hurricane Irma and
        replayed in week 11, which had been their scheduled bye. They have ONE
        hole and it sits in week 1, so week 1 genuinely is their functional bye.

    Hence: one hole is always the bye, wherever it falls. Only when there are
    several do we use the bye window to choose, and the leftovers are reported
    as anomalies rather than discarded silently.
    """
    gaps = no_game_weeks(season, schedules)
    bye_rows, anomaly_rows = [], []

    for team, grp in gaps.group_by("team"):
        team = team[0] if isinstance(team, tuple) else team
        weeks = sorted(grp["week"].to_list())
        if len(weeks) == 1:
            chosen = weeks[0]
        else:
            in_window = [w for w in weeks if w in BYE_WINDOW]
            if len(in_window) != 1:
                raise ByeWeekError(
                    f"Season {season}, {team}: cannot identify the bye among "
                    f"no-game weeks {weeks} (weeks inside the bye window "
                    f"{sorted(BYE_WINDOW)}: {in_window}).\n"
                    f"  Resolve by hand before trusting the §2.1 free-bye term."
                )
            chosen = in_window[0]
        bye_rows.append({"season": season, "team": team, "bye_week": chosen})
        for w in weeks:
            if w != chosen:
                anomaly_rows.append({
                    "season": season, "team": team, "week": w,
                    "reason": "no game scheduled, and not this team's bye "
                              "(cancelled, postponed, or rescheduled)",
                })

    byes_df = (
        pl.DataFrame(bye_rows)
        .with_columns(pl.col("bye_week").is_in(list(FREE_BYE_WEEKS)).alias("free_bye"))
        .select("season", "team", "bye_week", "free_bye")
        .sort("team")
    )
    anomalies_df = (
        pl.DataFrame(anomaly_rows, schema={"season": pl.Int64, "team": pl.Utf8,
                                           "week": pl.Int64, "reason": pl.Utf8})
        if anomaly_rows else
        pl.DataFrame(schema={"season": pl.Int64, "team": pl.Utf8,
                             "week": pl.Int64, "reason": pl.Utf8})
    ).sort(["team", "week"])
    return byes_df, anomalies_df


def schedule_anomalies(season: int, schedules: pl.DataFrame | None = None) -> pl.DataFrame:
    """No-game team-weeks that are NOT the team's bye. See _derive()."""
    return _derive(season, schedules)[1]


def bye_weeks(season: int, schedules: pl.DataFrame | None = None) -> pl.DataFrame:
    """team -> bye_week, free_bye (§2.1). Derived from absence. See _derive()."""
    byes_df = _derive(season, schedules)[0]
    if byes_df.height != 32:
        raise ByeWeekError(
            f"Season {season}: derived byes for {byes_df.height} teams, expected 32."
        )
    assert_canonical_teams(byes_df, season)
    return byes_df


def bye_summary(season: int) -> pl.DataFrame:
    """Teams per bye week, with the free-bye weeks marked."""
    b = bye_weeks(season)
    return (
        b.group_by("bye_week")
        .agg(pl.len().alias("n_teams"), pl.col("team").sort().alias("teams"))
        .with_columns(pl.col("bye_week").is_in(list(FREE_BYE_WEEKS)).alias("free_bye"))
        .sort("bye_week")
    )


def free_bye_teams(season: int) -> list[str]:
    """Teams whose bye lands in week 5 or 14 — every player on them is worth more."""
    b = bye_weeks(season)
    return b.filter(pl.col("free_bye"))["team"].sort().to_list()
