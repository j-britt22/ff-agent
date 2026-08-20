"""D/ST bucket inputs: points allowed and yards allowed.

ADD-§F calls the yardage buckets out as unique to this league, and §7.1 names the
two bucketed D/ST tables as the place a generic scoring function silently goes
wrong. Both inputs are derived, and both derivations are counter-intuitive.

**Yards allowed = opponent ``passing_yards + sack_yards_lost + rushing_yards``.**
``sack_yards_lost`` is stored NEGATIVE in nflverse, so it adds. Gross
passing+rushing is wrong. Verified exact on 119/119 team-weeks of 2025.

**Points allowed excludes only what the opponent scored against MY OFFENSE on a
scrimmage play** — pick-sixes, strip-sack fumble returns, and safeties conceded
by the offense. Points scored against my SPECIAL TEAMS still count as allowed:
kickoff-recovery TDs, blocked-field-goal returns, and a punter tackled in his own
end zone are all charged to the fantasy D/ST, because that unit is defense *and*
special teams. Using the opponent's final score is wrong; excluding every
non-offensive TD is also wrong. Each is off on 8 of 119 team-weeks, on different
rows.
"""

from __future__ import annotations

import polars as pl

from ff_agent.data import nflverse as nv

SCRIMMAGE_PLAY_TYPES = frozenset({"pass", "run", "qb_kneel", "qb_spike", "no_play"})
"""Plays run by the offense. A TD or safety conceded here is the offense's doing,
so ESPN does not charge it to the D/ST."""

SPECIAL_TEAMS_PLAY_TYPES = frozenset({"punt", "kickoff", "field_goal", "extra_point"})

ST_FORMATION_RE = r"(?i)\((?:Punt|Field Goal) formation\)"
"""A punt or FG snap that turns into a run/pass is typed ``run``/``pass`` by
nflverse with every special-teams flag at 0 — the formation tag in ``desc`` is
the only signal. 17 such plays conceded points across 2016-2025, e.g. PHI's
punter tackled in his own end zone in week 4 2025, which ESPN charges to the
D/ST. Narrow and explicit; this is not fuzzy matching."""


def _is_offensive_play() -> pl.Expr:
    """True for a genuine scrimmage snap by the offense."""
    return (
        pl.col("play_type").is_in(list(SCRIMMAGE_PLAY_TYPES))
        & ~pl.col("desc").fill_null("").str.contains(ST_FORMATION_RE)
    )


def offense_conceded(season: int, pbp: pl.DataFrame | None = None) -> pl.DataFrame:
    """Points a team's OFFENSE handed to the opponent, per team-week.

    These are subtracted from the opponent's final score to get points allowed.
    """
    p = nv.pbp(season, offline=True) if pbp is None else pbp
    p = p.filter(pl.col("season_type") == "REG")

    # TD scored by the team NOT in possession, on a scrimmage play == defensive return
    def_td = (
        p.filter(
            pl.col("td_team").is_not_null()
            & (pl.col("td_team") != pl.col("posteam"))
            & _is_offensive_play()
        )
        .group_by(["week", "posteam"])
        .len()
        .rename({"posteam": "team", "len": "def_tds_conceded"})
    )
    # safety on a scrimmage play == the offense was tackled/penalised in its end zone
    saf = (
        p.filter((pl.col("safety") == 1) & _is_offensive_play())
        .group_by(["week", "posteam"])
        .len()
        .rename({"posteam": "team", "len": "safeties_conceded"})
    )
    weeks = p.select("week").unique()
    teams = p.select(pl.col("posteam").alias("team")).drop_nulls().unique()
    grid = weeks.join(teams, how="cross")
    return (
        grid.join(def_td, on=["week", "team"], how="left")
        .join(saf, on=["week", "team"], how="left")
        .with_columns(
            pl.col("def_tds_conceded").fill_null(0),
            pl.col("safeties_conceded").fill_null(0),
        )
        .with_columns(
            (6 * pl.col("def_tds_conceded") + 2 * pl.col("safeties_conceded"))
            .alias("offense_conceded_points")
        )
        .with_columns(pl.lit(season).alias("season"))
    )


def dst_inputs(season: int) -> pl.DataFrame:
    """One row per team-week: points_allowed and yards_allowed for the buckets."""
    sched = nv.schedules(offline=True).filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    if sched.is_empty():
        raise ValueError(f"No REG schedule for {season}")

    home = sched.select(
        "week", pl.col("home_team").alias("team"), pl.col("away_team").alias("opp"),
        pl.col("away_score").alias("opp_score"),
    )
    away = sched.select(
        "week", pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"),
        pl.col("home_score").alias("opp_score"),
    )
    pairs = pl.concat([home, away])

    team_stats = (
        nv.team_stats(season, offline=True)
        .filter(pl.col("season_type") == "REG")
        .select(
            "week", pl.col("team").alias("opp"),
            "passing_yards", "rushing_yards", "sack_yards_lost",
        )
    )
    # NB: join on TEAM, not opp. My points-allowed is reduced by what MY OWN
    # offense handed the opponent (their score includes those points, but my
    # D/ST did not concede them).
    conceded = offense_conceded(season).select(
        "week", "team", "offense_conceded_points"
    )

    return (
        pairs.join(team_stats, on=["week", "opp"], how="left")
        .join(conceded, on=["week", "team"], how="left")
        .with_columns(pl.col("offense_conceded_points").fill_null(0))
        .with_columns(
            # sack_yards_lost is NEGATIVE, so it adds
            (pl.col("passing_yards") + pl.col("sack_yards_lost") + pl.col("rushing_yards"))
            .alias("yards_allowed"),
            (pl.col("opp_score") - pl.col("offense_conceded_points"))
            .alias("points_allowed"),
        )
        .with_columns(pl.lit(season).alias("season"))
        .select("season", "week", "team", "opp", "points_allowed", "yards_allowed",
                "opp_score", "offense_conceded_points")
        .sort(["week", "team"])
    )


def dst_sacks(season: int, pbp: pl.DataFrame | None = None) -> pl.DataFrame:
    """Team sacks per week, counted from play-by-play.

    ``team_stats.def_sacks`` aggregates PLAYER credits, which are fractional for
    shared sacks and occasionally leave a play uncredited — it misses one sack in
    2025 (LV week 6: six sack plays, 5.0 credited). Counting sack PLAYS matches
    ESPN on 128/128 team-weeks.
    """
    p = nv.pbp(season, offline=True) if pbp is None else pbp
    return (
        p.filter((pl.col("season_type") == "REG") & (pl.col("sack") == 1))
        .group_by(["week", "defteam"])
        .len()
        .rename({"defteam": "team", "len": "sacks"})
        .with_columns(pl.col("sacks").cast(pl.Float64))
    )
