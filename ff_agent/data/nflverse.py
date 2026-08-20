"""nflverse ingest, 2016-2025 (+ current season when it exists).

Every loader runs through the cache (§10) and through a **column contract**.

The contract is the point. §3.3 and §3.4 say this league's edge lives in two
categories nobody else scores — sacks taken and rushing attempts — and the real
column names are *not* what you would guess:

    sacks taken      -> ``sacks_suffered``   (NOT ``sacks``; ``def_sacks`` is
                                              sacks *recorded* by a defender)
    rushing attempts -> ``carries``          (``attempts`` is PASS attempts)

If nflverse ever renames one of these, this module fails at ingest with a named
column, instead of Milestone 4 quietly producing a board that misprices every QB.
"""

from __future__ import annotations

import polars as pl
import nflreadpy as nfl

from ff_agent.config import HISTORY_SEASONS, SEASON
from ff_agent.data.cache import cached


class ColumnContractError(RuntimeError):
    """A column the scoring engine depends on is missing or renamed."""


# ─── Column contracts ────────────────────────────────────────────────────────
# Only columns §1 scoring actually consumes. Kept deliberately tight so a
# rename in an unrelated column doesn't cry wolf.

PLAYER_STATS_REQUIRED: frozenset[str] = frozenset({
    # identity
    "player_id", "player_display_name", "position", "team", "season", "week", "season_type",
    # passing (§1: 0.04/yd, 6/TD, -2/INT, -1/sack taken, 2/2pt)
    "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "passing_2pt_conversions",
    # rushing (§1: 0.1/yd, 0.05/ATTEMPT, 6/TD, 2/2pt)
    "rushing_yards", "carries", "rushing_tds", "rushing_2pt_conversions",
    # receiving (§1: 0.1/yd, 0.5/rec, 6/TD, 2/2pt)
    "receiving_yards", "receptions", "receiving_tds", "receiving_2pt_conversions",
    # misc (§1: -2 fumble lost, 6/KR TD, 6/PR TD, 6/fumble rec TD)
    "fumbles_lost_total", "special_teams_tds", "fumble_recovery_tds",
    # kicking (§1: distance-weighted FG, -1/miss flat, 1/PAT)
    "pat_made", "fg_missed",
    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
    "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
})

TEAM_STATS_REQUIRED: frozenset[str] = frozenset({
    "team", "season", "week", "season_type",
    # D/ST (§1: 1/sack, 2/INT, 2/fum rec, 2/safety, 6/TD, 2/blocked kick)
    "def_sacks", "def_interceptions", "def_fumbles", "def_safeties", "def_tds",
    "def_fg_blocks", "def_pat_blocks", "def_punt_blocks",
    # needed to build the bucketed YARDS ALLOWED table (§1, ADD-§F):
    # opponent total yards = their passing_yards + rushing_yards
    "passing_yards", "rushing_yards",
})

SCHEDULES_REQUIRED: frozenset[str] = frozenset({
    "game_id", "season", "week", "game_type",
    "home_team", "away_team", "home_score", "away_score",
    # POINTS ALLOWED buckets come from these scores; byes come from absence
    "roof", "stadium",  # dome flags matter for K (ADD-§E) and weather (§6)
})


def assert_columns(df: pl.DataFrame, name: str, required: frozenset[str]) -> pl.DataFrame:
    """Blocking check. An unmatched column is an error, never a warning."""
    missing = sorted(required - set(df.columns))
    if missing:
        near = {}
        for m in missing:
            token = m.split("_")[0]
            near[m] = sorted(c for c in df.columns if token in c)[:6]
        detail = "\n".join(f"    {m!r}  did you mean: {near[m]}" for m in missing)
        raise ColumnContractError(
            f"{name}: {len(missing)} required column(s) missing — the scoring engine "
            f"depends on these.\n{detail}\n"
            f"  This is a BLOCKING error. nflverse likely renamed a column; fix the\n"
            f"  contract in ff_agent/data/nflverse.py and re-verify §1 scoring."
        )
    return df


# ─── Loaders ─────────────────────────────────────────────────────────────────
def players(**kw) -> pl.DataFrame:
    """Master player table. Carries gsis_id AND espn_id — this is the crosswalk."""
    return cached("players", lambda: nfl.load_players(), **kw)


def teams(**kw) -> pl.DataFrame:
    return cached("teams", lambda: nfl.load_teams(), **kw)


def schedules(**kw) -> pl.DataFrame:
    """All seasons in one file — small, and byes are derived from it."""
    def fetch():
        return assert_columns(nfl.load_schedules(seasons=True), "schedules", SCHEDULES_REQUIRED)
    return cached("schedules", fetch, **kw)


def player_stats(season: int, **kw) -> pl.DataFrame:
    def fetch():
        df = nfl.load_player_stats(seasons=season, summary_level="week")
        return assert_columns(df, f"player_stats[{season}]", PLAYER_STATS_REQUIRED)
    return cached("player_stats", fetch, season=season, **kw)


def team_stats(season: int, **kw) -> pl.DataFrame:
    def fetch():
        df = nfl.load_team_stats(seasons=season, summary_level="week")
        return assert_columns(df, f"team_stats[{season}]", TEAM_STATS_REQUIRED)
    return cached("team_stats", fetch, season=season, **kw)


def pbp(season: int, **kw) -> pl.DataFrame:
    """~19 MB/season. Not needed until M3b (xFP), cached now because it's the
    big download and draft day must be offline."""
    return cached("pbp", lambda: nfl.load_pbp(seasons=season), season=season, **kw)


def snap_counts(season: int, **kw) -> pl.DataFrame:
    return cached("snap_counts", lambda: nfl.load_snap_counts(seasons=season), season=season, **kw)


def depth_charts(season: int, **kw) -> pl.DataFrame:
    return cached("depth_charts", lambda: nfl.load_depth_charts(seasons=season), season=season, **kw)


def rosters_weekly(season: int, **kw) -> pl.DataFrame:
    return cached("rosters_weekly", lambda: nfl.load_rosters_weekly(seasons=season), season=season, **kw)


def nextgen_stats(season: int, stat_type: str = "passing", **kw) -> pl.DataFrame:
    """NGS covers exactly 2016-2025 — verified. This is why history starts at 2016."""
    return cached(
        f"ngs_{stat_type}",
        lambda: nfl.load_nextgen_stats(seasons=season, stat_type=stat_type),
        season=season, **kw,
    )


# ─── Bulk ingest ─────────────────────────────────────────────────────────────
PER_SEASON_TABLES = {
    "player_stats": player_stats,
    "team_stats": team_stats,
    "pbp": pbp,
    "snap_counts": snap_counts,
    "depth_charts": depth_charts,
    "rosters_weekly": rosters_weekly,
}


def ingest_all(seasons: list[int] | None = None, include_pbp: bool = True,
               verbose: bool = True) -> pl.DataFrame:
    """Pull and cache the full history. Returns a per-table report.

    The current season is allowed to be empty or absent — in August 2026 there
    is no 2026 regular-season data yet. Historical seasons are not allowed to
    fail: a failure there is a real problem and is reported as such.
    """
    seasons = seasons or HISTORY_SEASONS
    rows: list[dict] = []

    def note(table, season, status, detail=""):
        rows.append({"table": table, "season": season, "status": status, "detail": detail})
        if verbose:
            tag = {"ok": "  ok ", "empty": " -- ", "fail": "FAIL"}[status]
            print(f"  [{tag}] {table:16s} {season if season else '':>6}  {detail}", flush=True)

    for fn, name in ((players, "players"), (teams, "teams"), (schedules, "schedules")):
        try:
            df = fn()
            note(name, None, "ok", f"{df.height:,} rows x {df.width} cols")
        except Exception as e:
            note(name, None, "fail", f"{type(e).__name__}: {e}")

    for season in seasons:
        for name, loader in PER_SEASON_TABLES.items():
            if name == "pbp" and not include_pbp:
                continue
            try:
                df = loader(season)
                note(name, season, "empty" if df.is_empty() else "ok", f"{df.height:,} rows")
            except Exception as e:
                note(name, season, "fail", f"{type(e).__name__}: {e}")
        for st in ("passing", "receiving", "rushing"):
            try:
                df = nextgen_stats(season, st)
                note(f"ngs_{st}", season, "empty" if df.is_empty() else "ok", f"{df.height:,} rows")
            except Exception as e:
                note(f"ngs_{st}", season, "fail", f"{type(e).__name__}: {e}")

    return pl.DataFrame(rows)
