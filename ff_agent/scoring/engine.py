"""Stat line -> fantasy points under §1.

Every rule emits its own ``pts_<ABBR>`` column before they are summed. That costs
nothing and buys the thing §7.1 actually needs: when a total disagrees with
ESPN, the per-category columns say *which rule* is wrong instead of leaving you
with a number that is off by 1.3 for unknown reasons.

The two column names that matter most are the two the league scores and nobody
else does (§3.3, §3.4):
    SKD (sack taken)      <- ``sacks_suffered``
    RA  (rushing attempt) <- ``carries``
"""

from __future__ import annotations

import polars as pl

from ff_agent.data import nflverse as nv
from ff_agent.data.crosswalk import normalize_team
from ff_agent.scoring import dst as dst_mod
from ff_agent.scoring.buckets import (
    POINTS_ALLOWED_BUCKETS, YARDS_ALLOWED_BUCKETS,
    bucket_label_expr, bucket_points_expr,
)
from ff_agent.scoring.rules import load_rules

# ─── nflverse column -> ESPN rule ───────────────────────────────────────────
# A plain 1:1 stat -> rule mapping. Derived quantities are handled below.
PLAYER_STAT_MAP: dict[str, str] = {
    "passing_yards": "PY",
    "passing_tds": "PTD",
    "passing_interceptions": "INTT",
    "sacks_suffered": "SKD",            # NOT `sacks`; NOT `def_sacks`
    "passing_2pt_conversions": "2PC",
    "completions": "PC",     # 0 in 2025+; the league scored 0.25 through 2024

    "rushing_yards": "RY",
    "carries": "RA",                    # `attempts` is PASS attempts
    "rushing_tds": "RTD",
    "rushing_2pt_conversions": "2PR",
    "receiving_yards": "REY",
    "receptions": "REC",
    "receiving_tds": "RETD",
    "receiving_2pt_conversions": "2PRE",
    "fumbles_lost_total": "FUML",
    "fumble_recovery_tds": "FTD",
    "pat_made": "PAT",
    "fg_made_40_49": "FG40",
    "fg_made_50_59": "FG50",
    "fg_made_60_": "FG60",
}

PLAYER_IDENTITY = ["player_id", "player_display_name", "position", "team",
                   "season", "week", "season_type"]


def _num(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64).fill_null(0.0)


def score_players(
    season: int,
    rules: dict[str, float] | None = None,
    stats: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Score every player-week. Returns identity + pts_* columns + total."""
    r = load_rules(season) if rules is None else rules
    df = nv.player_stats(season, offline=True) if stats is None else stats

    exprs: list[pl.Expr] = []
    for col, abbr in PLAYER_STAT_MAP.items():
        exprs.append((_num(col) * pl.lit(r.get(abbr, 0.0))).alias(f"pts_{abbr}"))

    # Incompletions have no nflverse column: pass attempts minus completions.
    # Scored 0 from 2025 on; -0.1 each through 2024.
    exprs.append((
        (_num("attempts") - _num("completions")) * pl.lit(r.get("INC", 0.0))
    ).alias("pts_INC"))

    # §1's flat -1 "Total FG Missed" counts a BLOCKED kick as a miss. nflverse
    # tracks blocks separately, so fg_missed alone is short: 109/113 kicker-weeks
    # against ESPN, versus 113/113 with fg_blocked added.
    exprs.append((
        (_num("fg_missed") + _num("fg_blocked")) * pl.lit(r.get("FGM", 0.0))
    ).alias("pts_FGM"))

    # FG 0-39 is three nflverse buckets collapsed into one league bucket
    exprs.append((
        (_num("fg_made_0_19") + _num("fg_made_20_29") + _num("fg_made_30_39"))
        * pl.lit(r.get("FG0", 0.0))
    ).alias("pts_FG0"))

    # Punt- and kickoff-return TDs. `special_teams_tds` covers both; the punt
    # component is broken out so the two rules stay independent even though the
    # league happens to pay 6 for each.
    exprs.append((_num("pt_return_tds") * pl.lit(r.get("PRTD", 0.0))).alias("pts_PRTD"))
    exprs.append((
        (_num("special_teams_tds") - _num("pt_return_tds")).clip(lower_bound=0)
        * pl.lit(r.get("KRTD", 0.0))
    ).alias("pts_KRTD"))

    scored = df.select([c for c in PLAYER_IDENTITY if c in df.columns] + exprs)
    pts_cols = [c for c in scored.columns if c.startswith("pts_")]
    return scored.with_columns(
        pl.sum_horizontal(pts_cols).round(2).alias("fantasy_points")
    )


# ─── D/ST ────────────────────────────────────────────────────────────────────
# `fumble_recovery_opp` is the D/ST's fumble recoveries — recoveries of the
# OPPONENT's fumble. `def_fumbles` is a different quantity and matches ESPN on
# neither direction (MIN wk3 2025: ESPN 3 recoveries, fumble_recovery_opp 3,
# def_fumbles 0; CLE wk13: ESPN 0, fumble_recovery_opp 0, def_fumbles 3).
DST_FUMBLE_RECOVERY_COLUMN = "fumble_recovery_opp"

# nflverse splits a D/ST's touchdowns across three columns and ESPN pays 6 for
# every flavour. `def_tds` alone misses fumble-recovery and special-teams
# returns; the three together match ESPN on 128/128 team-weeks of 2025.
DST_TD_COLUMNS = ("def_tds", "fumble_recovery_tds", "special_teams_tds")


def score_dst(
    season: int,
    rules: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Score every team defense-week, including both bucketed tables."""
    r = load_rules(season) if rules is None else rules
    team = (
        nv.team_stats(season, offline=True)
        .filter(pl.col("season_type") == "REG")
        .select(
            "season", "week", "team",
            "def_sacks", "def_interceptions", DST_FUMBLE_RECOVERY_COLUMN,
            "def_safeties", *DST_TD_COLUMNS,
            "def_fg_blocks", "def_pat_blocks", "def_punt_blocks",
        )
    )
    sacks = dst_mod.dst_sacks(season)
    inputs = dst_mod.dst_inputs(season).select(
        "week", "team", "points_allowed", "yards_allowed",
        "opp", "opp_score", "offense_conceded_points",
    )
    df = (team.join(inputs, on=["week", "team"], how="left")
              .join(sacks, on=["week", "team"], how="left")
              .with_columns(pl.col("sacks").fill_null(0.0)))

    exprs = [
        # sacks counted from pbp, not team_stats — see dst.dst_sacks()
        (_num("sacks") * pl.lit(r.get("SK", 0.0))).alias("pts_SK"),
        (_num("def_interceptions") * pl.lit(r.get("INT", 0.0))).alias("pts_INT"),
        (_num(DST_FUMBLE_RECOVERY_COLUMN) * pl.lit(r.get("FR", 0.0))).alias("pts_FR"),
        (_num("def_safeties") * pl.lit(r.get("SF", 0.0))).alias("pts_SF"),
        # every defensive/return TD pays 6 (INTTD, FRTD, KRTD, PRTD, BLKKRTD)
        (sum(_num(c) for c in DST_TD_COLUMNS) * pl.lit(r.get("INTTD", 0.0)))
        .alias("pts_DEF_TD"),
        # a blocked punt, PAT or FG pays 2
        ((_num("def_fg_blocks") + _num("def_pat_blocks") + _num("def_punt_blocks"))
         * pl.lit(r.get("BLKK", 0.0))).alias("pts_BLKK"),
        bucket_points_expr("points_allowed", POINTS_ALLOWED_BUCKETS, r).alias("pts_PA"),
        bucket_points_expr("yards_allowed", YARDS_ALLOWED_BUCKETS, r).alias("pts_YA"),
        bucket_label_expr("points_allowed", POINTS_ALLOWED_BUCKETS).alias("pa_bucket"),
        bucket_label_expr("yards_allowed", YARDS_ALLOWED_BUCKETS).alias("ya_bucket"),
    ]
    scored = df.with_columns(exprs)
    pts_cols = [c for c in scored.columns if c.startswith("pts_")]
    return scored.with_columns(
        pl.sum_horizontal(pts_cols).round(2).alias("fantasy_points"),
        # nflverse spells the Rams "LA"; ESPN and the crosswalk use "LAR".
        pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("team"),
    )
