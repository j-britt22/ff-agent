"""Consensus baselines — the anchor §7.2 step 1 asks for.

Two independent sources, deliberately:

* **FantasyPros ECR, superflex** (``ecr_type = "rsf"``). §7.5 is explicit that
  standard ADP misstates positional demand in a 2-QB league — the superflex list
  puts 11 QBs in the top 24 where the standard list puts none. nflverse carries
  **weekly snapshots back to Dec 2019**, which is what makes a genuine no-lookahead
  preseason backtest possible. It also ships ``sd``/``best``/``worst``, a
  ready-made uncertainty band for §7.2 step 5.

* **ESPN's own season projections**, which arrive already in this league's exact
  scoring and carry a projected STAT LINE — what §7.2 step 2 wants.

Both are dirty in specific, documented ways. See ECR_QUALITY_FILTERS and
PER_GAME_FIELDS below; neither problem is visible unless you look for it.
"""

from __future__ import annotations

import polars as pl
import nflreadpy as nfl

from ff_agent.config import SEASON
from ff_agent.data import crosswalk as cw
from ff_agent.data.cache import cached

# ─── ECR ────────────────────────────────────────────────────────────────────
SUPERFLEX = "rsf"
"""Redraft superflex. §3.1: this league starts 18 QBs, so 1-QB ranks are wrong."""

ECR_QUALITY_FILTERS = """
The FantasyPros scrape carries artifacts that are invisible unless checked:
  * a literal placeholder row named "Player Name" ranked 12th OVERALL in the
    2025-09-05 superflex list;
  * single-expert entries (sd == 0 with best == worst) that land absurdly high —
    a free-agent RB at overall rank 8.
Both would poison a rank->points calibration at the very top, where it matters
most. Rows are dropped only for these explicit reasons, never silently.
"""

PLACEHOLDER_NAMES = frozenset({"player name", "", "n/a", "none"})


def _clean_ecr(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split an ECR snapshot into (usable, rejected-with-reason)."""
    df = df.with_columns(
        pl.col("player").str.strip_chars().str.to_lowercase().alias("_lname")
    )
    reason = (
        pl.when(pl.col("_lname").is_in(list(PLACEHOLDER_NAMES)))
        .then(pl.lit("placeholder name"))
        .when(pl.col("ecr").is_null())
        .then(pl.lit("null ecr"))
        # one expert, ranked inside the top 50 — no consensus behind it
        .when((pl.col("sd").fill_null(0) == 0) & (pl.col("ecr") <= 50))
        .then(pl.lit("single-expert rank inside top 50"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("reject_reason")
    )
    tagged = df.with_columns(reason)
    keep = tagged.filter(pl.col("reject_reason").is_null()).drop("reject_reason", "_lname")
    drop = tagged.filter(pl.col("reject_reason").is_not_null()).drop("_lname")
    return keep, drop


def ecr_snapshots(**kw) -> pl.DataFrame:
    """All historical ECR snapshots, cached."""
    def fetch() -> pl.DataFrame:
        return nfl.load_ff_rankings("all").with_columns(
            pl.col("scrape_date").str.to_date().alias("snapshot_date")
        )
    return cached("ff_rankings", fetch, source="ffverse", **kw)


def available_snapshots(ecr_type: str = SUPERFLEX) -> pl.DataFrame:
    return (
        ecr_snapshots().filter(pl.col("ecr_type") == ecr_type)
        .group_by("snapshot_date").agg(pl.len().alias("n"))
        .sort("snapshot_date")
    )


def preseason_snapshot_date(season: int, ecr_type: str = SUPERFLEX) -> pl.Date:
    """The last ECR snapshot before that season's week 1 — no lookahead.

    NFL week 1 is early September, so we take the latest snapshot on or before
    Sept 5 of the target season.
    """
    import datetime as _dt

    cutoff = _dt.date(season, 9, 5)
    snaps = available_snapshots(ecr_type).filter(pl.col("snapshot_date") <= cutoff)
    if snaps.is_empty():
        raise ValueError(f"No {ecr_type} ECR snapshot on or before {cutoff}")
    return snaps["snapshot_date"].max()


def ecr(
    season: int,
    ecr_type: str = SUPERFLEX,
    as_of: object | None = None,
    with_rejects: bool = False,
):
    """Consensus ranks for one season, resolved to canonical IDs.

    The ESPN crosswalk does not reach FantasyPros, so the join runs
    fantasypros_id -> ff_playerids -> gsis_id. Anything unresolved is REPORTED,
    never name-matched (§0.2).
    """
    date = as_of or preseason_snapshot_date(season, ecr_type)
    snap = ecr_snapshots().filter(
        (pl.col("ecr_type") == ecr_type) & (pl.col("snapshot_date") == date)
    )
    if snap.is_empty():
        raise ValueError(f"No {ecr_type} snapshot for {date}")

    keep, drop = _clean_ecr(snap)

    bridge = (
        nfl.load_ff_playerids()
        .select(
            pl.col("fantasypros_id").cast(pl.Utf8),
            pl.col("gsis_id").alias("canonical_id"),
        )
        .filter(pl.col("fantasypros_id").is_not_null() & pl.col("canonical_id").is_not_null())
        .unique(subset=["fantasypros_id"])
    )
    out = (
        keep.with_columns(pl.col("id").cast(pl.Utf8).alias("fantasypros_id"))
        .join(bridge, on="fantasypros_id", how="left")
        .with_columns(
            pl.lit(season).alias("season"),
            pl.col("team").map_elements(cw.normalize_team, return_dtype=pl.Utf8).alias("team"),
        )
        .select(
            "season", "snapshot_date", "canonical_id", "fantasypros_id",
            pl.col("player").alias("name"), pl.col("pos").alias("position"),
            "team", "ecr", "sd", "best", "worst",
        )  # NB: no `bye` here — the "all" snapshots omit it, and M1 derives
           # NFL byes from nflverse schedules anyway (§2.1).
        .sort("ecr")
    )
    return (out, drop) if with_rejects else out


def ecr_unresolved(season: int, ecr_type: str = SUPERFLEX) -> pl.DataFrame:
    """ECR rows that did not reach a canonical id — for human review."""
    return ecr(season, ecr_type).filter(pl.col("canonical_id").is_null())


# ─── ESPN season projections ────────────────────────────────────────────────
GAMES_IN_SEASON = 17

PER_GAME_FIELDS = frozenset({
    "passingYards", "rushingYards", "receivingYards",
})
"""ESPN's SEASON projection reports these four as PER-GAME while every other
field is a season total.

Gibbs 2026: ``rushingYards 80.74`` sits beside ``rushingAttempts 283.07`` and
``rushingYardsPerAttempt 4.849``. 283.07 x 4.849 = 1372.6, and 80.74 x 17 =
1372.6. Read literally, every yardage projection is wrong by a factor of ~17 —
and since yards are most of a skill player's score, so is the whole board.
"""

ESPN_PROJ_TO_RULE: dict[str, str] = {
    "passingYards": "PY", "passingTouchdowns": "PTD",
    "passingInterceptions": "INTT", "passingTimesSacked": "SKD",
    "passing2PtConversions": "2PC",
    "rushingYards": "RY", "rushingAttempts": "RA",
    "rushingTouchdowns": "RTD", "rushing2PtConversions": "2PR",
    "receivingYards": "REY", "receivingReceptions": "REC",
    "receivingTouchdowns": "RETD", "receiving2PtConversions": "2PRE",
    "lostFumbles": "FUML", "fumbleRecoveredForTD": "FTD",
    "madeExtraPoints": "PAT", "missedFieldGoals": "FGM",
    "madeFieldGoalsFromUnder40": "FG0", "madeFieldGoalsFrom40To49": "FG40",
}

KICKER_LONG_FG_FIELD = "madeFieldGoalsFrom50Plus"
"""ESPN projects one 50+ bucket; §1 pays 5 for 50-59 and 6 for 60+.

The projection collapses exactly the distinction ADD-§E says is worth money
here, so the split has to come from the historical distance distribution.
"""


def espn_season_projections(season: int = SEASON, size: int = 1200, **kw) -> pl.DataFrame:
    """ESPN's full-season projected stat line per player, yardage corrected."""

    def fetch() -> pl.DataFrame:
        from ff_agent.data.espn import get_league

        lg = get_league(season)
        players = list(lg.free_agents(size=size))
        for t in getattr(lg, "teams", []) or []:
            players += list(getattr(t, "roster", []) or [])

        rows: list[dict] = []
        for p in players:
            st = (getattr(p, "stats", {}) or {}).get(0) or {}
            bd = st.get("projected_breakdown") or {}
            if not bd:
                continue
            rec = {
                "espn_id": str(getattr(p, "playerId", "") or ""),
                "name": getattr(p, "name", None),
                "position": getattr(p, "position", None),
                "team": getattr(p, "proTeam", None),
                "espn_projected_points": float(st.get("projected_points") or 0.0),
            }
            for key, abbr in ESPN_PROJ_TO_RULE.items():
                v = float(bd.get(key) or 0.0)
                if key in PER_GAME_FIELDS:
                    v *= GAMES_IN_SEASON
                rec[f"proj_{abbr}"] = v
            rec["proj_FG50PLUS"] = float(bd.get(KICKER_LONG_FG_FIELD) or 0.0)
            rows.append(rec)
        if not rows:
            raise RuntimeError(f"No ESPN projections returned for {season}")
        return pl.DataFrame(rows).unique(subset=["espn_id"], keep="first")

    return cached("espn_projections", fetch, season=season, source="espn", **kw)
