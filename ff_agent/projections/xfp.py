"""Expected fantasy points (xFP) under §1 — ADD-§A.

xFP is what a player *should* have scored from opportunity alone: no talent, no
luck. The gap to actual is FPOE, which is mostly efficiency and regresses. ADD-§A
says drafting off last year's POINTS buys a mix of opportunity (sticky) and luck
(not), which is the whole reason to separate them.

**Why this is a genuine edge here.** ``ff_opportunity`` publishes expected
COMPONENTS — expected receptions, expected yards, expected touchdowns — and its
own headline ``total_fantasy_points_exp`` is **full PPR with 4-point passing TDs**
(verified: 0.091 mean absolute difference against a PPR/4 recomputation). Every
public xFP number is therefore priced for a game nobody in this league is
playing. Feeding the components through the Milestone 2 engine instead prices
them for §1.

Two league-specific pieces do not exist upstream, and they are precisely the two
rules that make this league unusual:

* **0.05 per carry is pure opportunity.** A carry IS the opportunity, so those
  points are expected by construction — there is no luck to strip out and no
  model to build. No public xFP includes them because no public scoring pays for
  carries.
* **-1 per sack taken has no expected-value model anywhere.** Built here from
  dropbacks against an opponent-adjusted baseline. See ``expected_sacks``.
"""

from __future__ import annotations

import polars as pl
import nflreadpy as nfl

from ff_agent.config import HISTORY_SEASONS, SEASON
from ff_agent.data import nflverse as nv
from ff_agent.data.cache import cached
from ff_agent.scoring.rules import load_rules

MODEL_VERSION = "v1.0.0"
"""Pinned. ``latest`` would let an upstream model change every historical number
under us — the silent-corruption failure §0.2 and §10 are both about."""

# expected component -> §1 rule
EXPECTED_TO_RULE: dict[str, str] = {
    "pass_yards_gained_exp": "PY",
    "pass_touchdown_exp": "PTD",
    "pass_interception_exp": "INTT",
    "pass_two_point_conv_exp": "2PC",
    "rush_yards_gained_exp": "RY",
    "rush_touchdown_exp": "RTD",
    "rush_two_point_conv_exp": "2PR",
    "rec_yards_gained_exp": "REY",
    "receptions_exp": "REC",
    "rec_touchdown_exp": "RETD",
    "rec_two_point_conv_exp": "2PRE",
}

# actual columns used where "expected" is either meaningless or unavailable
ACTUAL_TO_RULE: dict[str, str] = {
    # a carry IS the opportunity — 0.05/carry needs no model (§3.4)
    "rush_attempt": "RA",
}

FUMBLE_COLUMNS = ("rec_fumble_lost", "rush_fumble_lost")
"""No expected-fumble model upstream. At -2 and low frequency these are close to
pure noise, so actuals are used and the fact is recorded rather than modelled."""


def opportunity(season: int, regular_season_only: bool = True, **kw) -> pl.DataFrame:
    """ff_opportunity weekly, cached. NB: a different repo from nflverse-data.

    **Includes the postseason** — weeks 19-22 — while every actual we compare
    against is regular-season only. Left unfiltered, players whose teams made a
    deep run accumulate extra expected points against a regular-season actual,
    and FPOE goes negative for nearly everyone. The regular-season boundary is
    taken from the schedule rather than hardcoded, because it moved from 17
    weeks to 18 in 2021.
    """
    def fetch() -> pl.DataFrame:
        return nfl.load_ff_opportunity(
            seasons=[season], stat_type="weekly", model_version=MODEL_VERSION
        )
    df = cached("ff_opportunity", fetch, season=season, source="ffopportunity", **kw)
    if not regular_season_only:
        return df
    reg = (
        nv.schedules(offline=True)
        .filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
        .select("game_id")
        .unique()
    )
    # rows with no player_id are unattributed/team-level plays; they aggregate
    # into a phantom "player" with hundreds of games and zero actual points.
    return df.join(reg, on="game_id", how="semi").filter(
        pl.col("player_id").is_not_null() & (pl.col("player_id") != "")
    )


# ─── expected sacks (the piece that does not exist upstream) ────────────────
def league_sack_baseline(seasons: list[int] | None = None) -> pl.DataFrame:
    """League sack rate per dropback, per season.

    Deliberately NOT the quarterback's own rate. xFP asks what opportunity alone
    implies, so the baseline has to be QB-neutral — the QB's deviation from it is
    then a measurable trait rather than an assumption (see ``sacks_over_expected``).
    """
    seasons = seasons or HISTORY_SEASONS
    rows = []
    for s in seasons:
        ps = nv.player_stats(s, offline=True).filter(
            (pl.col("season_type") == "REG") & (pl.col("position") == "QB")
        )
        att = float(ps["attempts"].sum() or 0)
        sk = float(ps["sacks_suffered"].sum() or 0)
        db = att + sk
        rows.append({"season": s, "league_sack_rate": sk / db if db else 0.0})
    return pl.DataFrame(rows)


def defense_sack_factor(season: int) -> pl.DataFrame:
    """Opponent sack-rate-allowed multiplier, from the PRIOR season.

    Prior season only — using the current season would leak outcome into an
    "expected" quantity.
    """
    prior = season - 1
    ts = nv.team_stats(prior, offline=True).filter(pl.col("season_type") == "REG")
    per_team = ts.group_by("team").agg(
        pl.col("def_sacks").sum().alias("sacks"),
        (pl.col("attempts").sum() + pl.col("def_sacks").sum()).alias("dropbacks_faced"),
    )
    lg = per_team.select(
        (pl.col("sacks").sum() / pl.col("dropbacks_faced").sum()).alias("lg")
    ).item()
    return per_team.with_columns(
        pl.when(pl.col("dropbacks_faced") > 0)
        .then((pl.col("sacks") / pl.col("dropbacks_faced")) / lg)
        .otherwise(1.0)
        .alias("def_sack_factor")
    ).select(pl.col("team").alias("opponent"), "def_sack_factor")


def expected_sacks(season: int) -> pl.DataFrame:
    """Expected sacks taken per QB-game from dropbacks alone.

    §3.3 makes this worth 12-15% of a QB's season and no public ranking prices
    it. ADD-§B.1 corrects sacks to be substantially a QB trait rather than a line
    property, so the QB's gap to this neutral baseline is signal — measured at
    0.394 year-over-year in Milestone 3, correlating -0.134 with next-season
    scoring.
    """
    ps = nv.player_stats(season, offline=True).filter(
        (pl.col("season_type") == "REG") & (pl.col("position") == "QB")
    ).select("player_id", "player_display_name", "week", "team",
             "attempts", "sacks_suffered")

    base = league_sack_baseline([season])["league_sack_rate"][0]

    sched = nv.schedules(offline=True).filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    opps = pl.concat([
        sched.select("week", pl.col("home_team").alias("team"),
                     pl.col("away_team").alias("opponent")),
        sched.select("week", pl.col("away_team").alias("team"),
                     pl.col("home_team").alias("opponent")),
    ])
    try:
        factor = defense_sack_factor(season)
    except Exception:
        factor = opps.select("opponent").unique().with_columns(
            pl.lit(1.0).alias("def_sack_factor")
        )

    return (
        ps.join(opps, on=["week", "team"], how="left")
        .join(factor, on="opponent", how="left")
        .with_columns(pl.col("def_sack_factor").fill_null(1.0))
        .with_columns(
            (pl.col("attempts") + pl.col("sacks_suffered")).alias("dropbacks")
        )
        .with_columns(
            (pl.col("dropbacks") * base * pl.col("def_sack_factor"))
            .round(3).alias("sacks_exp")
        )
        .with_columns(
            (pl.col("sacks_suffered") - pl.col("sacks_exp")).round(3)
            .alias("sacks_over_expected")
        )
        .select("player_id", "week", "dropbacks", "sacks_suffered",
                "sacks_exp", "sacks_over_expected")
    )


# ─── xFP ────────────────────────────────────────────────────────────────────
def xfp_weekly(season: int, rules: dict[str, float] | None = None) -> pl.DataFrame:
    """Expected §1 points per player-week, with actual points and FPOE."""
    r = load_rules(SEASON) if rules is None else rules
    o = opportunity(season)

    exp_terms = [
        (pl.col(c).cast(pl.Float64).fill_null(0.0) * r.get(a, 0.0))
        for c, a in EXPECTED_TO_RULE.items() if c in o.columns
    ]
    act_terms = [
        (pl.col(c).cast(pl.Float64).fill_null(0.0) * r.get(a, 0.0))
        for c, a in ACTUAL_TO_RULE.items() if c in o.columns
    ]
    fum = [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in FUMBLE_COLUMNS
           if c in o.columns]

    df = o.with_columns(
        (sum(exp_terms) if exp_terms else pl.lit(0.0)).alias("_xfp_core"),
        (sum(act_terms) if act_terms else pl.lit(0.0)).alias("xfp_carry_points"),
        ((sum(fum) if fum else pl.lit(0.0)) * r.get("FUML", 0.0)).alias("_fumble_points"),
    )

    # ff_opportunity types `week` as f64; nflverse player_stats uses i32
    sacks = expected_sacks(season).select(
        pl.col("player_id").cast(pl.Utf8),
        pl.col("week").cast(pl.Int64),
        "sacks_exp", "sacks_over_expected",
    )
    df = (
        df.with_columns(pl.col("week").cast(pl.Int64),
                        pl.col("player_id").cast(pl.Utf8))
        .join(sacks, on=["player_id", "week"], how="left")
        .with_columns(pl.col("sacks_exp").fill_null(0.0))
        .with_columns(
            (pl.col("sacks_exp") * r.get("SKD", 0.0)).alias("xfp_sack_points")
        )
        .with_columns(
            (pl.col("_xfp_core") + pl.col("xfp_carry_points")
             + pl.col("_fumble_points") + pl.col("xfp_sack_points"))
            .round(3).alias("xfp")
        )
    )

    # expected touchdowns (ADD-§A.1) — the highest-value, least-stable event
    td_exp = [c for c in ("pass_touchdown_exp", "rush_touchdown_exp",
                          "rec_touchdown_exp") if c in o.columns]
    td_act = [c for c in ("pass_touchdown", "rush_touchdown", "rec_touchdown")
              if c in o.columns]
    return df.with_columns(
        (sum(pl.col(c).cast(pl.Float64).fill_null(0.0) for c in td_exp))
        .round(3).alias("xtd"),
        (sum(pl.col(c).cast(pl.Float64).fill_null(0.0) for c in td_act))
        .alias("actual_td"),
    ).select(
        "season", "week", "player_id", "full_name", "position",
        pl.col("posteam").alias("team"),
        "xfp", "xfp_carry_points", "xfp_sack_points",
        "xtd", "actual_td", "sacks_exp", "sacks_over_expected",
    )


def xfp_season(season: int) -> pl.DataFrame:
    """Season aggregate, joined to realised §1 points so FPOE is available."""
    from ff_agent.projections import actuals as A

    w = xfp_weekly(season)
    agg = (
        w.group_by("player_id", "full_name", "position")
        .agg(
            pl.col("xfp").sum().round(2).alias("xfp"),
            pl.col("xtd").sum().round(2).alias("xtd"),
            pl.col("actual_td").sum().alias("actual_td"),
            pl.col("sacks_exp").sum().round(2).alias("sacks_exp"),
            pl.col("sacks_over_expected").sum().round(2).alias("sacks_over_expected"),
            pl.col("xfp_carry_points").sum().round(2).alias("xfp_carry_points"),
            pl.len().alias("games"),
        )
        .rename({"player_id": "canonical_id", "full_name": "name"})
    )
    act = A.player_season_actuals(season).select(
        "canonical_id", pl.col("points").alias("actual_points")
    )
    return (
        agg.join(act, on="canonical_id", how="left")
        .with_columns(
            pl.col("actual_points").fill_null(0.0),
            (pl.col("xfp") / pl.col("games")).round(3).alias("xfp_pg"),
            pl.lit(season).alias("season"),
        )
        .with_columns(
            (pl.col("actual_points") - pl.col("xfp")).round(2).alias("fpoe"),
            (pl.col("actual_td") - pl.col("xtd")).round(2).alias("td_over_expected"),
            (pl.col("actual_points") / pl.col("games")).round(3).alias("actual_pg"),
        )
        .sort("xfp", descending=True)
    )
