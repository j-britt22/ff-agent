"""Opportunity features — §7.2 step 3, built per-game not per-season.

ADD-§B is the governing finding: **volume is sticky, efficiency is not**, and
per-game rates beat season totals because totals are contaminated by games
missed. So every feature here is a rate, and efficiency metrics are carried only
so their instability can be *measured* (ADD-§I-3c) rather than assumed.

The features track what the addendum says actually predicts:
  * WR/TE — target share (~0.70 YoY, the stickiest skill metric), air-yards
    share, and WOPR, the composite that captures volume *and* its quality (§B.2)
  * RB    — carries and touches per game (~0.60, the top RB predictor), which
            §3.4 also makes a first-class scoring input at 0.05/carry
  * QB    — pass volume, and sack rate, which §B.1 corrects to be substantially
            a QB trait (~0.50 sticky) rather than a line property
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import HISTORY_SEASONS
from ff_agent.data import nflverse as nv
from ff_agent.projections import actuals as A

# Per-game rate features. Efficiency entries are deliberate: ADD-§B says not to
# project them forward, and measuring their instability on our own data is how
# §I-3c justifies giving them near-zero weight.
VOLUME_FEATURES = [
    "targets_pg", "receptions_pg", "air_yards_pg", "carries_pg",
    "rush_yards_pg", "rec_yards_pg", "pass_att_pg", "pass_yards_pg",
    "rz_touches_pg",
]
SHARE_FEATURES = ["target_share", "air_yards_share", "wopr"]
EFFICIENCY_FEATURES = ["yards_per_carry", "yards_per_target", "td_rate", "sack_rate"]

ROLE_FEATURES = ["games", "points"]
"""Prior-season games played and total §1 points.

Per-game rates alone are a trap: a backup with two good appearances shows elite
rates and no role. Without these, the model ranked Jimmy Garoppolo and Joe Milton
III as top-10 QBs off 2024 rates. `games` proxies role security and `points` is
the naive last-season baseline any projection must beat."""

ALL_FEATURES = VOLUME_FEATURES + SHARE_FEATURES + EFFICIENCY_FEATURES + ROLE_FEATURES


def _safe_div(a: str, b: str) -> pl.Expr:
    return pl.when(pl.col(b) > 0).then(pl.col(a) / pl.col(b)).otherwise(None)


def assert_one_row_per_player(df: pl.DataFrame, where: str) -> None:
    """§0.2, applied to a feature table.

    "Every rostered and drafted player must resolve to exactly one canonical ID.
    Assert it." Resolution is only half of §0.2 — a table that carries the same
    person twice breaks it just as thoroughly, and every join downstream
    multiplies the damage. Fail here rather than deduplicating downstream, where
    which row survives is an accident of sort order.
    """
    dupes = df.group_by("canonical_id").len().filter(pl.col("len") > 1)
    if not dupes.height:
        return
    cols = [c for c in ("canonical_id", "name", "position", "team", "games",
                        "n_teams", "points", "model_points") if c in df.columns]
    # nulls_equal, and a semi-join rather than is_in: a NULL id duplicated is
    # still a §0.2 failure, but `null.is_in(...)` is null, so filtering would
    # drop exactly the rows being complained about and raise a blocking error
    # naming nobody. (.implode() does not help — it fixes the deprecation that
    # board/build.py already avoids, not the nulls.)
    detail = (
        df.join(dupes.select("canonical_id"), on="canonical_id", how="semi",
                nulls_equal=True)
        .select(cols).sort("canonical_id", nulls_last=True)
    )
    raise ValueError(
        f"{dupes.height} canonical_id(s) appear more than once in {where} "
        f"({int(dupes['len'].sum())} rows). A group key or a join fanned out; "
        f"find it rather than dropping a row here.\n{detail}"
    )


def player_season_features(season: int) -> pl.DataFrame:
    """One row per PLAYER-season of per-game opportunity — not per (player, team).

    nflverse stats are weekly and carry whichever team the player suited up for,
    so a mid-season trade splits his year into two stints. Keying on team emits
    both, and because every feature here is a RATE, the shorter stint often
    projects HIGHER: Adam Thielen's 5 games in PIT came out at 49.7 against his
    9 in MIN at 37.2, and 9 of 2025's 25 traded skill players were like that.
    Whatever the downstream then dropped, it was dropping most of a real season.

    So the stints are summed into one player-season here, at the root, and the
    rates recomputed over the COMBINED games. Two stints genuinely ARE two rows
    of opportunity — the fix is to add them, never to pick a winner. Efficiency
    and share features aggregate correctly for free: a ratio of the summed
    components is the volume-weighted season rate, and the share means now run
    over all of a player's weeks instead of one stint's.

    ``games`` is deliberately NOT clamped at 17. A traded player can legitimately
    log 18 in an 18-week season by missing neither team's bye — Rashid Shaheed
    did exactly that in 2025, 9 with NO and 9 with SEA.
    """
    ps = nv.player_stats(season, offline=True).filter(
        (pl.col("season_type") == "REG")
        # nflverse ships one placeholder weekly row per week with a NULL
        # player_id, no position and zero of every stat. Per-team they were 14-18
        # obvious junk rows; aggregated per player they collapse into a single
        # phantom with 18 games, which reads like a real player. Nothing
        # downstream would use it (every consumer filters on position) but a row
        # with no identity has no place in a §0.2 table.
        & pl.col("player_id").is_not_null()
    )

    agg = (
        # Checked across all ten history seasons: zero players carry two
        # positions or two display names within a season, so team was the only
        # thing the old key added — and all it added was the trade split.
        ps.group_by("player_id", "player_display_name", "position")
        .agg(
            pl.len().alias("games"),
            # the team he finished on, which is the closest this table gets to
            # where he lines up next year. Informational only: the board takes
            # team from ECR, never from here.
            pl.col("team").sort_by("week").last().alias("team"),
            pl.col("team").n_unique().alias("n_teams"),
            pl.col("targets").sum().alias("targets"),
            pl.col("receptions").sum().alias("receptions"),
            pl.col("receiving_air_yards").sum().alias("air_yards"),
            pl.col("receiving_yards").sum().alias("rec_yards"),
            pl.col("carries").sum().alias("carries"),
            pl.col("rushing_yards").sum().alias("rush_yards"),
            pl.col("attempts").sum().alias("pass_att"),
            pl.col("passing_yards").sum().alias("pass_yards"),
            pl.col("sacks_suffered").sum().alias("sacks_taken"),
            (pl.col("rushing_tds").sum() + pl.col("receiving_tds").sum()
             + pl.col("passing_tds").sum()).alias("total_tds"),
            pl.col("target_share").mean().alias("target_share"),
            pl.col("air_yards_share").mean().alias("air_yards_share"),
            pl.col("wopr").mean().alias("wopr"),
            # red-zone proxy: first downs inside scoring range are not exposed
            # directly, so rushing + receiving first downs stand in for RZ usage
            (pl.col("rushing_first_downs").sum()
             + pl.col("receiving_first_downs").sum()).alias("rz_touches"),
        )
        .rename({"player_id": "canonical_id", "player_display_name": "name"})
    )

    g = pl.col("games")
    out = agg.with_columns(
        (pl.col("targets") / g).alias("targets_pg"),
        (pl.col("receptions") / g).alias("receptions_pg"),
        (pl.col("air_yards") / g).alias("air_yards_pg"),
        (pl.col("rec_yards") / g).alias("rec_yards_pg"),
        (pl.col("carries") / g).alias("carries_pg"),
        (pl.col("rush_yards") / g).alias("rush_yards_pg"),
        (pl.col("pass_att") / g).alias("pass_att_pg"),
        (pl.col("pass_yards") / g).alias("pass_yards_pg"),
        (pl.col("rz_touches") / g).alias("rz_touches_pg"),
        _safe_div("rush_yards", "carries").alias("yards_per_carry"),
        _safe_div("rec_yards", "targets").alias("yards_per_target"),
        _safe_div("total_tds", "games").alias("td_rate"),
        # §B.1: sack rate is ~0.50 sticky and substantially the QB's own trait
        pl.when(pl.col("pass_att") + pl.col("sacks_taken") > 0)
        .then(pl.col("sacks_taken") / (pl.col("pass_att") + pl.col("sacks_taken")))
        .otherwise(None).alias("sack_rate"),
        pl.lit(season).alias("season"),
    )
    assert_one_row_per_player(out, f"player_season_features({season})")
    return out


def features_with_actuals(season: int) -> pl.DataFrame:
    """One season's features joined to that season's realised §1 points.

    ``points`` and ``games`` are ROLE features (see ROLE_FEATURES), so they have
    to travel with the opportunity rates when projecting forward. They only agree
    with each other because the stints are already summed upstream: actuals were
    always whole-season (``player_season_actuals`` never keyed on team), so while
    features were per-stint a traded player carried a full season of ``points``
    against a partial ``games`` — one role feature describing the year and the
    other describing half of it.
    """
    f = player_season_features(season)
    a = A.player_season_actuals(season).select("canonical_id", "points", "ppg")
    return f.join(a, on="canonical_id", how="left").with_columns(
        pl.col("points").fill_null(0.0)
    )


def feature_panel(seasons: list[int] | None = None) -> pl.DataFrame:
    """Features joined to that season's realised §1 points."""
    seasons = seasons or HISTORY_SEASONS
    return pl.concat([features_with_actuals(s) for s in seasons], how="diagonal")


def transitions(
    seasons: list[int] | None = None, min_games: int = 6
) -> pl.DataFrame:
    """Year-over-year pairs: season N features -> season N+1 outcome.

    This is the table ADD-§I-3c needs. The 2016 history window gives 9 such
    transitions instead of the 4 a 5-season window would allow, which is the
    difference between measuring stability and guessing at it.
    """
    panel = feature_panel(seasons)
    nxt = panel.select(
        "canonical_id",
        (pl.col("season") - 1).alias("season"),
        pl.col("ppg").alias("next_ppg"),
        pl.col("points").alias("next_points"),
        pl.col("games").alias("next_games"),
        # games and points are already emitted above as next_games/next_points
        *[pl.col(c).alias(f"next_{c}") for c in ALL_FEATURES
          if c not in ("games", "points")],
    )
    return (
        panel.join(nxt, on=["canonical_id", "season"], how="inner")
        .filter((pl.col("games") >= min_games) & (pl.col("next_games") >= min_games))
    )


def stability_table(seasons: list[int] | None = None) -> pl.DataFrame:
    """Measured year-over-year correlation of every feature, by position.

    ADD-§I-3c: weight inputs by measured stability on *your* data, not intuition.
    ADD-§B predicts target share ~0.70, RB touches ~0.60, sack rate ~0.50 and
    efficiency near zero — this is where those claims get checked.
    """
    t = transitions(seasons)
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        sub = t.filter(pl.col("position") == pos)
        if sub.height < 30:
            continue
        for f in ALL_FEATURES:
            pair = sub.select(f, f"next_{f}").drop_nulls()
            if pair.height < 30:
                continue
            r = pair.select(pl.corr(f, f"next_{f}")).item()
            # and how well it predicts next-season POINTS, which is what matters
            pp = sub.select(f, "next_ppg").drop_nulls()
            rp = pp.select(pl.corr(f, "next_ppg")).item() if pp.height >= 30 else None
            rows.append({
                "position": pos, "feature": f, "n": pair.height,
                "yoy_stability": round(r, 3) if r is not None else None,
                "corr_next_ppg": round(rp, 3) if rp is not None else None,
                "kind": ("volume" if f in VOLUME_FEATURES
                         else "share" if f in SHARE_FEATURES else "efficiency"),
            })
    return pl.DataFrame(rows).sort(["position", "yoy_stability"], descending=[False, True])
