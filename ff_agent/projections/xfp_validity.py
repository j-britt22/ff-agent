"""MILESTONE 3b TEST (ADD-§I-3b).

    "recomputed xFP correlates with next-season actual points better than
     prior-season actual points does. That test is the whole justification for
     the milestone — if it fails, the model is wrong."

So the comparison is deliberately unkind to xFP: the baseline is not a straw man,
it is the single strongest simple predictor there is. Milestone 3 already
measured prior-season points at 0.750 (WR), 0.721 (RB), 0.740 (TE) and 0.496
(QB) against next-season scoring. Beating that at WR is a high bar, and a
negative result is a real possible outcome worth reporting rather than tuning
away.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import HISTORY_SEASONS
from ff_agent.projections import xfp as X

MIN_GAMES = 8
"""Both seasons of a pair. Per-game rates from tiny samples are noise, and ADD-§B
is explicit that season totals are contaminated by games missed."""


def transitions(seasons: list[int] | None = None, min_games: int = MIN_GAMES) -> pl.DataFrame:
    """Season N xFP and points -> season N+1 points. One row per player-pair."""
    seasons = seasons or HISTORY_SEASONS
    frames = [X.xfp_season(s) for s in seasons]
    panel = pl.concat(frames, how="diagonal")

    nxt = panel.select(
        "canonical_id",
        (pl.col("season") - 1).alias("season"),
        pl.col("actual_points").alias("next_points"),
        pl.col("actual_pg").alias("next_pg"),
        pl.col("games").alias("next_games"),
        pl.col("actual_td").alias("next_td"),
    )
    return (
        panel.join(nxt, on=["canonical_id", "season"], how="inner")
        .filter((pl.col("games") >= min_games) & (pl.col("next_games") >= min_games))
        .filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    )


def _corr(df: pl.DataFrame, a: str, b: str) -> float | None:
    sub = df.select(a, b).drop_nulls()
    if sub.height < 25:
        return None
    return sub.select(pl.corr(a, b)).item()


def validity(seasons: list[int] | None = None, target: str = "next_pg") -> pl.DataFrame:
    """Does xFP/game beat prior points/game at predicting next-season scoring?"""
    t = transitions(seasons)
    rows = []
    for pos in ("QB", "RB", "WR", "TE", "ALL"):
        sub = t if pos == "ALL" else t.filter(pl.col("position") == pos)
        if sub.height < 25:
            continue
        r_xfp = _corr(sub, "xfp_pg", target)
        r_pts = _corr(sub, "actual_pg", target)
        rows.append({
            "position": pos, "n": sub.height,
            "xfp_pg": round(r_xfp, 4) if r_xfp else None,
            "prior_points_pg": round(r_pts, 4) if r_pts else None,
            "xfp_wins": bool(r_xfp and r_pts and r_xfp > r_pts),
            "delta": round(r_xfp - r_pts, 4) if (r_xfp and r_pts) else None,
        })
    return pl.DataFrame(rows)


def td_regression_evidence(seasons: list[int] | None = None) -> pl.DataFrame:
    """ADD-§A.1: does touchdown-over-expected regress?

    If TD luck regresses, this season's TD-over-expected should be NEGATIVELY
    related to next season's — a player who overshot comes back down.
    """
    t = transitions(seasons)
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        sub = t.filter(pl.col("position") == pos)
        if sub.height < 25:
            continue
        rows.append({
            "position": pos, "n": sub.height,
            "xtd_to_next_td": round(_corr(sub, "xtd", "next_td") or 0, 4),
            "actual_td_to_next_td": round(_corr(sub, "actual_td", "next_td") or 0, 4),
            "tdoe_persistence": round(_corr(sub, "td_over_expected", "next_td") or 0, 4),
        })
    return pl.DataFrame(rows)


def sack_trait_evidence(seasons: list[int] | None = None) -> pl.DataFrame:
    """ADD-§B.1: sacks-over-expected should PERSIST (a trait), unlike TD luck.

    This is the distinction that matters for a -1/sack league: TD-over-expected
    is noise to fade, sack-over-expected is a durable property to price in.
    """
    t = transitions(seasons).filter(pl.col("position") == "QB")
    nxt = (
        pl.concat([X.xfp_season(s) for s in (HISTORY_SEASONS)], how="diagonal")
        .select("canonical_id", (pl.col("season") - 1).alias("season"),
                pl.col("sacks_over_expected").alias("next_soe"))
    )
    j = t.join(nxt, on=["canonical_id", "season"], how="inner")
    return pl.DataFrame([{
        "n": j.height,
        "soe_year_over_year": round(_corr(j, "sacks_over_expected", "next_soe") or 0, 4),
        "tdoe_year_over_year": round(_corr(j, "td_over_expected", "next_td") or 0, 4),
    }])
