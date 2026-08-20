"""MILESTONE 3 TEST (§11): does the blend beat raw consensus?

Rules of the exercise, chosen before looking at any result:

* **No lookahead.** Consensus is the last ECR snapshot on or before Sept 5 of the
  target season; the model is fitted only on transitions that completed before it.
* **The universe is the consensus board**, not the players who happened to
  produce. A consensus-ranked player who never played scores zero and stays in
  the sample — dropping him would flatter consensus by hiding its misses.
* **Spearman**, because §11 asks for rank correlation and a draft board is a
  ranking. Reported overall and by position, since a blend can help at one
  position and hurt at another.

If the blend loses, the honest answer is to ship consensus. §7.2 step 4 says
blend rather than replace, and a model that loses to consensus is worse than no
model at all.
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl

from ff_agent.projections import actuals as A
from ff_agent.projections import calibration as CAL
from ff_agent.projections import consensus as C
from ff_agent.projections import model as M

DEFAULT_WEIGHT = 0.12
"""Share given to our own model; consensus keeps the rest.

Chosen from a PLATEAU, not a peak. Every weight in 0.05-0.20 beats consensus in
all five backtested seasons, and walk-forward selection (weight fitted only on
prior seasons) independently picks 0.10 every time and wins 4/4. A result that
survives across a range of weights is evidence; a result that needs one exact
weight is a fitted artifact.

The pre-registered 0.35 merely tied — the model deserves less weight than
intuition suggested, which is itself the finding. Model-only scores 0.687
against consensus 0.761, so consensus stays the anchor exactly as §7.2 step 4
prescribes."""


@lru_cache(maxsize=16)
def _base_frame(season: int) -> pl.DataFrame:
    """Consensus + model + outcome for one season. Weight-independent, so the
    expensive parts (curve fit, model fit) are computed once per season."""
    ecr = C.ecr(season).filter(pl.col("canonical_id").is_not_null())
    curve = CAL.smooth_curve(
        CAL.rank_points_curve(seasons=[s for s in range(2016, season)])
    )
    cons = CAL.consensus_to_points(ecr, curve)
    proj = M.project(season)
    act = A.player_season_actuals(season).select(
        "canonical_id", pl.col("points").alias("actual_points")
    )
    return (
        cons.join(proj.select("canonical_id", "model_points"),
                  on="canonical_id", how="left")
        .join(act, on="canonical_id", how="left")
        .with_columns(
            pl.col("actual_points").fill_null(0.0),
            pl.col("model_points").fill_null(pl.col("consensus_points")),
        )
    )


def evaluation_frame(season: int, weight: float = DEFAULT_WEIGHT) -> pl.DataFrame:
    """Consensus, model and blend for one season, joined to what happened."""
    return _base_frame(season).with_columns(
        (weight * pl.col("model_points")
         + (1 - weight) * pl.col("consensus_points")).round(2).alias("blended_points")
    )


def _spearman(df: pl.DataFrame, a: str, b: str) -> float | None:
    sub = df.select(a, b).drop_nulls()
    if sub.height < 10:
        return None
    return sub.select(pl.corr(a, b, method="spearman")).item()


def score_season(season: int, weight: float = DEFAULT_WEIGHT) -> pl.DataFrame:
    ev = evaluation_frame(season, weight)
    rows = []
    for label, groups in (("ALL", [ev]),
                          ("BY_POS", [ev.filter(pl.col("position") == p)
                                      for p in ("QB", "RB", "WR", "TE")])):
        for g in groups:
            if g.is_empty():
                continue
            pos = "ALL" if label == "ALL" else g["position"][0]
            rows.append({
                "season": season, "scope": pos, "n": g.height,
                "consensus": _spearman(g, "consensus_points", "actual_points"),
                "model": _spearman(g, "model_points", "actual_points"),
                "blend": _spearman(g, "blended_points", "actual_points"),
            })
    return pl.DataFrame(rows).with_columns(
        (pl.col("blend") - pl.col("consensus")).alias("blend_minus_consensus")
    )


def backtest(
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
    weight: float = DEFAULT_WEIGHT,
) -> pl.DataFrame:
    return pl.concat([score_season(s, weight) for s in seasons])


def weight_sweep(
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
    weights: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0),
) -> pl.DataFrame:
    """Blend weight vs overall Spearman. 0.0 is pure consensus, 1.0 pure model."""
    rows = []
    for w in weights:
        r = backtest(seasons, weight=w).filter(pl.col("scope") == "ALL")
        rows.append({
            "weight": w,
            "mean_blend": round(r["blend"].mean(), 4),
            "mean_consensus": round(r["consensus"].mean(), 4),
            "seasons_won": int((r["blend"] > r["consensus"]).sum()),
            "n_seasons": r.height,
        })
    return pl.DataFrame(rows)


# ─── walk-forward weight selection ──────────────────────────────────────────
CANDIDATE_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def walk_forward(
    seasons: tuple[int, ...] = (2022, 2023, 2024, 2025),
    weights: tuple[float, ...] = CANDIDATE_WEIGHTS,
    per_position: bool = True,
) -> pl.DataFrame:
    """Choose the blend weight using only PRIOR seasons, then evaluate.

    Sweeping weights and reporting the best is how a backtest lies to you: the
    winning weight was chosen with knowledge of the answer. Here season N's
    weight is fitted on seasons < N only, so the reported figure is one a person
    could actually have achieved on draft day.
    """
    scored = {s: {w: score_season(s, w) for w in weights} for s in seasons}
    rows = []
    for i, season in enumerate(seasons):
        prior = seasons[:i]
        if not prior:
            continue
        scopes = ("QB", "RB", "WR", "TE") if per_position else ("ALL",)
        for scope in scopes:
            best_w, best_v = weights[0], -9.9
            for w in weights:
                vals = [
                    scored[p][w].filter(pl.col("scope") == scope)["blend"][0]
                    for p in prior
                ]
                v = sum(vals) / len(vals)
                if v > best_v:
                    best_w, best_v = w, v
            row = scored[season][best_w].filter(pl.col("scope") == scope)
            rows.append({
                "season": season, "scope": scope,
                "chosen_weight": best_w,
                "consensus": round(row["consensus"][0], 4),
                "blend": round(row["blend"][0], 4),
                "delta": round(row["blend"][0] - row["consensus"][0], 4),
                "n": int(row["n"][0]),
            })
    return pl.DataFrame(rows)
