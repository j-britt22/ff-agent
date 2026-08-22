"""Own projection model — opportunity in, §1 points out.

Fitted per position on year-over-year transitions, with **no lookahead**: to
project season N the model only ever sees transitions that completed before N.

Feature weighting follows ADD-§I-3c — measured stability on our own data rather
than intuition. Measured here across 2016-2025: volume features average 0.584
year-over-year, efficiency 0.292, which is ADD-§B's "volume is sticky, efficiency
is not" reproduced on this league's scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ff_agent.projections import opportunity as O

MIN_STABILITY = 0.45
"""Features below this year-over-year correlation are dropped. ADD-§B is blunt
that prior-year efficiency should not be projected forward; this is where that
becomes a rule instead of an opinion."""

RIDGE_LAMBDA = 1.0
POSITIONS = ("QB", "RB", "WR", "TE")


@dataclass
class PositionModel:
    position: str
    features: list[str]
    coef: np.ndarray
    intercept: float
    n_train: int
    r2: float
    feature_stability: dict[str, float] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef + self.intercept


def _design(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Feature matrix with per-column median imputation for missing values."""
    cols = []
    for f in features:
        v = df[f].cast(pl.Float64).to_numpy().astype(float)
        med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
        cols.append(np.where(np.isfinite(v), v, med))
    return np.column_stack(cols) if cols else np.zeros((df.height, 0))


def select_features(
    stability: pl.DataFrame, position: str, min_stability: float = MIN_STABILITY
) -> list[str]:
    sub = stability.filter(
        (pl.col("position") == position)
        & (pl.col("yoy_stability") >= min_stability)
        & pl.col("corr_next_ppg").is_not_null()
    )
    # keep only features that also carry signal about next-season scoring
    sub = sub.filter(pl.col("corr_next_ppg").abs() >= 0.15)
    return sub.sort("yoy_stability", descending=True)["feature"].to_list()


def fit_position(
    train: pl.DataFrame, position: str, features: list[str], target: str = "next_points"
) -> PositionModel | None:
    sub = train.filter(pl.col("position") == position).drop_nulls(subset=[target])
    if sub.height < 40 or not features:
        return None
    X = _design(sub, features)
    y = sub[target].cast(pl.Float64).to_numpy().astype(float)

    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd

    n, p = Xs.shape
    A = Xs.T @ Xs + RIDGE_LAMBDA * np.eye(p)
    b = Xs.T @ (y - y.mean())
    beta = np.linalg.solve(A, b)

    pred = Xs @ beta + y.mean()
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # fold standardisation back so predict() takes raw features
    coef = beta / sd
    intercept = float(y.mean() - (mu / sd) @ beta)
    return PositionModel(position, features, coef, intercept, sub.height, round(r2, 4))


def fit(
    upto_season: int, transitions: pl.DataFrame | None = None,
    stability: pl.DataFrame | None = None,
) -> dict[str, PositionModel]:
    """Fit on transitions whose OUTCOME season is strictly before ``upto_season``.

    A transition labelled season N carries season N features and season N+1
    outcomes, so training for 2025 uses N+1 <= 2024, i.e. N <= 2023.
    """
    t = O.transitions() if transitions is None else transitions
    train = t.filter(pl.col("season") <= upto_season - 2)
    st = O.stability_table() if stability is None else stability

    models: dict[str, PositionModel] = {}
    for pos in POSITIONS:
        feats = select_features(st, pos)
        m = fit_position(train, pos, feats)
        if m:
            m.feature_stability = {
                r["feature"]: r["yoy_stability"]
                for r in st.filter(pl.col("position") == pos).to_dicts()
                if r["feature"] in feats
            }
            models[pos] = m
    return models


def project(season: int, models: dict[str, PositionModel] | None = None) -> pl.DataFrame:
    """Project ``season`` §1 points from season-1 opportunity."""
    models = fit(season) if models is None else models
    prior = O.features_with_actuals(season - 1)
    out = []
    for pos, m in models.items():
        sub = prior.filter(pl.col("position") == pos)
        if sub.is_empty():
            continue
        pred = m.predict(_design(sub, m.features))
        out.append(
            sub.select("canonical_id", "name", "position", "team")
            .with_columns(
                pl.Series("model_points", np.clip(pred, 0.0, None)).round(2),
                pl.lit(season).alias("season"),
            )
        )
    if not out:
        return pl.DataFrame(schema={"canonical_id": pl.Utf8, "name": pl.Utf8,
                                    "position": pl.Utf8, "team": pl.Utf8,
                                    "model_points": pl.Float64, "season": pl.Int64})
    df = pl.concat(out).sort("model_points", descending=True)

    # A player who changed NFL teams mid-season has TWO rows in the prior-season
    # opportunity table, one per team, and both survive to here. Left as-is they
    # fan out through blend()'s left join (1 -> 2) and again through the board's
    # tier join (2 -> 4), so §0.2's "exactly one canonical ID" quietly fails —
    # by the 2026 board, 15 players appeared four times each. Keep the row with
    # the most projected production, which is the fuller stint; ordering above
    # already puts it first.
    return df.unique(subset=["canonical_id"], keep="first", maintain_order=True)


def blend(
    consensus: pl.DataFrame, model: pl.DataFrame, weight: float = 0.12
) -> pl.DataFrame:
    """§7.2 step 4: blend, never replace.

    ``weight`` is the share given to our own model. Consensus encodes offseason
    news no historical model can see, so it stays the anchor; we lean on our own
    numbers where we have signal it lacks (sacks taken, carries).
    """
    j = consensus.join(
        model.select("canonical_id", "model_points"), on="canonical_id", how="left"
    )
    return j.with_columns(
        pl.when(pl.col("model_points").is_null())
        .then(pl.col("consensus_points"))
        .otherwise(
            weight * pl.col("model_points") + (1 - weight) * pl.col("consensus_points")
        )
        .round(2)
        .alias("blended_points")
    ).sort("blended_points", descending=True, nulls_last=True)
