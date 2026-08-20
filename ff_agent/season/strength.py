"""Team strength from a roster (§8 step 1).

Weekly expected points come from solving the optimal lineup, not from summing a
roster — §9.2 is explicit that the FLEX has to be assigned rather than filled.

Variance is shrunk toward the league's measured 25.7: eight teams over fourteen
weeks is far too little to estimate eight separate variances, and an overfitted
per-team sd would produce confident nonsense about who is "consistent".
"""

from __future__ import annotations

import polars as pl

from ff_agent.season.lineup import lineup_points
from ff_agent.season.simulate import LEAGUE_WEEKLY_SD

GAMES = 17
VARIANCE_SHRINKAGE_WEEKS = 20.0
"""Pseudo-weeks pulling a team's observed sd toward the league value."""


def roster_strength(
    rosters: pl.DataFrame, projections: pl.DataFrame, games: int = GAMES
) -> dict[str, float]:
    """Expected weekly points per fantasy team from its optimal lineup.

    ``rosters`` needs ``fantasy_team`` and ``canonical_id``; ``projections``
    needs ``canonical_id``, ``position`` and ``blended_points``.
    """
    proj = projections.select(
        "canonical_id", "position",
        (pl.col("blended_points") / games).alias("weekly_points"),
    )
    joined = rosters.join(proj, on="canonical_id", how="left").with_columns(
        pl.col("weekly_points").fill_null(0.0),
        pl.col("position").fill_null("NA"),
    )
    out: dict[str, float] = {}
    for team in joined["fantasy_team"].unique().to_list():
        if team is None:
            continue
        out[team] = round(lineup_points(joined.filter(pl.col("fantasy_team") == team)), 2)
    return out


def observed_variance(weekly_scores: pl.DataFrame) -> dict[str, float]:
    """Per-team weekly sd, shrunk toward the league value."""
    k = VARIANCE_SHRINKAGE_WEEKS
    agg = weekly_scores.group_by("team").agg(
        pl.col("points").std().alias("sd"), pl.len().alias("n")
    )
    out = {}
    for r in agg.to_dicts():
        sd = r["sd"] if r["sd"] is not None else LEAGUE_WEEKLY_SD
        n = r["n"] or 0
        out[r["team"]] = round((n * sd + k * LEAGUE_WEEKLY_SD) / (n + k), 2)
    return out
