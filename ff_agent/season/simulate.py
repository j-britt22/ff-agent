"""Monte Carlo season simulator (§8) — the objective function.

    "Everything else optimizes against this."

Two design facts, both measured rather than assumed, and both from 2025's 112
team-weeks:

* **Week-to-week noise (sd 25.7) is roughly three times the talent spread
  between teams (sd 8.7).** Over a 12-game season, head-to-head results are
  dominated by luck. That is not a defect in the model, it is the league — and it
  is precisely why §2.4 concludes that roster variance is *good* and marginal
  regular-season wins are worth little. Expect wide probability bands; narrow
  ones here would be a lie.
* **The seeding rule is CONFIRMED as win percentage** (§2.5), from the live
  standings page ranking on PCT with a GB column. Both rules are still
  implemented, because the sensitivity is the thing that made confirming it
  worthwhile: on raw wins a 12-game team would have given up ~8 points of both
  P(playoffs) and P(top-2 seed). On win percentage, nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ff_agent.config import (
    FIRST_ROUND_BYES, MY_TEAM_NAME, PLAYOFF_TEAMS, SEASON, SEEDING_RULE,
)
from ff_agent.season import schedule as SCH

LEAGUE_WEEKLY_SD = 25.7
"""Median within-team weekly standard deviation, measured on 2025."""

TALENT_SD = 8.7
"""Between-team spread of season means, measured on 2025. The ratio to
LEAGUE_WEEKLY_SD is the single most important number in this module."""

SEEDING_RULES = ("wins", "win_pct")


@dataclass
class SimResult:
    teams: list[str]
    seeding_rule: str
    n_sims: int
    p_playoffs: np.ndarray
    p_top2: np.ndarray
    p_title: np.ndarray
    mean_wins: np.ndarray
    mean_points: np.ndarray

    def table(self) -> pl.DataFrame:
        return pl.DataFrame({
            "team": self.teams,
            "p_playoffs": np.round(self.p_playoffs, 4),
            "p_top2_seed": np.round(self.p_top2, 4),
            "p_title": np.round(self.p_title, 4),
            "mean_wins": np.round(self.mean_wins, 2),
            "mean_points": np.round(self.mean_points, 1),
        }).sort("p_title", descending=True)


def _bracket(seed_order: np.ndarray, scores_playoff: np.ndarray) -> np.ndarray:
    """6-team, 3-week bracket with byes for the top two seeds.

    Confirmed empirically: in the real 2025 bracket the top two seeds sat out
    week 15. Round 1 is 3v6 and 4v5; round 2 pairs seed 1 with the lowest
    surviving seed; round 3 is the final.
    """
    n_sims = seed_order.shape[0]
    champ = np.zeros(n_sims, dtype=int)
    s = seed_order                       # s[:, k] = team index seeded k+1

    def wins(a, b, wk):
        return np.where(scores_playoff[np.arange(n_sims), wk, a]
                        >= scores_playoff[np.arange(n_sims), wk, b], a, b)

    if PLAYOFF_TEAMS == 6 and FIRST_ROUND_BYES == 2:
        w36 = wins(s[:, 2], s[:, 5], 0)
        w45 = wins(s[:, 3], s[:, 4], 0)
        # reseed: the better remaining seed faces seed 2
        lo = np.where(np.isin(w36, s[:, 2]), w36, w45)
        hi = np.where(lo is w36, w45, w36)
        semi1 = wins(s[:, 0], np.where(w36 == s[:, 2], w45, w36), 1)
        semi2 = wins(s[:, 1], np.where(w36 == s[:, 2], w36, w45), 1)
        champ = wins(semi1, semi2, 2)
    else:  # generic single-elimination fallback
        champ = s[:, 0]
    return champ


def simulate(
    team_means: dict[str, float],
    team_sds: dict[str, float] | None = None,
    season: int = SEASON,
    n_sims: int = 20_000,
    seeding_rule: str = SEEDING_RULE,
    seed: int = 7,
    team_mean_sds: dict[str, float] | None = None,
) -> SimResult:
    """Run the season ``n_sims`` times and return seeding/title probabilities.

    ``team_sds`` is week-to-week noise, redrawn every week. ``team_mean_sds`` is
    a different animal: uncertainty in the season mean itself, drawn ONCE per
    simulated season and applied to every week. Added for M7, where two rosters
    can share a projected mean and differ in how sure that projection is.

    The distinction matters because correlated error moves standings far more
    than independent noise of the same size — being wrong about a player is
    being wrong in all fourteen weeks. Omit it and behaviour is unchanged.
    """
    if seeding_rule not in SEEDING_RULES:
        raise ValueError(f"seeding_rule must be one of {SEEDING_RULES}")

    sched = SCH.league_schedule(season)
    teams = sorted(team_means)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    weeks = sorted(sched["week"].unique().to_list())
    n_weeks = len(weeks)

    mu = np.array([team_means[t] for t in teams], dtype=float)
    sd = np.array([(team_sds or {}).get(t, LEAGUE_WEEKLY_SD) for t in teams], dtype=float)

    rng = np.random.default_rng(seed)
    offset = np.zeros((n_sims, 1, n))
    if team_mean_sds:
        msd = np.array([team_mean_sds.get(t, 0.0) for t in teams], dtype=float)
        offset = rng.normal(0.0, np.maximum(msd, 1e-9), size=(n_sims, n))[:, None, :]
    scores = (rng.normal(mu, sd, size=(n_sims, n_weeks, n)) + offset).clip(min=0.0)

    games = SCH.matchups(season)
    wins = np.zeros((n_sims, n), dtype=np.int32)
    played = np.zeros(n, dtype=np.int32)
    for row in games.iter_rows(named=True):
        wi = weeks.index(row["week"])
        a, b = idx[row["a"]], idx[row["b"]]
        a_wins = scores[:, wi, a] >= scores[:, wi, b]
        wins[:, a] += a_wins
        wins[:, b] += ~a_wins
        played[a] += 1
        played[b] += 1

    points = scores.sum(axis=1)

    if seeding_rule == "wins":
        primary = wins.astype(float)
    else:
        primary = wins / np.maximum(played, 1)

    # points-for breaks ties, scaled small so it never outranks the primary key
    key = primary + points * 1e-9
    order = np.argsort(-key, axis=1)

    made = np.zeros((n_sims, n), dtype=bool)
    top2 = np.zeros((n_sims, n), dtype=bool)
    rows = np.arange(n_sims)[:, None]
    made[rows, order[:, :PLAYOFF_TEAMS]] = True
    top2[rows, order[:, :FIRST_ROUND_BYES]] = True

    # the same season-long mean error carries into the bracket
    playoff_scores = (rng.normal(mu, sd, size=(n_sims, 3, n)) + offset).clip(min=0.0)
    champ = _bracket(order[:, :PLAYOFF_TEAMS], playoff_scores)
    title = np.zeros((n_sims, n), dtype=bool)
    title[np.arange(n_sims), champ] = True

    return SimResult(
        teams=teams, seeding_rule=seeding_rule, n_sims=n_sims,
        p_playoffs=made.mean(axis=0), p_top2=top2.mean(axis=0),
        p_title=title.mean(axis=0), mean_wins=wins.mean(axis=0),
        mean_points=points.mean(axis=0),
    )


def seeding_sensitivity(
    team_means: dict[str, float],
    team_sds: dict[str, float] | None = None,
    season: int = SEASON,
    n_sims: int = 20_000,
) -> pl.DataFrame:
    """§2.5's open question, answered as a measured difference.

    You play 12 games; four teams play 13. Under raw wins that is a structural
    disadvantage, under win percentage it is not. This reports how much it
    actually matters instead of assuming an answer.
    """
    a = simulate(team_means, team_sds, season, n_sims, "wins").table()
    b = simulate(team_means, team_sds, season, n_sims, "win_pct").table()
    g = SCH.games_played(season).select("team", "games")
    return (
        a.rename({c: f"{c}_wins" for c in a.columns if c != "team"})
        .join(b.rename({c: f"{c}_pct" for c in b.columns if c != "team"}), on="team")
        .join(g, on="team", how="left")
        .with_columns(
            (pl.col("p_playoffs_wins") - pl.col("p_playoffs_pct")).round(4).alias("d_playoffs"),
            (pl.col("p_top2_seed_wins") - pl.col("p_top2_seed_pct")).round(4).alias("d_top2"),
            (pl.col("p_title_wins") - pl.col("p_title_pct")).round(4).alias("d_title"),
        )
        .select("team", "games", "p_playoffs_wins", "p_playoffs_pct", "d_playoffs",
                "p_top2_seed_wins", "p_top2_seed_pct", "d_top2", "d_title")
        .sort("games")
    )


def bye_value(result_with_bye: float, result_without: float) -> float:
    """§2.4 claims a first-round bye is worth about as much as everything else
    combined (12.5% vs 25% between evenly matched teams). Measurable, not assumed."""
    return result_with_bye - result_without
