"""season_sim.json (§8) — the objective function every recommendation reports against."""

from __future__ import annotations

import json

import polars as pl

from ff_agent.config import ARTIFACTS_DIR, MY_TEAM_NAME, SEASON, SEEDING_RULE
from ff_agent.season import evaluate as EV
from ff_agent.season import schedule as SCH
from ff_agent.season import simulate as SIM


def build(
    team_means: dict[str, float] | None = None,
    season: int = SEASON,
    n_sims: int = 20_000,
) -> dict:
    teams = SCH.games_played(season)["team"].to_list()
    means = team_means or {t: 130.0 for t in teams}
    equal_strength = team_means is None

    out = {}
    for rule in SIM.SEEDING_RULES:
        r = SIM.simulate(means, season=season, n_sims=n_sims, seeding_rule=rule)
        out[rule] = r.table().to_dicts()

    return {
        "season": season,
        "my_team": MY_TEAM_NAME,
        "n_sims": n_sims,
        "equal_strength_baseline": equal_strength,
        "games_played": SCH.games_played(season).to_dicts(),
        "seeding_rule": SEEDING_RULE,
        "seeding_rule_status": (
            "CONFIRMED 2026-08-20 as win percentage — the standings page ranks on "
            "PCT with a GB column. §2.5's structural handicap does NOT apply. Both "
            "rules remain simulated because the sensitivity is what made this worth "
            "confirming: on raw wins a 12-game team would give up ~8 points of both "
            "P(playoffs) and P(top-2 seed)."
        ),
        "by_seeding_rule": out,
        "sensitivity": SIM.seeding_sensitivity(means, season=season, n_sims=n_sims).to_dicts(),
        "variance_model": {
            "within_team_weekly_sd": SIM.LEAGUE_WEEKLY_SD,
            "between_team_talent_sd": SIM.TALENT_SD,
            "note": (
                "Weekly noise is ~3x the talent spread, measured on 2025. Head-to-head "
                "results over 12 games are dominated by luck, which is exactly why "
                "§2.4 prefers ceiling over floor and treats marginal wins as cheap."
            ),
        },
    }


def write(
    team_means: dict[str, float] | None = None,
    season: int = SEASON,
    n_sims: int = 20_000,
) -> str:
    path = ARTIFACTS_DIR / "season_sim.json"
    path.write_text(json.dumps(build(team_means, season, n_sims), indent=2, default=str))
    return str(path)


def championship_delta(
    base_means: dict[str, float],
    changed_means: dict[str, float],
    team: str = MY_TEAM_NAME,
    season: int = SEASON,
    n_sims: int = 10_000,
) -> dict:
    """§8: "every recommendation reports its delta in championship probability".

    This is the function the waiver, lineup and trade jobs will call.
    """
    a = SIM.simulate(base_means, season=season, n_sims=n_sims)
    b = SIM.simulate(changed_means, season=season, n_sims=n_sims)
    ia = a.teams.index(team)
    ib = b.teams.index(team)
    return {
        "team": team,
        "p_title_before": round(float(a.p_title[ia]), 4),
        "p_title_after": round(float(b.p_title[ib]), 4),
        "delta_title": round(float(b.p_title[ib] - a.p_title[ia]), 4),
        "delta_top2": round(float(b.p_top2[ib] - a.p_top2[ia]), 4),
        "delta_playoffs": round(float(b.p_playoffs[ib] - a.p_playoffs[ia]), 4),
    }
