"""Projected standings and playoff odds from the rosters the draft produced.

**This is the weakest of the three outputs and it ships with the number that
says so.** M6 ran precisely this task — simulate a season from opening-day
rosters, compare to what actually happened — on 2025:

* preseason projections scored Spearman **-0.21** against the real standings;
* a **perfect** simulator, handed each team's true realised strength, scores only
  **+0.52** on the same test (5th percentile -0.05);
* the same simulator fed **perfect player knowledge** on the same rosters scores
  **+0.43**.

So the mechanics are sound and the projections are the limit, and even a
flawless forecast could not score much above half. The reason is measured, not
excused: weekly noise (sd 25.7) runs about **three times** the spread of team
season means (sd 8.7), so twelve games cannot resolve team quality very far.

Read the output accordingly. **P(playoffs) and P(title) are the honest form of
this forecast; the projected finishing order is not.** Six of nine teams qualify
(§2.4), which makes the tournament the default outcome rather than the
achievement, and the top-two bye — worth roughly as much as everything else
combined — is what the probabilities are actually about.

Roster churn is not the missing piece, incidentally: M6 checked, and 81% of
2025's starter points came from drafted players.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import (
    DOUBLE_UP_MANAGERS, FIRST_ROUND_BYES, MY_TEAM_NAME, PLAYOFF_TEAMS, SEASON,
    SEEDING_RULE,
)
from ff_agent.season import simulate as SIM

N_SIMS = 20_000

CALIBRATION = {
    "test": (
        "M6 (§11): simulate 2025 from opening-day rosters, compare to the real "
        "final standings."
    ),
    "preseason_projections_spearman": -0.21,
    "perfect_simulator_ceiling_spearman": 0.52,
    "perfect_player_knowledge_spearman": 0.43,
    "binding_constraint": (
        "The projections, not the simulator. Even perfect player knowledge only "
        "reaches +0.43 on this test, and a perfect simulator +0.52, because "
        "weekly noise (sd 25.7) is ~3x the between-team talent spread (sd 8.7)."
    ),
    "how_to_read": (
        "Trust the probabilities, not the finishing order. A two-point gap in "
        "projected weekly points is inside the noise; a twenty-point gap is not."
    ),
    "roster_churn": (
        "Not the missing piece — 81% of 2025's starter points came from drafted "
        "players. But the waiver wire in this league stays rich all season "
        "(§3.2), and nothing here models it."
    ),
}


def predict(
    ratings: pl.DataFrame,
    season: int = SEASON,
    n_sims: int = N_SIMS,
    seeding_rule: str = SEEDING_RULE,
    use_uncertainty: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """Simulate the season from post-draft rosters.

    ``use_uncertainty`` feeds each team's projection uncertainty in as a season-
    long mean offset rather than weekly noise. M7 added that channel for a
    reason worth repeating: being wrong about a player is being wrong in all
    fourteen weeks, so correlated error moves standings far more than
    independent noise of the same size. A team built on four rookies and a team
    built on five known quantities can share a projected mean and should not
    share a distribution.
    """
    means = {r["team"]: float(r["weekly_points"]) for r in ratings.to_dicts()}
    _assert_covers_schedule(means, season)
    msds = (
        {r["team"]: max(float(r["weekly_points_sd"]), 1e-6) for r in ratings.to_dicts()}
        if use_uncertainty else None
    )
    res = SIM.simulate(
        means, season=season, n_sims=n_sims, seeding_rule=seeding_rule,
        team_mean_sds=msds,
    )
    table = res.table().join(
        ratings.select("team", "manager", "weekly_points", "weekly_points_sd",
                       "strength_rank"),
        on="team", how="left",
    )
    # §2.5 again, and it bit here first: ranking on RAW WINS put a 13-game team
    # above a 12-game team it was losing to on every other measure, so the
    # projected order openly contradicted the playoff odds beside it. The league
    # seeds on win PERCENTAGE (confirmed 2026-08-20), and so does this.
    from ff_agent.season import schedule as SCH

    games = SCH.games_played(season).select("team", "games")
    table = (
        table.join(games, on="team", how="left")
        .with_columns(
            (pl.col("mean_wins") / pl.col("games")).round(4).alias("mean_win_pct")
        )
        .with_columns(
            pl.col("mean_win_pct" if seeding_rule == "win_pct" else "mean_wins")
            .rank("ordinal", descending=True).cast(pl.Int32)
            .alias("projected_finish")
        )
        .sort("projected_finish")
    )

    meta = {
        "season": season,
        "n_sims": n_sims,
        "seeding_rule": seeding_rule,
        "seeding_rule_status": (
            "CONFIRMED win percentage (§2.5) from the standings page. With five "
            "teams playing 12 games and four playing 13, raw wins would have cost "
            "a 12-game team ~8 points of P(playoffs); on win percentage, nothing."
        ),
        "playoff_teams": PLAYOFF_TEAMS,
        "first_round_byes": FIRST_ROUND_BYES,
        "uncertainty_modelled": bool(use_uncertainty),
        "calibration": CALIBRATION,
    }
    return table, meta


def _assert_covers_schedule(means: dict[str, float], season: int) -> None:
    """Every team on the schedule must have a rating, and vice versa.

    Without this the simulator raises a bare ``KeyError: "leah's team"`` from
    inside a matchup loop, which says nothing about the cause. The cause is
    always the same one: a team was renamed and the ratings are keyed on a
    spelling the schedule does not use.
    """
    from ff_agent.season import schedule as SCH

    try:
        sched = set(SCH.games_played(season)["team"].to_list())
    except Exception:
        return
    missing = sorted(sched - set(means))
    extra = sorted(set(means) - sched)
    if missing or extra:
        raise ValueError(
            "the post-draft ratings do not line up with the league schedule, so "
            "the season cannot be simulated.\n"
            f"  on the schedule but not rated: {missing}\n"
            f"  rated but not on the schedule: {extra}\n"
            "Team names change and the two ESPN tables cache independently. "
            "Refresh with `cli settings` / `cli ingest`, or read the draft from "
            "the session log (--source log), which keys on managers."
        )


def my_schedule_view(
    table: pl.DataFrame, ratings: pl.DataFrame, my_team: str = MY_TEAM_NAME
) -> dict:
    """§2.3: two thirds of my schedule is four managers, so weight them 2x.

    A juggernaut faced twice hurts twice. This does not change the simulation —
    the schedule is already in it — it changes what is worth READING out of it.
    """
    # §1 spells one of these managers "R. Sharrett" and ESPN says "Rayne
    # Sharrett" — M5 hit this and built the alias map rather than matching
    # fuzzily. Skipping it here silently listed three of the four opponents who
    # account for two thirds of my schedule.
    from ff_agent.opponents.history import canonical_manager

    strength = {
        canonical_manager(r["manager"]): r["weekly_points"] for r in ratings.to_dicts()
    }
    rank = {
        canonical_manager(r["manager"]): r["strength_rank"] for r in ratings.to_dicts()
    }
    wanted = [canonical_manager(m) for m in DOUBLE_UP_MANAGERS]
    missing = [m for m in wanted if m not in strength]
    doubles = [
        {"manager": m, "weekly_points": strength.get(m), "strength_rank": rank.get(m)}
        for m in wanted if m in strength
    ]
    me = table.filter(pl.col("team") == my_team)
    return {
        "my_team": my_team,
        "faced_twice": sorted(doubles, key=lambda d: -(d["weekly_points"] or 0)),
        "faced_twice_mean_strength": (
            round(sum(d["weekly_points"] for d in doubles) / len(doubles), 2)
            if doubles else None
        ),
        "league_mean_strength": round(float(ratings["weekly_points"].mean()), 2),
        "my_row": me.to_dicts()[0] if me.height else None,
        "unmatched_double_up_managers": missing,
        "note": (
            "§2.2: my season ends in WEEK 13 — week 14 is a bye, so my record is "
            "locked a week before everyone else's. Waiver aggression should peak "
            "earlier than my leaguemates' instincts, and week 14 is a free week "
            "to churn the roster purely for the playoffs."
        ),
    }
