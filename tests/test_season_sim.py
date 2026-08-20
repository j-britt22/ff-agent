"""Milestone 6 — season simulator (§8, §2.4, §2.5)."""
import numpy as np
import polars as pl
import pytest

from ff_agent.config import FIRST_ROUND_BYES, MY_TEAM_NAME, PLAYOFF_TEAMS
from ff_agent.season import build as SB
from ff_agent.season import evaluate as EV
from ff_agent.season import lineup as LU
from ff_agent.season import schedule as SCH
from ff_agent.season import simulate as SIM

pytestmark = pytest.mark.espn


@pytest.fixture(scope="module")
def equal_means():
    return {t: 130.0 for t in SCH.games_played(2026)["team"].to_list()}


# ─── schedule and §2.5 ──────────────────────────────────────────────────────
def test_bye_is_encoded_as_playing_yourself():
    """ESPN represents a bye as a team facing itself — read naively that gives a
    14-game season to someone who plays 12."""
    s = SCH.league_schedule(2026)
    assert s.filter(pl.col("opponent").is_null()).height == 14
    assert (s["team"] != s["opponent"]).all()


def test_unequal_games_are_real_and_i_am_on_the_short_side():
    g = SCH.games_played(2026)
    counts = dict(zip(g["team"].to_list(), g["games"].to_list()))
    assert sorted(set(counts.values())) == [12, 13]
    assert counts[MY_TEAM_NAME] == 12
    assert sum(1 for v in counts.values() if v == 12) == 5
    assert int(g["n_byes"].sum()) == 14           # one bye per week


def test_four_games_every_week_with_nine_teams():
    m = SCH.matchups(2026)
    per_week = m.group_by("week").len()["len"].unique().to_list()
    assert per_week == [4]
    assert m.height == 56


# ─── simulator invariants ───────────────────────────────────────────────────
def test_probabilities_are_coherent(equal_means):
    r = SIM.simulate(equal_means, season=2026, n_sims=8000)
    assert abs(float(r.p_playoffs.sum()) - PLAYOFF_TEAMS) < 0.05
    assert abs(float(r.p_top2.sum()) - FIRST_ROUND_BYES) < 0.05
    assert abs(float(r.p_title.sum()) - 1.0) < 0.01


def test_identical_teams_get_the_spec_qualification_rate(equal_means):
    """§2.4: 6 of 9 make the playoffs, so a berth is the default outcome."""
    r = SIM.simulate(equal_means, season=2026, n_sims=8000)
    assert abs(float(r.p_playoffs.mean()) - 6 / 9) < 0.02
    assert abs(float(r.p_title.mean()) - 1 / 9) < 0.01


def test_a_stronger_team_wins_more(equal_means):
    m = dict(equal_means)
    best = sorted(m)[0]
    m[best] = 160.0
    r = SIM.simulate(m, season=2026, n_sims=8000)
    i = r.teams.index(best)
    assert r.p_title[i] > 2.0 / 9
    assert r.p_playoffs[i] > 0.85


# ─── §2.5, answered ─────────────────────────────────────────────────────────
def test_raw_wins_penalises_the_twelve_game_teams(equal_means):
    """The open question, converted into a measured number: with every team
    identical, seeding on raw wins costs a 12-game team real probability."""
    s = SIM.seeding_sensitivity(equal_means, season=2026, n_sims=20000)
    g12 = s.filter(pl.col("games") == 12)
    g13 = s.filter(pl.col("games") == 13)
    gap = float(g13["p_playoffs_wins"].mean() - g12["p_playoffs_wins"].mean())
    assert gap > 0.05, f"expected a real raw-wins penalty, got {gap:.3f}"
    # win percentage should remove it
    gap_pct = abs(float(g13["p_playoffs_pct"].mean() - g12["p_playoffs_pct"].mean()))
    assert gap_pct < 0.02, f"win_pct should be neutral, got {gap_pct:.3f}"


def test_both_seeding_rules_are_shipped():
    assert set(SIM.SEEDING_RULES) == {"wins", "win_pct"}


# ─── lineup ─────────────────────────────────────────────────────────────────
def test_flex_takes_the_best_remaining_not_a_starter():
    """§9.2: greedy slot-filling gets the FLEX wrong. Strict slots first, then
    FLEX from the leftovers, is optimal for this slot structure."""
    roster = pl.DataFrame({
        "canonical_id": [f"p{i}" for i in range(9)],
        "position": ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "TE", "K"],
        "weekly_points": [20, 18, 15, 14, 13, 12, 11, 9, 8],
    })
    lu = LU.optimal_lineup(roster)
    flex = lu.filter(pl.col("slot") == "FLEX")
    assert flex.height == 1
    assert flex["weekly_points"][0] == 13      # RB3, the best true leftover
    assert set(lu.filter(pl.col("slot") == "QB")["weekly_points"].to_list()) == {20, 18}


# ─── §11 test, and its honest limits ────────────────────────────────────────
def test_structural_invariants_hold_where_the_backtest_cannot_reach():
    """2025 had NO regular-season byes (8 teams, all 14 games), so bye and
    unequal-game handling cannot be validated against it. Checked as invariants."""
    assert EV.structural_checks(2026) == []


def test_simulator_mechanics_are_sound_given_good_inputs():
    """Fed PERFECT player knowledge on the same rosters and schedule, the
    simulator reaches +0.43 against actual standings — close to the +0.52 a
    perfect-strength model achieves. The mechanics are not the limitation."""
    d = EV.decompose(2025)
    perfect = d.filter(pl.col("team_strength_from") == "perfect player knowledge")
    assert float(perfect["vs_actual_standings"][0]) > 0.3


def test_preseason_projections_are_the_weak_link_and_that_is_recorded():
    """Fed preseason projections instead, the same simulator scores NEGATIVE.
    Recorded so the limitation is attributed to the projections rather than
    quietly blamed on luck."""
    d = EV.decompose(2025)
    proj = d.filter(pl.col("team_strength_from") == "preseason projections")
    perfect = d.filter(pl.col("team_strength_from") == "perfect player knowledge")
    assert float(proj["vs_actual_standings"][0]) < float(perfect["vs_actual_standings"][0])


def test_the_test_itself_has_a_low_ceiling():
    """A PERFECT simulator averages only ~0.52 on this test, with a 5th
    percentile near zero. Weekly noise is ~3x the talent spread, so a 14-game
    season simply cannot be reproduced closely."""
    c = EV.ceiling(2025, trials=120)
    assert 0.3 < c["mean_rho"] < 0.75
    assert c["p05"] < 0.15


# ─── output ─────────────────────────────────────────────────────────────────
def test_season_sim_json_ships_both_rules_and_names_the_confirmed_one():
    """Both rules stay simulated even though the rule is now known — the
    sensitivity is what made confirming it worth doing."""
    d = SB.build(season=2026, n_sims=3000)
    assert set(d["by_seeding_rule"]) == {"wins", "win_pct"}
    assert d["seeding_rule"] == "win_pct"
    assert "CONFIRMED" in d["seeding_rule_status"]
    assert d["variance_model"]["within_team_weekly_sd"] > d["variance_model"]["between_team_talent_sd"]


def test_championship_delta_is_available_for_downstream_jobs(equal_means):
    """§8: every recommendation reports its delta in championship probability."""
    better = dict(equal_means)
    better[MY_TEAM_NAME] = 145.0
    d = SB.championship_delta(equal_means, better, n_sims=4000)
    assert d["delta_title"] > 0.02
    assert d["delta_playoffs"] > 0.0


def test_seeding_rule_is_confirmed_as_win_percentage():
    """§2.5 CLOSED. The standings page ranks on PCT with a GB column, so the
    12-game teams carry no structural handicap.

    Had it been raw wins, M6 measured the cost at ~8 points of both P(playoffs)
    and P(top-2 seed) — which is why this was worth confirming rather than
    assuming.
    """
    from ff_agent.config import SEEDING_RULE
    assert SEEDING_RULE == "win_pct"
    assert SEEDING_RULE in SIM.SEEDING_RULES


def test_confirmed_rule_removes_the_unequal_games_handicap(equal_means):
    """Under the confirmed rule, playing 12 games instead of 13 costs nothing."""
    from ff_agent.config import SEEDING_RULE
    r = SIM.simulate(equal_means, season=2026, n_sims=20000, seeding_rule=SEEDING_RULE)
    tbl = r.table().join(SCH.games_played(2026).select("team", "games"), on="team")
    by = tbl.group_by("games").agg(pl.col("p_playoffs").mean().alias("p"))
    d = dict(zip(by["games"].to_list(), by["p"].to_list()))
    assert abs(d[12] - d[13]) < 0.02, f"12-game {d[12]:.3f} vs 13-game {d[13]:.3f}"
