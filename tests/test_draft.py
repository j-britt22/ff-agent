"""M7 — Monte Carlo draft simulator.

Three of these are load-bearing and the rest support them:

* ``test_fast_path_matches_fitted_pick_model`` — the engine is a rewrite of
  M5's model for speed. If it drifts, every number downstream is fiction.
* ``test_vectorized_lineup_matches_optimal_lineup`` — same argument for M6's
  lineup rule.
* ``test_simulator_is_symmetric`` — me drafting exactly like the opponent model
  must land at zero. If it does not, the simulator has a seat bias and no policy
  comparison means anything.

The findings are pinned too, so a later change that quietly erases one fails.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ff_agent.config import (
    MY_TEAM_NAME, N_TEAMS, POSITION_MAXIMA, ROSTER_TOTAL,
)
from ff_agent.draft import build as DB
from ff_agent.draft import calibrate as CB
from ff_agent.draft import engine as E
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.draft import surrogate as SUR
from ff_agent.opponents import pickmodel as PM
from ff_agent.opponents import tendencies as TD
from ff_agent.season import lineup as LU
from ff_agent.season import schedule as SCH
from ff_agent.season import simulate as SIM

pytestmark = pytest.mark.espn


@pytest.fixture(scope="module")
def pool() -> PL.DraftPool:
    return PL.build_pool(2026, N_TEAMS)


@pytest.fixture(scope="module")
def lookups():
    return PM.build_lookups(
        TD.picks_with_consensus().filter(pl.col("canonical_id").is_not_null())
    )


@pytest.fixture(scope="module")
def sim(pool, lookups):
    tilt, reach = lookups
    return E.simulate_drafts(
        pool, DB.managers_for_slot(5), my_index=4, policy=POL.BestVOR(),
        tilt_lookup=tilt, reach=reach, n_sims=300, seed=7,
    )


# ── pool ─────────────────────────────────────────────────────────────────────
def test_no_player_appears_twice_in_the_pool(pool):
    """§0.2 is about people, not just ids resolving.

    ``model.project`` emitted one row per (player, prior NFL team) for anyone who
    changed teams mid-season; those fanned out through blend()'s join and again
    through the board's tier join, so 15 players reached the 2026 board FOUR
    times each. The engine marks pool INDICES taken, so a duplicate lets two
    fantasy teams draft the same person in one simulated draft.
    """
    d = pool.df.group_by("canonical_id").len().filter(pl.col("len") > 1)
    assert d.height == 0, f"duplicated players in the pool:\n{d}"


def test_pool_carries_kickers_and_defenses(pool):
    """M4's board is skill-only; without these, 12% of the draft does not exist."""
    counts = dict(pool.df.group_by("position").len().iter_rows())
    assert counts.get("K", 0) >= N_TEAMS
    assert counts.get("DST", 0) >= N_TEAMS


def test_pool_is_large_enough_for_a_full_draft(pool):
    assert pool.n > N_TEAMS * ROSTER_TOTAL


def test_every_pool_position_has_a_roster_slot(pool):
    assert set(pool.df["position"].to_list()) <= set(PL.POSITIONS)


def test_pool_is_sorted_by_vor(pool):
    """The engine's 'j-th best available at position p' is one cumulative sum
    ONLY because of this."""
    v = pool.vor
    assert np.all(v[:-1] >= v[1:] - 1e-6)


def test_kickers_and_defenses_get_real_nfl_byes(pool):
    """nflverse spells the Rams LA and ESPN spells them LAR. An unnormalised
    join silently hands those two a 17-week season."""
    kd = pool.df.filter(pl.col("value_source").str.starts_with("rank_calibrated"))
    assert kd.height > 0
    assert kd.filter(pl.col("bye_week") == 0).height == 0
    assert kd.filter(pl.col("team") == "LAR").height >= 1


def test_kickers_and_defenses_are_priced_late(pool):
    """26 of 26 kickers and 27 of 27 defenses went one per team, late."""
    kd = pool.df.filter(pl.col("position").is_in(["K", "DST"]))
    assert float(kd["consensus_fraction"].min()) > 0.5


def test_backtest_pool_uses_prior_season_not_todays_ownership():
    """ESPN's percent_owned is TODAY's. Reading it into a 2025 draft is
    lookahead of the laziest kind."""
    kd = PL.kdst_frame(2025, 8, source="prior_season")
    assert kd.height > 0
    assert set(kd["value_source"].unique()) == {"rank_calibrated_prior_season"}


# ── snake geometry (§5) ──────────────────────────────────────────────────────
def test_snake_gaps_match_the_spec_table():
    """§5: slot 1 is 17-then-1, slot 9 is 1-then-17, slot 5 is a flat 9."""
    def gaps(slot):
        t = E.turns_for_slot(slot, N_TEAMS, ROSTER_TOTAL)
        return sorted(set(np.diff(t).tolist()))

    assert gaps(1) == [1, 17]
    assert gaps(9) == [1, 17]
    assert gaps(5) == [9], "odd team counts make the middle slot uniquely smooth"


def test_every_slot_gets_one_pick_per_round():
    order = E.snake_order(N_TEAMS, ROSTER_TOTAL)
    for t in range(N_TEAMS):
        assert int((order == t).sum()) == ROSTER_TOTAL


# ── the engine is a faithful rewrite ─────────────────────────────────────────
def test_fast_path_matches_fitted_pick_model(pool, lookups):
    """THE GUARD. The engine reimplements M5's softmax in numpy for speed.

    If this drifts, the simulator is no longer running the model that was gated
    in M5, and every downstream number is fiction.
    """
    tilt, reach = lookups
    mgr = "Camden Sims"
    for pf in (0.05, 0.31, 0.72):
        slow = PM.pick_probabilities(
            pool.df.select("canonical_id", "position", "consensus_fraction", "name"),
            pf, mgr, tilt, reach,
        )
        row = E.tilt_array(tilt, [mgr])[0, E._phase_index(pf)]
        u = E.opponent_utility(
            pool.consensus_fraction, pool.pos_idx, pf, reach.get(mgr, 0.0), row
        ).astype(np.float64)
        u -= u.max()
        fast = np.exp(u) / np.exp(u).sum()
        assert np.abs(fast - slow).max() < 1e-6


def test_no_player_is_drafted_twice(sim):
    for row in sim.picks[:20]:
        assert len(set(row.tolist())) == len(row)


def test_every_team_fills_its_roster(sim):
    for t in range(sim.n_teams):
        assert sim.roster_of(t).shape[1] == ROSTER_TOTAL


def test_simulation_is_deterministic(pool, lookups):
    tilt, reach = lookups
    kw = dict(managers=DB.managers_for_slot(3), my_index=2, policy=POL.BestVOR(),
              tilt_lookup=tilt, reach=reach, n_sims=60, seed=99)
    a = E.simulate_drafts(pool, **kw)
    b = E.simulate_drafts(pool, **kw)
    assert np.array_equal(a.picks, b.picks)


def test_positional_offsets_move_the_aggregate_mix(pool, lookups):
    """M5's manager tilts are measured against the league rate, so they nearly
    cancel and CANNOT move the league's mix. This is the term that can."""
    tilt, reach = lookups
    kw = dict(managers=DB.managers_for_slot(1), my_index=-1, policy=None,
              tilt_lookup=tilt, reach=reach, n_sims=200, seed=5)
    base = CB.simulated_rates(E.simulate_drafts(pool, **kw))

    off = np.zeros((CB.N_PHASES, len(PL.POSITIONS)))
    off[0, PL.POS_INDEX["QB"]] = -3.0
    shifted = CB.simulated_rates(E.simulate_drafts(pool, pos_offsets=off, **kw))
    assert shifted[0, PL.POS_INDEX["QB"]] < base[0, PL.POS_INDEX["QB"]]


# ── policy legality ──────────────────────────────────────────────────────────
def test_my_roster_never_exceeds_position_maxima(sim):
    pos = np.array(sim.pool.df["position"].to_list())
    mine = pos[sim.my_roster()]
    for p, cap in POSITION_MAXIMA.items():
        assert int((mine == p).sum(axis=1).max()) <= cap


def test_my_roster_can_always_field_a_legal_lineup(sim):
    """§1's ten starters: QB2 RB2 WR2 TE1 FLEX1 DST1 K1."""
    pos = np.array(sim.pool.df["position"].to_list())
    mine = pos[sim.my_roster()]
    for need, p in ((2, "QB"), (2, "RB"), (2, "WR"), (1, "TE"), (1, "DST"), (1, "K")):
        assert int((mine == p).sum(axis=1).min()) >= need, p
    flex = sum((mine == p).sum(axis=1) for p in ("RB", "WR", "TE"))
    assert int(flex.min()) >= 6


def test_no_kicker_or_defense_before_the_last_two_rounds(sim):
    """§3.5: 'Still never before the last two rounds.' §10 calls a K before
    round 13 an alarm; this policy is stricter and that is deliberate."""
    pos = np.array(sim.pool.df["position"].to_list())
    cols = np.where(sim.team_at_pick == sim.my_index)[0]
    early = cols[cols < (ROSTER_TOTAL - 2) * N_TEAMS]
    taken = pos[sim.picks[:, early]]
    assert not (taken == "K").any()
    assert not (taken == "DST").any()


def test_qb_by_policy_forces_the_second_quarterback(pool, lookups):
    tilt, reach = lookups
    res = E.simulate_drafts(
        pool, DB.managers_for_slot(5), my_index=4, policy=POL.QBBy(pick=40),
        tilt_lookup=tilt, reach=reach, n_sims=200, seed=13,
    )
    pos = np.array(pool.df["position"].to_list())
    cols = np.where(res.team_at_pick == 4)[0]
    by_40 = cols[cols < 60]
    n_qb = (pos[res.picks[:, by_40]] == "QB").sum(axis=1)
    assert int(n_qb.min()) >= 2


def test_allowed_positions_forces_the_deficit_when_picks_run_out():
    counts = np.zeros((1, len(PL.POSITIONS)), dtype=np.int16)
    counts[0, PL.POS_INDEX["QB"]] = 2
    counts[0, PL.POS_INDEX["RB"]] = 2
    counts[0, PL.POS_INDEX["WR"]] = 2
    counts[0, PL.POS_INDEX["TE"]] = 1
    # one pick left, still no defense and no kicker: both cannot be had
    allowed = POL.allowed_positions(counts, round_no=17, rounds=17)
    assert allowed[0, PL.POS_INDEX["DST"]] or allowed[0, PL.POS_INDEX["K"]]
    assert not allowed[0, PL.POS_INDEX["WR"]]


# ── the surrogate ────────────────────────────────────────────────────────────
def test_vectorized_lineup_matches_optimal_lineup(pool):
    """THE OTHER GUARD. M6 proves strict-slots-then-FLEX is optimal for this
    slot structure; the numpy rewrite must be the same algorithm."""
    rng = np.random.default_rng(3)
    idx = {p: np.where(np.array(pool.df["position"].to_list()) == p)[0]
           for p in PL.POSITIONS}
    roster = np.array([
        np.concatenate([rng.choice(idx[p], n, replace=False) for p, n in
                        (("QB", 3), ("RB", 5), ("WR", 5), ("TE", 2), ("DST", 1), ("K", 1))])
        for _ in range(8)
    ])
    fast = SUR.lineup_stats(pool, roster, play_weeks=(3,))
    for i in range(roster.shape[0]):
        sub = pool.df[roster[i].tolist()].with_columns(
            (pl.col("blended_points") / 17).alias("weekly_points")
        ).filter(pl.col("bye_week") != 3)
        assert abs(LU.lineup_points(sub, "weekly_points") - fast.mean[i]) < 1e-3


def test_weekly_scale_divides_by_seventeen_games_not_sixteen():
    """The NFL plays 18 WEEKS and 17 GAMES — the bye is the eighteenth week.

    An earlier version divided by 16 for anyone with a bye, inflating every
    player by 6.25% and silently disagreeing with board/build.py,
    season/strength.py and projections/consensus.py, all of which use 17.
    Measured on 2024 actuals, 215 players logged games == 17.
    """
    pts = np.array([[170.0, 170.0]])
    bye = np.array([[7, 0]], dtype=np.int8)
    w = SUR.weekly_scale(pts, bye)
    assert w[0, 0] == pytest.approx(10.0)
    assert w[0, 1] == pytest.approx(10.0)
    assert w[0, 0] == w[0, 1], "a bye must not change the per-GAME rate"


def test_weekly_scale_agrees_with_the_rest_of_the_pipeline():
    from ff_agent.season import strength as ST
    from ff_agent.projections import consensus as C
    from ff_agent.board import build as BB

    assert SUR.GAMES == ST.GAMES == C.GAMES_IN_SEASON == BB.GAMES == 17


def test_free_bye_players_cost_nothing(pool):
    """§2.1: my fantasy byes are weeks 5 and 14, so an NFL bye there is free."""
    df = pool.df
    free = df.filter(pl.col("free_bye_week_5_or_14") & (pl.col("position") == "WR"))
    paid = df.filter(
        (~pl.col("free_bye_week_5_or_14")) & (pl.col("position") == "WR")
        & (pl.col("bye_week") == 9)
    )
    assert free.height and paid.height
    # a roster of free-bye players loses nothing across MY twelve weeks
    ids = {c: df["canonical_id"].to_list().index(c) for c in
           free["canonical_id"].to_list()[:1] + paid["canonical_id"].to_list()[:1]}
    fi, pi = list(ids.values())
    assert pool.bye_week[fi] in (5, 14)
    assert pool.bye_week[pi] == 9


def test_bilinear_interpolation_reproduces_grid_points():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0])
    z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            got = SUR._bilinear(xs, ys, z, np.array([x]), np.array([y]))
            assert got[0] == pytest.approx(z[i, j])


# ── the season simulator change M7 needed ────────────────────────────────────
def test_mean_uncertainty_is_opt_in_and_changes_nothing_when_absent():
    teams = SCH.games_played(2026)["team"].to_list()
    means = {t: 130.0 for t in teams}
    a = SIM.simulate(means, season=2026, n_sims=2000, seed=4)
    b = SIM.simulate(means, season=2026, n_sims=2000, seed=4, team_mean_sds=None)
    assert np.allclose(a.p_title, b.p_title)


def test_variance_helps_an_average_team_and_hurts_a_leader():
    """§2.4 says roster variance is good, without qualification. Measured on
    THIS schedule it is good below a crossover and costs seeding above it — and
    the top-2 crossover arrives first, so the first-round bye is what variance
    spends. §2.4 calls that bye worth about as much as everything else combined.
    """
    teams = SCH.games_played(2026)["team"].to_list()

    def run(delta, sd):
        means = {t: 130.0 for t in teams}
        means[MY_TEAM_NAME] = 130.0 + delta
        msds = {t: 4.5 for t in teams}
        msds[MY_TEAM_NAME] = sd
        r = SIM.simulate(means, season=2026, n_sims=20_000, seed=99,
                         team_mean_sds=msds)
        i = r.teams.index(MY_TEAM_NAME)
        return float(r.p_title[i]), float(r.p_top2[i])

    avg_low, _ = run(0, 1.0)
    avg_high, _ = run(0, 12.0)
    assert avg_high > avg_low * 1.15, "variance should clearly help an average team"

    _, lead_top2_low = run(30, 1.0)
    _, lead_top2_high = run(30, 12.0)
    assert lead_top2_high < lead_top2_low, "variance should cost a leader the bye"


# ── the simulator has no seat bias ───────────────────────────────────────────
def test_simulator_is_symmetric(pool, lookups):
    """THE THIRD GUARD. If I draft exactly like the opponent model, my roster
    must be worth about what theirs are. Any bias here would masquerade as a
    policy edge."""
    tilt, reach = lookups
    off = np.zeros((CB.N_PHASES, len(PL.POSITIONS)))
    run = DB.run_slot(
        pool, 5, POL.ModelSeat(tilt=tilt, reach=reach, manager=DB.MY_MANAGER,
                               pos_offsets=off),
        tilt, reach, off, n_sims=1500,
    )
    assert abs(float(run.delta.mean())) < 6.0


def test_a_consensus_only_policy_still_beats_the_opponent_model(pool, lookups):
    """Pins the honest caveat rather than leaving it as prose.

    ``best_consensus`` has ZERO board edge by construction — it ranks by the same
    superflex ECR the opponent model is built on. Whatever margin it shows is
    bought entirely from M5's opponents drafting with an 8-pick spread while it
    does not, and it is most of what ``best_vor`` shows. Absolute title
    probabilities out of this simulator are therefore optimistic.
    """
    tilt, reach = lookups
    off = np.zeros((CB.N_PHASES, len(PL.POSITIONS)))
    consensus = DB.run_slot(pool, 5, POL.BestConsensus(), tilt, reach, off, n_sims=1500)
    vor = DB.run_slot(pool, 5, POL.BestVOR(), tilt, reach, off, n_sims=1500)
    c, v = float(consensus.delta.mean()), float(vor.delta.mean())
    assert c > 10.0, "the noise-exploitation margin is large and must stay visible"
    assert abs(v - c) < c, "board edge must not be the majority of the margin"


# ── wiring ───────────────────────────────────────────────────────────────────
def test_managers_for_slot_places_me_correctly():
    for slot in range(1, N_TEAMS + 1):
        m = DB.managers_for_slot(slot)
        assert len(m) == N_TEAMS
        assert len(set(m)) == N_TEAMS
        assert m[slot - 1] == DB.MY_MANAGER


def test_every_manager_maps_to_a_fantasy_team():
    for m in DB.managers_for_slot(1):
        assert DB.team_name_for(m)


def test_team_names_match_the_league_schedule():
    """§1 spells a team with a curly apostrophe; the schedule normalises it.

    Missing that produced a KeyError inside the season simulator only AFTER all
    twenty-seven slot runs had finished, which is the expensive way to find out.
    """
    DB.assert_teams_match_schedule(2026)


def test_surrogate_grid_covers_everything_the_simulator_produces(pool, lookups):
    """Bilinear interpolation clamps SILENTLY at the grid edge.

    The first run fixed the grid at -30..+30 while draft outcomes reached +49,
    scoring 0.21 absolute error on P(title) that looked exactly like an
    interpolation result. Samples inside the grid were accurate to 0.003.
    """
    tilt, reach = lookups
    off = np.zeros((CB.N_PHASES, len(PL.POSITIONS)))
    runs = [DB.run_slot(pool, s, POL.BestVOR(), tilt, reach, off, n_sims=400)
            for s in (1, 9)]
    g = DB.grid_deltas(runs)
    for r in runs:
        assert float(r.delta.min()) >= g[0]
        assert float(r.delta.max()) <= g[-1]


def test_uncapped_vor_takes_the_maximum_number_of_quarterbacks(pool, lookups):
    """Pins M7's most actionable finding.

    Unconstrained, ``best_vor`` drafts FOUR quarterbacks in the first five
    rounds, every time — because the board really does rank them there. VOR is
    measured against the STARTER replacement (QB18) and so prices a third and
    fourth QB as if they would start, in a league that starts two. §3.6 wants
    2-3. If a later board change removes this, the test should fail and the
    finding be revisited, not quietly deleted.
    """
    tilt, reach = lookups
    off = np.zeros((CB.N_PHASES, len(PL.POSITIONS)))
    r = DB.run_slot(pool, 5, POL.BestVOR(), tilt, reach, off, n_sims=400)
    pos = np.array(pool.df["position"].to_list())
    mine = pos[r.result.my_roster()]
    assert float((mine == "QB").sum(axis=1).mean()) >= 3.5


def test_qb_cap_is_honoured(pool, lookups):
    tilt, reach = lookups
    off = np.zeros((CB.N_PHASES, len(PL.POSITIONS)))
    r = DB.run_slot(pool, 5, POL.BestVOR(caps={"QB": 2}), tilt, reach, off, n_sims=400)
    pos = np.array(pool.df["position"].to_list())
    mine = pos[r.result.my_roster()]
    assert int((mine == "QB").sum(axis=1).max()) == 2
