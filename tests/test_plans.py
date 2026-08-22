"""M8 — the nine slot plans (§5, §11 step 8).

The load-bearing ones:

* ``test_survival_excludes_my_own_pick`` — the pass-or-take number must be a
  counterfactual. Counting sims where I took the player makes every player I
  want read "survives 0%", which is true and useless.
* ``test_reseed_actually_changes_the_draft`` — the §11 gate compares between-slot
  divergence against re-seed noise. ``run_slot`` derives its default seed from
  the slot, so a naive re-run reproduces the draft exactly and the gate passes
  with a ratio of infinity. It did, before this test existed.
* ``test_t60_load_needs_no_computation`` — §5's whole design constraint.
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
import pytest

from ff_agent.config import ARTIFACTS_DIR, N_TEAMS, ROSTER_TOTAL
from ff_agent.draft import build as DB
from ff_agent.draft import calibrate as CB
from ff_agent.draft import engine as E
from ff_agent.draft import plan_build as PB
from ff_agent.draft import plans as PN
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.draft import scenarios as SC
from ff_agent.draft import sheet as SH
from ff_agent.opponents import pickmodel as PM
from ff_agent.opponents import tendencies as TD

pytestmark = pytest.mark.espn


@pytest.fixture(scope="module")
def pool():
    return PL.build_pool(2026, N_TEAMS)


@pytest.fixture(scope="module")
def lookups():
    return PM.build_lookups(
        TD.picks_with_consensus().filter(pl.col("canonical_id").is_not_null())
    )


@pytest.fixture(scope="module")
def run(pool, lookups):
    tilt, reach = lookups
    off = np.zeros((CB.N_PHASES, len(PL.POSITIONS)))
    return DB.run_slot(pool, 5, POL.NeedWeighted(caps={"QB": 3}), tilt, reach,
                       off, n_sims=1200)


@pytest.fixture(scope="module")
def plan(run):
    return PN.build_slot_plan(run.result, 5, run.value.mean)


# ── availability mechanics ───────────────────────────────────────────────────
def test_gone_matrix_records_the_pick_each_player_went_at(run):
    res = run.result
    gone = PN.gone_matrix(res)
    total = res.picks.shape[1]
    for s in range(5):
        for k in range(total):
            assert gone[s, res.picks[s, k]] == k + 1
    undrafted = gone == total + 1
    assert int(undrafted[0].sum()) == res.pool.n - total


def test_availability_falls_monotonically_through_the_draft(run):
    gone = PN.gone_matrix(run.result)
    a1 = PN.availability(gone, 1)
    a80 = PN.availability(gone, 80)
    a150 = PN.availability(gone, 150)
    assert a1.mean() >= a80.mean() >= a150.mean()
    assert a1.max() == pytest.approx(1.0)


def test_survival_excludes_my_own_pick(run):
    """The pass-or-take question is 'if I DON'T take him, does he come back?'

    Counting the sims where I took him makes every player the policy wants read
    as vanishing — true, and about my own pick rather than the draft.
    """
    res = run.result
    gone = PN.gone_matrix(res)
    turns = E.turns_for_slot(5, res.n_teams, res.rounds)
    t, nxt = int(turns[1]), int(turns[2])

    naive = PN.conditional_survival(gone, t, nxt)
    honest = PN.conditional_survival(gone, t, nxt, res.picks[:, t - 1])

    mine = res.picks[:, t - 1]
    top = int(np.bincount(mine, minlength=res.pool.n).argmax())
    assert naive[top] < 0.5
    assert np.isnan(honest[top]) or honest[top] > naive[top]


def test_a_player_i_never_pass_on_is_flagged_take_now(plan):
    """NaN survival means the counterfactual never happened — the strongest
    take-now signal there is. Compared against a threshold it is False."""
    for turn in plan.per_turn:
        for c in turn["shortlist"]["candidates"]:
            if c["never_passed_on"]:
                assert c["take_now"], c["name"]
                assert c["survival_to_next_turn"] is None


def test_shortlist_shows_players_i_would_actually_take(plan):
    """Ranking by VOR alone fills the sheet with the un-takeable QB glut."""
    mid = plan.per_turn[5]["shortlist"]["candidates"]
    assert mid, "a turn with no candidates is a broken sheet"
    assert all(c["p_i_take_him_here"] >= PN.CANDIDATE_FLOOR for c in mid)
    assert mid == sorted(mid, key=lambda c: -c["p_i_take_him_here"])


def test_slipping_away_never_lists_a_below_replacement_player(plan):
    for turn in plan.per_turn:
        for c in turn["shortlist"]["slipping_away"]:
            assert c["vor"] > 0, c["name"]


def test_best_available_tier_probabilities_are_a_distribution(plan):
    for turn in plan.per_turn:
        for entry in turn["best_available_by_position"]:
            total = sum(entry["tiers"].values())
            assert total == pytest.approx(entry["p_any"], abs=0.02)


# ── branch points ────────────────────────────────────────────────────────────
def test_branches_are_only_reported_when_the_call_changes(plan):
    """A branch that does not change the recommendation is not a branch."""
    for turn in plan.per_turn:
        for b in turn["branches"]:
            assert b["if_true_take"] != b["if_false_take"]
            assert PN.MIN_BRANCH_PROB <= b["p_condition"] <= 1 - PN.MIN_BRANCH_PROB


# ── QB scenarios ─────────────────────────────────────────────────────────────
def test_qb_scenarios_order_by_how_hard_quarterbacks_go(pool, lookups):
    """cold < shipped < hot on early-QB pressure, by construction: 2023-24 were
    ONE-QB seasons and 2025 is the only 2-QB draft the league has run."""
    tilt, reach = lookups
    scen = SC.fit_all(pool, DB.managers_for_slot(1), tilt, reach, ROSTER_TOTAL,
                      n_sims=400)
    p = SC.qb_pressure(scen)
    m = dict(zip(p["scenario"].to_list(), p["early_qb_multiplier"].to_list()))
    assert m["qb_cold"] < m["shipped"] < m["qb_hot"]


# ── §11 step 8 ───────────────────────────────────────────────────────────────
def test_reseed_actually_changes_the_draft(pool, lookups):
    """``run_slot`` derives its default seed from the slot number.

    The gate divides between-slot divergence by re-seed noise. Re-running a slot
    without changing the seed reproduces it EXACTLY, making the noise zero and
    the ratio infinite — which is how the gate first "passed".
    """
    tilt, reach = lookups
    off = np.zeros((CB.N_PHASES, len(PL.POSITIONS)))
    a = DB.run_slot(pool, 5, POL.BestVOR(), tilt, reach, off, n_sims=300)
    b = DB.run_slot(pool, 5, POL.BestVOR(), tilt, reach, off, n_sims=300,
                    seed=PB.RESEED)
    assert not np.array_equal(a.result.picks, b.result.picks)


def test_divergence_is_zero_for_a_plan_against_itself(plan):
    m = PN.shape_matrix({5: plan})
    assert PN.divergence(m[0], m[0]) == pytest.approx(0.0)


def test_snake_gaps_differ_between_wheel_and_middle_slots():
    """§5: slot 5 is a flat 9; the wheels alternate 1 and 17. If the plans do
    not differ, this is why they should have."""
    g = {s: sorted(set(np.diff(E.turns_for_slot(s, N_TEAMS, ROSTER_TOTAL)).tolist()))
         for s in (1, 5, 9)}
    assert g[5] == [9]
    assert g[1] == g[9] == [1, 17]


# ── the T-60 drill (§5) ──────────────────────────────────────────────────────
def test_t60_load_needs_no_computation():
    """§5: "Design so that hour needs zero computation."

    Skipped when the plans have not been built; it is a property of the shipped
    artifact, not of the code.
    """
    if not (ARTIFACTS_DIR / "plan_6.json").exists():
        pytest.skip("run `cli plans` first — this checks the shipped artifact")
    t = time.perf_counter()
    plan = PB.load(6)
    elapsed = time.perf_counter() - t
    assert elapsed < 5.0, f"§5's budget is 5 seconds; took {elapsed:.1f}s"
    assert plan["slot"] == 6
    assert len(plan["pick_numbers"]) == ROSTER_TOTAL
    assert SH.render(plan)


def test_t60_load_fails_loudly_rather_than_computing():
    """A missing plan must not trigger a rebuild while I am on the clock."""
    with pytest.raises(FileNotFoundError, match="BEFORE draft day"):
        PB.load(99)


def test_every_slots_picks_are_that_slots_picks():
    for slot in range(1, N_TEAMS + 1):
        turns = E.turns_for_slot(slot, N_TEAMS, ROSTER_TOTAL)
        assert turns[0] == slot
        assert len(turns) == ROSTER_TOTAL
        assert len(set(turns.tolist())) == ROSTER_TOTAL


def test_sheet_renders_without_a_plan_file(plan):
    """The renderer must not require the artifact to exist."""
    d = plan.to_dict()
    d["shared"] = {
        "season": 2026, "policy": "test",
        "qb_count_decision": {"decision": 3, "runner_up": 2,
                              "margin_over_runner_up": 0.01},
    }
    d["scenario_divergence"] = []
    out = SH.render(d, max_turns=3)
    assert "DRAFT SLOT 5" in out
    assert "NEVER auto-submit" in out
