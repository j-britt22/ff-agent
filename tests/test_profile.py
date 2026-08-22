"""Reading a league's shape from ESPN instead of hardcoding it.

CLAUDE.md §1 pins this project to one 9-team, 2-QB league. That is correct for
its owner, and useless to anybody else. These tests assert that detection
independently reproduces every one of §1's constants — which is simultaneously
a regression guard on the detection AND the evidence that the engine is not
actually specific to this league, only its config was.
"""

from __future__ import annotations

import pytest

from ff_agent import config as c
from ff_agent.live import profile as PR

pytestmark = pytest.mark.espn


@pytest.fixture(scope="module")
def prof():
    return PR.detect(c.SEASON)


def test_detection_reproduces_every_section_1_constant(prof):
    """The load-bearing test. §1 was transcribed by a human from ESPN's UI;
    this reads the same facts from the API and they must agree exactly.
    """
    assert prof.n_teams == c.N_TEAMS
    assert prof.starter_slots == c.STARTER_SLOTS
    assert prof.roster_total == c.ROSTER_TOTAL
    assert prof.bench_slots == c.BENCH_SLOTS
    assert prof.ir_slots == c.IR_SLOTS
    assert prof.position_maxima == c.POSITION_MAXIMA
    assert set(prof.my_bye_weeks) == set(c.MY_BYE_WEEKS)
    assert prof.playoff_teams == c.PLAYOFF_TEAMS
    assert prof.is_keeper is c.IS_KEEPER_LEAGUE


def test_my_team_is_identified_by_swid_not_by_name(prof):
    """Names are not identity — §1's "A Chane Reaction" is called "TBD" now.

    The SWID cookie is the logged-in account, so it survives a rename.
    """
    assert prof.my_team_name == c.MY_TEAM_NAME
    assert prof.my_manager == "Jordan Britt"


def test_the_position_caps_are_real_and_not_a_fallback(prof):
    """M7: without caps the simulated league drafted eight QBs and no TE, worth
    ~50 points a week. A cap silently defaulting to roster_total is that bug.
    """
    assert prof.position_maxima["QB"] == 4 < prof.roster_total
    assert prof.position_maxima["TE"] == 3 < prof.roster_total
    assert not any("could not read position limits" in w for w in prof.warnings)


def test_this_league_is_recognised_as_superflex(prof):
    """M3: 2 starting QBs means superflex ECR is the only valid consensus.
    Scoring a 2-QB league against standard ranks manufactures fake reaches."""
    assert prof.qb_slots == 2 and prof.is_superflex


def test_an_idp_league_is_refused_rather_than_mis_scored():
    """§10: fail loudly. The projections cover QB/RB/WR/TE/K/D-ST only, so a
    league starting linebackers would get a silently wrong board.
    """
    assert "LB" in PR.UNSUPPORTED_SLOTS
    assert "DP" in PR.UNSUPPORTED_SLOTS
    assert "FLEX" not in PR.UNSUPPORTED_SLOTS


def test_the_flex_is_translated_from_espns_own_name(prof):
    """ESPN calls it RB/WR/TE; §1 calls it FLEX. A league with a flex must not
    lose it in translation — it is a full starting slot."""
    assert PR.SLOT_ALIASES["RB/WR/TE"] == "FLEX"
    assert prof.starter_slots["FLEX"] == 1


def test_the_profile_is_readable_by_a_human(prof):
    s = prof.summary()
    assert prof.my_team_name in s and str(prof.n_teams) in s


def test_the_draft_slot_is_detected_when_espn_has_published_it(prof):
    """§5 treats the slot as unknowable before ~T-60, but ESPN's own
    draftSettings.pickOrder is just sitting in the settings payload whenever
    the commissioner has set it -- which can be well before T-60. `cli gui
    --auto` should never need `--slot` once this is populated.
    """
    if prof.my_slot is None:
        pytest.skip("this league has no published draft order right now")
    assert 1 <= prof.my_slot <= prof.n_teams


def test_a_detected_slot_seats_me_at_the_right_index():
    from ff_agent.live import profile as PR

    prof = PR.detect()
    if prof.my_slot is None:
        pytest.skip("this league has no published draft order right now")
    seats = PR.managers_for_slot(prof, prof.my_slot)
    assert seats[prof.my_slot - 1] == prof.my_manager


# ── the seat order, not just my seat ─────────────────────────────────────────
# The first version of `_detect_slot` read `draftSettings.pickOrder` — the WHOLE
# round-1 order — and returned only my own index from it. Every other seat was
# then filled in ESPN's team_id order, so the GUI named the wrong manager on the
# clock and `record_index` attributed each pick to that wrong manager. My own
# slot was correct, which is precisely why it looked right. These pin the whole
# order, offline, against a stub.


class _StubProfile:
    """Enough of a LeagueProfile to exercise the seating, with no network."""

    @staticmethod
    def make(my_slot, draft_order):
        return PR.LeagueProfile(
            season=2026, league_id="x", league_name="stub", n_teams=4,
            my_team_id=1, my_team_name="mine", my_manager="me",
            managers=["me", "b", "c", "d"], team_ids=[1, 2, 3, 4],
            starter_slots={"QB": 1}, bench_slots=1, ir_slots=0, roster_total=2,
            position_maxima={}, playoff_teams=2, regular_season_weeks=14,
            my_bye_weeks=(), is_keeper=False, uses_faab=False,
            my_slot=my_slot, draft_order=draft_order,
        )


def test_seats_follow_espns_published_order_not_team_id_order():
    # ESPN says round 1 goes 3, 1(me), 4, 2 — which is NOT team_id order.
    p = _StubProfile.make(my_slot=2, draft_order=[3, 1, 4, 2])
    assert PR.managers_for_slot(p, 2) == ["c", "me", "d", "b"]
    assert p.warnings == []


def test_the_old_team_id_shortcut_would_have_failed_this():
    """Guard the guard: the fallback really does give a different answer."""
    p = _StubProfile.make(my_slot=2, draft_order=[3, 1, 4, 2])
    p.draft_order = []                      # simulate "no published order"
    assert PR.managers_for_slot(p, 2) == ["b", "me", "c", "d"]


def test_a_slot_override_is_obeyed_but_says_the_seats_are_guesses():
    p = _StubProfile.make(my_slot=2, draft_order=[3, 1, 4, 2])
    seats = PR.managers_for_slot(p, 4)
    assert seats[3] == "me"                 # the human's override wins (§6)
    assert any("overrides ESPN's published order" in w for w in p.warnings)


def test_a_published_order_that_contradicts_itself_is_refused_loudly():
    p = _StubProfile.make(my_slot=2, draft_order=[3, 4, 1, 2])   # me at seat 3
    seats = PR.managers_for_slot(p, 2)
    assert seats[1] == "me"                 # still seats me where asked
    assert any("does not seat you where expected" in w for w in p.warnings)


def test_detected_order_matches_the_live_league(prof):
    """Against the real league: every seat, not just mine."""
    if prof.my_slot is None or not prof.draft_order:
        pytest.skip("this league has no published draft order right now")
    seats = PR.managers_for_slot(prof, prof.my_slot)
    assert len(seats) == prof.n_teams
    assert seats[prof.my_slot - 1] == prof.my_manager
    by_id = dict(zip(prof.team_ids, prof.managers))
    assert seats == [by_id[t] for t in prof.draft_order]
    assert sorted(seats) == sorted(prof.managers)   # a permutation, no drops
