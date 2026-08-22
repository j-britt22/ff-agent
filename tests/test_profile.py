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
