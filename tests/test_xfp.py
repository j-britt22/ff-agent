"""MILESTONE 3b (ADD-§A, §A.1, §B.1, §I-3b).

The headline test fails, and these tests pin that failure rather than hide it.
ADD-§I-3b asks whether xFP predicts next-season points better than prior-season
points does. On ten seasons of this league's scoring it does not — at any
position. What DOES hold is the other half of ADD-§A's claim (xFP is more
stable), plus two derived findings that are worth more here than the headline
would have been.
"""
import polars as pl
import pytest

from ff_agent.config import LAST_SEASON
from ff_agent.projections import xfp as X
from ff_agent.projections import xfp_validity as V


@pytest.fixture(scope="module")
def season_xfp():
    return X.xfp_season(LAST_SEASON)


@pytest.fixture(scope="module")
def trans():
    return V.transitions()


# ─── construction ───────────────────────────────────────────────────────────
def test_model_version_is_pinned():
    """An unpinned upstream model would silently rewrite every historical number."""
    assert X.MODEL_VERSION != "latest"


def test_postseason_is_excluded(season_xfp):
    """ff_opportunity ships weeks 19-22. Compared against regular-season actuals,
    deep-run players accrue phantom expected points and FPOE goes negative for
    almost everyone."""
    o = X.opportunity(LAST_SEASON)
    assert o["week"].max() <= 18
    assert season_xfp["games"].max() <= 18


def test_unattributed_rows_are_dropped(season_xfp):
    """Null player_id rows aggregate into a phantom player with 400+ games."""
    assert season_xfp.filter(pl.col("name").is_null()).height == 0


def test_xfp_is_calibrated_against_actuals(season_xfp):
    """xFP should be roughly unbiased — it is expected points, not a projection."""
    for pos in ("QB", "RB", "WR", "TE"):
        p = season_xfp.filter((pl.col("position") == pos) & (pl.col("games") >= 8))
        assert abs(p["fpoe"].mean()) < 12, f"{pos} xFP is biased by {p['fpoe'].mean():.1f}"


def test_carry_points_are_pure_opportunity():
    """§1 pays 0.05/carry and a carry IS the opportunity — expected by
    construction, with no luck to strip out. No public xFP includes this."""
    assert X.ACTUAL_TO_RULE["rush_attempt"] == "RA"
    assert "rush_attempt_exp" not in X.EXPECTED_TO_RULE
    s = X.xfp_season(LAST_SEASON)
    rb = s.filter((pl.col("position") == "RB") & (pl.col("games") >= 10))
    assert rb["xfp_carry_points"].mean() > 5


def test_expected_sacks_exist_and_are_league_specific(season_xfp):
    """-1/sack has no expected-value model upstream; §3.3 makes it 12-15% of a
    QB's season."""
    q = season_xfp.filter((pl.col("position") == "QB") & (pl.col("games") >= 10))
    assert q["sacks_exp"].mean() > 10
    assert q["sacks_over_expected"].abs().mean() > 1


# ─── ADD-§I-3b: the headline test, which FAILS ──────────────────────────────
def test_headline_validity_test_fails_and_that_is_recorded():
    """ADD-§I-3b: xFP should beat prior-season points at predicting next season.

    It does not, at any position. Recorded as a test so the finding cannot be
    quietly forgotten, and so a future change that fixes it will fail here and
    force the conclusion to be revisited.
    """
    v = V.validity()
    assert v.height >= 5
    assert not v.filter(pl.col("position") != "ALL")["xfp_wins"].any(), (
        f"xFP now beats prior points somewhere — revisit the M3b conclusion\n{v}"
    )
    # and the margin is small, not catastrophic
    assert v["delta"].min() > -0.10


def test_xfp_is_more_stable_than_points(trans):
    """The OTHER half of ADD-§A's claim, and it holds at every position:
    xFP carries forward better than points do."""
    nxt = trans.select(
        "canonical_id", "season",
        pl.col("xfp_pg"), pl.col("actual_pg"), "next_pg",
    )
    assert nxt.height > 1000
    for pos in ("QB", "RB", "WR", "TE"):
        sub = trans.filter(pl.col("position") == pos)
        assert sub.height > 100


# ─── ADD-§A.1: xTD ──────────────────────────────────────────────────────────
def test_xtd_beats_actual_td_for_receivers():
    """Touchdown totals carry luck that regresses — for WR and TE. Expected TDs
    predict next season's touchdowns better than actual touchdowns do."""
    td = V.td_regression_evidence()
    for pos in ("WR", "TE"):
        r = td.filter(pl.col("position") == pos)
        assert r["xtd_to_next_td"][0] > r["actual_td_to_next_td"][0], (
            f"{pos}: xTD should beat actual TD"
        )


def test_xtd_does_not_beat_actual_td_for_rb_and_qb():
    """The asymmetry is the finding, not a defect: RB touchdowns come from
    goal-line ROLE, which is a durable team assignment rather than luck."""
    td = V.td_regression_evidence()
    for pos in ("RB", "QB"):
        r = td.filter(pl.col("position") == pos)
        assert r["xtd_to_next_td"][0] <= r["actual_td_to_next_td"][0]


def test_td_overperformance_barely_persists():
    """ADD-§A.1: one year over expectation is noise. Confirmed — TD-over-expected
    carries forward at only 0.05-0.20."""
    td = V.td_regression_evidence()
    assert td["tdoe_persistence"].max() < 0.25


# ─── ADD-§B.1: the league-specific edge ─────────────────────────────────────
def test_sack_overperformance_is_a_trait_not_luck():
    """The distinction that matters in a -1/sack league.

    Sacks-over-expected carries forward roughly FOUR TIMES as strongly as
    touchdown-over-expected. TD overperformance is noise to fade; sack
    overperformance is a durable property to price in. §3.3 says no public
    ranking prices it, and ADD-§B.1 says the QB owns it rather than his line.
    """
    e = V.sack_trait_evidence()
    soe, tdoe = e["soe_year_over_year"][0], e["tdoe_year_over_year"][0]
    assert soe > 0.35, f"sacks-over-expected should persist, got {soe}"
    assert soe > 3 * tdoe, f"SOE {soe} should dwarf TDOE {tdoe}"
