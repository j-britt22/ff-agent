"""§1 scoring rules, loaded from ESPN rather than hand-transcribed."""
import pytest

from ff_agent.config import LAST_SEASON, SEASON
from ff_agent.scoring import rules as R


def test_loaded_rules_match_spec_for_the_target_season():
    r = R.load_rules(SEASON)
    R.assert_matches_spec(r)


def test_last_season_uses_the_same_ruleset_as_the_draft_season():
    """2025 must match 2026 or the exact-match test validates the wrong league."""
    assert LAST_SEASON in R.SPEC_SEASONS
    assert R.load_rules(LAST_SEASON) == R.load_rules(SEASON)


def test_the_two_unusual_rules_are_present():
    r = R.load_rules(SEASON)
    assert r["SKD"] == -1.0    # §3.3 sack taken
    assert r["RA"] == 0.05     # §3.4 rushing attempt


def test_six_point_passing_td_and_half_ppr():
    r = R.load_rules(SEASON)
    assert r["PTD"] == 6.0 and r["REC"] == 0.5 and r["PY"] == 0.04


def test_distance_weighted_field_goals():
    r = R.load_rules(SEASON)
    assert (r["FG0"], r["FG40"], r["FG50"], r["FG60"]) == (3.0, 4.0, 5.0, 6.0)
    assert r["FGM"] == -1.0


def test_zero_valued_dst_buckets_are_absent_not_scored():
    """§1 lists 18-21, 22-27 and 300-349 as 0. ESPN omits zero rules entirely."""
    r = R.load_rules(SEASON)
    for abbr in R.MUST_BE_ABSENT:
        assert abbr not in r


def test_spec_mismatch_is_blocking():
    bad = dict(R.SPEC_RULES)
    bad["SKD"] = 0.0
    with pytest.raises(R.ScoringRulesError) as e:
        R.assert_matches_spec(bad)
    assert "SKD" in str(e.value)


def test_a_reintroduced_zero_bucket_is_blocking():
    bad = dict(R.SPEC_RULES)
    bad["PA22"] = -1.0          # ESPN's own default, which this league overrode
    with pytest.raises(R.ScoringRulesError) as e:
        R.assert_matches_spec(bad)
    assert "PA22" in str(e.value)


def test_scoring_changed_after_2024():
    """2023/2024 additionally scored completions and incompletions. Recorded
    fantasy points are therefore NOT comparable across that boundary."""
    old = R.load_rules(2024)
    assert old.get("PC") == 0.25 and old.get("INC") == -0.1
    assert "PC" not in R.load_rules(SEASON)
    assert 2024 not in R.SPEC_SEASONS
