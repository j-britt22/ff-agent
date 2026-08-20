"""Milestone 5 — opponent model (§7.5, §2.3).

The league changed shape underneath itself and these tests exist mostly to stop
that being papered over: 2023 and 2024 were ONE-QB leagues, team counts ran
8/10/8/9, and the four managers §2.3 weights most heavily are not the ones with
the most evidence.
"""
import polars as pl
import pytest

from ff_agent.config import DOUBLE_UP_MANAGERS
from ff_agent.opponents import build as OB
from ff_agent.opponents import evaluate as E
from ff_agent.opponents import history as H
from ff_agent.opponents import tendencies as TD

pytestmark = pytest.mark.espn


@pytest.fixture(scope="module")
def hist():
    return H.draft_history()


@pytest.fixture(scope="module")
def picks():
    return TD.picks_with_consensus()


# ─── the format change ──────────────────────────────────────────────────────
def test_only_2025_was_a_two_qb_season():
    assert 2025 in H.TWO_QB_SEASONS
    assert 2023 not in H.TWO_QB_SEASONS and 2024 not in H.TWO_QB_SEASONS


def test_format_change_moved_qb_timing_enormously(hist):
    """First QB off the board: pick 16, then 35, then 7. Pre-2025 QB behaviour is
    not thin evidence — it is evidence about a different game."""
    f = H.format_shift_evidence(hist)
    by = {r["season"]: r for r in f.to_dicts()}
    assert by[2024]["first_qb_pick"] > 30
    assert by[2025]["first_qb_pick"] < 10
    assert by[2025]["first_qb_fraction"] < by[2023]["first_qb_fraction"]
    assert by[2025]["first_qb_fraction"] < by[2024]["first_qb_fraction"]


def test_picks_are_normalised_across_changing_team_counts(hist):
    assert set(hist["n_teams"].unique().to_list()) == {8, 10}
    assert hist["pick_fraction"].min() > 0
    assert hist["pick_fraction"].max() <= 1.0


def test_manager_aliases_are_explicit_not_fuzzy():
    """§1 says 'R. Sharrett'; ESPN says 'Rayne Sharrett'."""
    assert H.canonical_manager("R. Sharrett") == "Rayne Sharrett"
    assert H.canonical_manager("Rayne Sharrett") == "Rayne Sharrett"


# ─── evidence, and its absence ──────────────────────────────────────────────
def test_every_current_manager_resolves(hist):
    cov = H.manager_coverage(hist)
    known = set(cov["manager"].to_list())
    for m in DOUBLE_UP_MANAGERS:
        assert H.canonical_manager(m) in known


def test_a_double_up_manager_has_only_one_draft(hist):
    """Kylie Leahy is faced twice — two-thirds of the schedule rides on four
    managers, and she has 17 picks of history."""
    cov = H.manager_coverage(hist)
    k = cov.filter(pl.col("manager") == "Kylie Leahy")
    assert k.height == 1
    assert int(k["n_drafts"][0]) == 1
    assert k["faced_twice"][0] is True


def test_qb_confidence_is_labelled_low_everywhere(hist):
    """Nobody has more than one 2-QB draft, and the model must say so."""
    cov = H.manager_coverage(hist)
    assert (cov["n_drafts_two_qb"] <= 1).all()
    assert cov.filter(pl.col("qb_confidence").str.contains("moderate")).height == 0


# ─── shrinkage and format weighting ─────────────────────────────────────────
def test_tendencies_are_shrunk_toward_the_league(picks):
    """With 17-49 picks, a raw rate would be confident nonsense."""
    t = TD.positional_tendency(picks)
    thin = t.filter(pl.col("phase_picks") < 10)
    assert thin.height > 0
    assert (thin["rate"] - thin["league_rate"]).abs().max() < 0.12
    assert (t["evidence_weight"] <= 1.0).all()


def test_format_matched_season_carries_more_weight():
    assert TD.FORMAT_MATCH_WEIGHT > 1.0
    assert TD.season_weight(2025, target_two_qb=True) > TD.season_weight(2024, target_two_qb=True)


def test_consensus_baseline_matches_each_season_format(picks):
    """Scoring a 1-QB draft against superflex ranks would manufacture fake
    reaches at QB — the format difference posing as a personality trait."""
    ranked = picks.filter(pl.col("ecr_type").is_not_null())
    by = ranked.group_by("season", "ecr_type").len().sort("season")
    d = {r["season"]: r["ecr_type"] for r in by.to_dicts()}
    assert d[2023] == "ro" and d[2024] == "ro"
    assert d[2025] == "rsf"
    # the unranked residue is D/ST and kickers, which skill-position ECR omits
    unranked = picks.filter(pl.col("ecr_type").is_null())
    assert set(unranked["position"].unique().to_list()) <= {"DST", "K", None}


# ─── §11 GATE ───────────────────────────────────────────────────────────────
def test_beats_superflex_adp_at_predicting_our_own_draft():
    """THE GATE. Fit on 2023-24, predict 2025, with no lookahead.

    The baseline is SUPERFLEX ADP, not standard — §7.5 already rules standard out
    for a 2-QB league, so beating it would prove nothing.
    """
    r = E.strict_holdout(2025)
    d = {row["model"]: row for row in r.to_dicts()}
    base = d["superflex ADP only"]
    model = d["ADP + manager terms"]
    assert model["mean_log_prob"] > base["mean_log_prob"], r
    assert model["top1_accuracy"] >= base["top1_accuracy"]


def test_the_gain_is_uneven_and_that_is_recorded():
    """Two managers get WORSE, the worst being Jordan Britt, whose 1-QB drafts
    give no hint that he would take the first QB at pick 7. A manager's own
    history can mislead across a format change — which is why the shipped model
    weights the format-matched season 3x."""
    b = E.per_manager_breakdown(2025)
    assert b.filter(pl.col("gain") < 0).height >= 1
    assert b["gain"].mean() > 0


def test_a_manager_with_no_training_history_falls_back_to_adp():
    """Kylie Leahy has no pre-2025 drafts, so a 2023-24 fit gives her no terms.
    The model must degrade to pure ADP rather than invent a tendency."""
    b = E.per_manager_breakdown(2025)
    k = b.filter(pl.col("manager") == "Kylie Leahy")
    assert k.height == 1
    assert abs(float(k["gain"][0])) < 1e-9


# ─── output ─────────────────────────────────────────────────────────────────
def test_opponents_json_carries_its_own_caveats():
    d = OB.build(2026)
    assert len(d["managers"]) == 8
    assert "format_change" in d["caveats"] and "2-QB" in d["caveats"]["format_change"]
    for m in d["managers"]:
        assert "evidence" in m and "n_drafts" in m["evidence"]
        assert m["schedule_weight"] in (1, 2)
    dbl = [m for m in d["managers"] if m["faced_twice"]]
    assert len(dbl) == 4
