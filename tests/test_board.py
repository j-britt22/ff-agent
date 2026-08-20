"""Milestone 4 — VOR, tiers, bye adjustment, board.json (§7.3, §7.4, §2.1)."""
import polars as pl
import pytest

from ff_agent.config import N_TEAMS, SEASON, STARTER_SLOTS
from ff_agent.board import build as B
from ff_agent.board import replacement as R
from ff_agent.board import review as RV
from ff_agent.board import tiers as T

pytestmark = pytest.mark.espn


@pytest.fixture(scope="module")
def board():
    return B.build(SEASON)


# ─── replacement levels ─────────────────────────────────────────────────────
def test_flex_share_is_measured_not_assumed():
    """§7.3 estimates ~4 RB / 4 WR / 1 TE. This league's own history disagrees:
    across 364 flex starts in 2023-2025 a tight end was started ZERO times."""
    share = R.flex_share()
    assert abs(sum(share.values()) - 1.0) < 1e-6
    assert share["RB"] > share["WR"] > share["TE"]
    assert share["TE"] < 0.02, f"TE flex share measured at {share['TE']}"


def test_replacement_ranks_follow_measured_flex():
    ranks = R.replacement_ranks()
    assert ranks["QB"] == N_TEAMS * STARTER_SLOTS["QB"] == 18
    # TE drops to 9 because the flex never takes one
    assert ranks["TE"] == pytest.approx(9.0, abs=0.2)
    # RB absorbs most of the flex, so its replacement sits deeper than §7.3's 22
    assert ranks["RB"] > 22.5
    assert 21.0 < ranks["WR"] < 22.5


def test_replacement_points_are_ordered_sensibly(board):
    r = board.select("position", "replacement_points").unique().drop_nulls()
    d = dict(zip(r["position"].to_list(), r["replacement_points"].to_list()))
    assert d["QB"] > d["RB"] and d["QB"] > d["WR"]   # 6-pt pass TDs, 2-QB league
    assert d["TE"] < d["WR"]


# ─── VOR ────────────────────────────────────────────────────────────────────
def test_vor_uses_projected_not_realised_points(board):
    """M3 found realised 'elite' figures are upward-biased — the top scorer is a
    maximum over players and banks luck. VOR must come from projections."""
    assert "blended_points" in board.columns
    j = board.filter(pl.col("vor_raw").is_not_null()).head(50)
    recomputed = (j["blended_points"] - j["replacement_points"]).round(2)
    assert (recomputed - j["vor_raw"]).abs().max() < 0.02


def test_qb_vor_leads_rb_vor(board):
    """§3.1 called QB and RB 'near-tied'. Measured under §1 they are not."""
    best = board.group_by("position").agg(pl.col("vor").max().alias("top_vor"))
    d = dict(zip(best["position"].to_list(), best["top_vor"].to_list()))
    assert d["QB"] > d["RB"]


# ─── §2.1 bye adjustment ────────────────────────────────────────────────────
def test_free_bye_is_never_a_penalty(board):
    """A player below replacement gains nothing from a free bye, but must not
    lose anything either — you would be starting the replacement regardless."""
    assert board["bye_adjustment"].min() >= 0.0


def test_only_week_5_and_14_byes_are_adjusted(board):
    adjusted = board.filter(pl.col("bye_adjustment") > 0)
    assert set(adjusted["bye_week"].unique().to_list()) <= {5, 14}
    assert adjusted.height > 0


def test_bye_adjustment_equals_one_week_of_vor(board):
    """§2.1 asks for a real point value, not a hand-tuned nudge: the free bye is
    worth exactly one week of (player - replacement)."""
    a = board.filter(pl.col("bye_adjustment") > 0).head(20)
    expected = (a["vor_raw"] / B.GAMES).round(2)
    assert (expected - a["bye_adjustment"]).abs().max() < 0.02


def test_free_bye_players_visibly_rise(board):
    """The §11 test, measured rather than eyeballed."""
    lift = RV.bye_lift(board, top=100)
    free = lift.filter(pl.col("free_bye_week_5_or_14"))
    not_free = lift.filter(~pl.col("free_bye_week_5_or_14"))
    assert free.height and not_free.height
    assert free["mean_bye_adj"][0] > 0
    assert free["mean_rank_gain_vs_ecr"][0] > not_free["mean_rank_gain_vs_ecr"][0]


# ─── §3.3 sack adjustment ───────────────────────────────────────────────────
def test_sack_adjustment_applies_to_quarterbacks_only(board):
    non_qb = board.filter(pl.col("position") != "QB")
    assert (non_qb["sack_adjustment"].abs() < 1e-9).all()
    assert board.filter(pl.col("position") == "QB")["sack_adjustment"].abs().max() > 1


def test_sack_adjustment_uses_measured_persistence():
    """M3b measured sacks-over-expected carrying forward at 0.434 — four times
    TD-over-expected. Consensus prices none of it (§3.3) and the M3 model dropped
    sack_rate for falling under the stability cutoff, so this term is the only
    place the edge is captured."""
    assert 0.3 < B.SOE_PERSISTENCE < 0.55


def test_sack_adjustment_direction(board):
    """Taking MORE sacks than expected must cost points, not gain them."""
    qb = board.filter((pl.col("position") == "QB")
                      & pl.col("prior_sacks_over_expected").is_not_null())
    worst = qb.sort("prior_sacks_over_expected", descending=True).head(3)
    assert (worst["sack_adjustment"] < 0).all()
    best = qb.sort("prior_sacks_over_expected").head(3)
    assert (best["sack_adjustment"] > 0).all()


# ─── §7.4 tiers ─────────────────────────────────────────────────────────────
def test_tiers_are_ordered_within_position(board):
    for pos in ("QB", "RB", "WR", "TE"):
        sub = board.filter(pl.col("position") == pos).sort("vor", descending=True)
        tiers = sub["tier"].to_list()
        assert tiers == sorted(tiers), f"{pos} tiers must not go backwards"


def test_players_remaining_in_tier_counts_down(board):
    sub = board.filter((pl.col("position") == "RB") & (pl.col("tier") == 1))
    assert sub["players_remaining_in_tier"].min() == 1
    assert sub["players_remaining_in_tier"].max() == sub.height


def test_tier_boundaries_survive_perturbation(board):
    """Risk control: a cliff that exists only at one exact projection is false
    precision, not a cliff."""
    assert board["tier_stability"].median() > 0.5


# ─── §2.1 QB count, §10 alarms ──────────────────────────────────────────────
def test_qb_count_decision_is_explicit(board):
    d = RV.qb_count_decision(board)
    assert "decision" in d and len(d["decision"]) > 20
    assert isinstance(d["weeks_qb3_would_start"], int)
    assert d["weeks_qb3_would_start"] == len(d["byes_needing_cover"])


def test_no_sanity_alarms(board):
    """§10: a K before round 13 or a null VOR each mean something is broken."""
    assert RV.sanity_alarms(board) == []


def test_divergences_all_carry_a_reason(board):
    d = RV.divergences(board, top=50)
    assert d.height > 0
    assert d["reason"].null_count() == 0
    assert d.filter(pl.col("reason").str.len_chars() < 10).height == 0
