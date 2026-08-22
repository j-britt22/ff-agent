"""Milestone 4 — VOR, tiers, bye adjustment, board.json (§7.3, §7.4, §2.1)."""
import numpy as np
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


def test_tier_stability_is_reproducible_on_the_real_board(board):
    """The synthetic version of this lives in tests/test_tier_stability.py; this
    one runs it on the frame the bug was actually found on, which carries the
    ties and the ragged position sizes a hand-built frame does not.
    """
    df = board.select("canonical_id", "position", "vor")
    base = T.tier_stability(df, value="vor")
    perm = np.random.default_rng(0).permutation(df.height).tolist()
    shuffled = T.tier_stability(df[perm], value="vor")
    j = base.join(shuffled, on="canonical_id", how="inner", suffix="_shuffled")
    assert j.height == base.height
    moved = j.filter(pl.col("tier_stability") != pl.col("tier_stability_shuffled"))
    assert moved.height == 0, f"{moved.height} of {j.height} players moved"


def test_tier_stability_is_a_frequency(board):
    """It is ``count / trials``, so it cannot exceed 1.

    The 2026 board shipped 28 rows above 1.0, topping out at exactly 2.0 — the
    signature of a player counted once per DUPLICATE ROW per trial. A field
    documented as "how often each break survives" reading 2.0 is not a rounding
    artefact, it is a duplicated player wearing a plausible-looking number.
    """
    assert board["tier_stability"].max() <= 1.0
    assert board["tier_stability"].min() >= 0.0


# ─── §0.2 — one row per player, and joins that cannot fan out ───────────────
def test_board_has_exactly_one_row_per_player(board):
    """§0.2's hard gate, at the board rather than at ID resolution.

    Resolving to exactly one canonical ID is only half of it: a join that fans
    out breaks the same rule just as thoroughly and far more quietly. The
    simulator marks pool INDICES taken, not people, so a doubled row lets two
    fantasy teams draft the same person in one simulated draft.
    """
    B.assert_one_row_per_player(board)
    assert board.height == board["canonical_id"].n_unique()


def test_tier_stability_refuses_duplicated_input():
    """Where the >1.0 values were actually born.

    Guarding here rather than only in build() means every caller gets the
    failure, and it arrives naming the duplication rather than a strange
    stability score fifty lines downstream.
    """
    df = pl.DataFrame({
        "canonical_id": ["a", "b", "c", "d"],
        "position": ["RB"] * 4,
        "vor": [100.0, 60.0, 55.0, 10.0],
    })
    assert T.tier_stability(df, value="vor").height == 4

    dup = pl.concat([df, df.head(1)])
    with pytest.raises(ValueError, match="one row per canonical_id"):
        T.tier_stability(dup, value="vor")


def test_tier_stability_output_cannot_fan_out_a_join(board):
    """The right-hand side of build()'s join, checked as a join key.

    ``tier_stability`` emits one row per input ROW, not per player, which is
    what turned 15 duplicated players into 60 board rows. Unique input, unique
    output, height-preserving join.
    """
    stab = T.tier_stability(board.select("canonical_id", "position", "vor"),
                            value="vor", trials=3)
    assert stab.height == stab["canonical_id"].n_unique()
    assert board.join(stab, on="canonical_id", how="left").height == board.height


def test_build_refuses_to_join_duplicated_projections(monkeypatch):
    """§0.2 must fail BEFORE the tier join, not after it.

    ``write_board`` already asserted, but only once the damage was squared and
    only on the path that writes the artifact — ``draft/pool.py`` and
    ``draft/backtest.py`` call ``build()`` directly. Checking the left side
    before the join is what makes the join unable to multiply rows at all.
    """
    from ff_agent.projections import board_inputs as BI

    real = BI.build(SEASON)
    monkeypatch.setattr(BI, "build", lambda *a, **k: pl.concat([real, real.head(1)]))

    with pytest.raises(ValueError, match="the board's tier join"):
        B.build(SEASON)


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
