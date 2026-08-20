"""MILESTONE 3 GATE (§11): blended projections beat raw consensus on rank
correlation under §1 scoring.

The honest version of this test is walk-forward. Sweeping blend weights and
reporting the best is how a backtest lies to you — the winning weight was chosen
knowing the answer. Here each season's weight is fitted only on seasons before
it, so the figure is one a person could actually have had on draft day.
"""
import polars as pl
import pytest

from ff_agent.projections import backtest as B

pytestmark = pytest.mark.espn

BACKTEST_SEASONS = (2021, 2022, 2023, 2024, 2025)


@pytest.fixture(scope="module")
def walk_forward():
    return B.walk_forward(seasons=BACKTEST_SEASONS, per_position=False)


def test_blend_beats_consensus_walk_forward(walk_forward):
    """THE GATE. Weight fitted on prior seasons only, evaluated out of sample."""
    won = int((walk_forward["delta"] > 0).sum())
    n = walk_forward.height
    assert n >= 4, "need at least four out-of-sample seasons"
    assert won == n, (
        f"blend beat consensus in only {won}/{n} seasons\n{walk_forward}"
    )
    assert walk_forward["delta"].mean() > 0


@pytest.fixture(scope="module")
def sweep():
    return B.weight_sweep(
        seasons=BACKTEST_SEASONS,
        weights=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0),
    )


def test_the_gain_is_a_plateau_not_a_knife_edge(sweep):
    """A result that survives a RANGE of weights is evidence; one that needs a
    single exact weight is a fitted artifact."""
    good = sweep.filter(
        (pl.col("weight") >= 0.05) & (pl.col("weight") <= 0.20)
    )
    assert good.height >= 3
    assert (good["seasons_won"] == good["n_seasons"]).all(), (
        f"blend should win every season across the plateau\n{good}"
    )


def test_consensus_alone_is_strong_and_model_alone_is_not(sweep):
    """§7.2 step 4 says blend, don't replace — this is why. Our model on its own
    is materially worse than consensus, so consensus stays the anchor."""
    pure_consensus = sweep.filter(pl.col("weight") == 0.0)["mean_blend"][0]
    pure_model = sweep.filter(pl.col("weight") == 1.0)["mean_blend"][0]
    assert pure_consensus > 0.70
    assert pure_model < pure_consensus - 0.03


def test_default_weight_sits_inside_the_winning_plateau(sweep):
    assert 0.05 <= B.DEFAULT_WEIGHT <= 0.20
    row = sweep.filter(pl.col("weight") == 0.10)
    assert row["seasons_won"][0] == row["n_seasons"][0]


def test_universe_keeps_consensus_busts(walk_forward):
    """A consensus-ranked player who never played must stay in the sample at
    zero points. Dropping him would flatter consensus by hiding its misses."""
    ev = B.evaluation_frame(2025)
    assert ev.filter(pl.col("actual_points") == 0.0).height > 20
    assert ev["actual_points"].null_count() == 0
