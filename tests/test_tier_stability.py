"""§7.4 — tier_stability must be reproducible, or it cannot do its job.

The function exists to expose false precision: a cliff that survives only one
exact projection is not a cliff. Measured 2026-08-21, it was itself the worst
offender — two `cli board` runs on byte-identical code disagreed on 85 of 515
rows, up to 0.225, Gibbs and Chase among them. More than an actual change to the
projection layer moved (44 rows).

The cause was not the seed. The noise vector was drawn once and applied
POSITIONALLY, so which perturbation hit which player depended on the frame's row
order — and `assign_tiers` builds that order from `unique()` over positions plus
a sort on a float column full of ties, neither of which polars orders
deterministically by default.

These tests need no ESPN data on purpose: the property is arithmetic, and an
invariant that skips when credentials are missing is not an invariant.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ff_agent.board import tiers as T

TRIALS = 40
"""Small on purpose — these tests pin ORDER-INVARIANCE, which is exact, not the
shipped precision. T.TRIALS is checked directly in the precision test below."""


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    """Board-shaped, carrying the two features that broke reproducibility:
    several positions, and exact TIES on the value column."""
    rng = np.random.default_rng(4)
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        pts = np.round(np.sort(rng.uniform(20.0, 300.0, 30))[::-1], 1)
        pts[5] = pts[4]          # ties: polars' default sort reorders these
        pts[12] = pts[11]
        rows += [
            {"canonical_id": f"{pos}-{i:03d}", "position": pos,
             "blended_points": float(p)}
            for i, p in enumerate(pts)
        ]
    return pl.DataFrame(rows)


def _stability(df: pl.DataFrame) -> pl.DataFrame:
    return T.tier_stability(df, trials=TRIALS)


# ─── the reproducibility gate ───────────────────────────────────────────────
def test_tier_stability_repeats_exactly(frame):
    assert _stability(frame).equals(_stability(frame))


def test_tier_stability_is_invariant_to_row_order(frame):
    """The regression test for the 2026-08-21 bug. Same data, shuffled rows —
    every player must keep the same number, because the perturbation is keyed on
    the player, not on where the player happens to sit in the frame."""
    base = _stability(frame)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(frame.height).tolist()
        shuffled = _stability(frame[perm])
        j = base.join(shuffled, on="canonical_id", how="inner", suffix="_shuffled")
        assert j.height == base.height
        moved = j.filter(pl.col("tier_stability") != pl.col("tier_stability_shuffled"))
        assert moved.height == 0, (
            f"shuffle seed {seed}: {moved.height} players changed, worst "
            f"{float((j['tier_stability'] - j['tier_stability_shuffled']).abs().max())}"
        )


def test_assign_tiers_row_order_is_deterministic(frame):
    """The mechanism, pinned one level down. Without maintain_order on unique()
    and sort(), the position blocks and the tied rows inside them came back in a
    different order on nearly every call — 7 distinct orders in 8 calls."""
    orders = {tuple(T.assign_tiers(frame)["canonical_id"].to_list()) for _ in range(10)}
    assert len(orders) == 1, f"{len(orders)} distinct row orders in 10 calls"


def test_a_players_draw_does_not_depend_on_the_pool(frame):
    """Keying on canonical_id buys more than order-invariance: reprojecting or
    removing ONE player must not redraw everybody else, or a real projection
    change shows up in rows it never touched."""
    ids = frame["canonical_id"].to_list()
    full = T._noise_matrix(ids, trials=8, noise_pct=0.05, seed=17)
    without_first = T._noise_matrix(ids[1:], trials=8, noise_pct=0.05, seed=17)
    assert np.array_equal(full[1:], without_first)


def test_draws_survive_a_different_process_hash_seed(frame):
    """Guards the blake2b-not-hash() choice. Python salts str hashing per
    process, so keying on hash() would be reproducible inside one run and drift
    between runs — the same bug wearing a different hat, and invisible to every
    test above because they share a process."""
    root = Path(__file__).resolve().parents[1]
    script = (
        "import numpy as np;"
        "from ff_agent.board import tiers as T;"
        "print(list(np.round(T._noise_matrix(['a','b','c'], 3, 0.05, 17).ravel(), 12)))"
    )
    out = []
    for hashseed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        r = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out.append(r.stdout.strip())
    assert out[0] == out[1], "noise draws changed with PYTHONHASHSEED"


# ─── precision ──────────────────────────────────────────────────────────────
def test_reported_precision_matches_the_trial_count():
    """k/trials is a binomial proportion; its standard error peaks at
    0.5/sqrt(trials). The shipped 40 trials gave 0.079 while board.json reported
    three decimals — 79x more precision than the estimator had."""
    se = 0.5 / math.sqrt(T.TRIALS)
    grid = 10.0 ** -T.STABILITY_DECIMALS
    assert se < 0.03, f"{T.TRIALS} trials gives se {se:.3f}: too noisy to publish"
    assert grid <= se, (
        f"reporting a {grid} grid against se {se:.3f} throws away real resolution")
    assert grid >= se / 10, (
        f"a {grid} grid resolves {se / grid:.0f} steps inside one standard error "
        "— that is the false precision this function exists to prevent")


def test_reported_values_are_probabilities(frame):
    """Only holds while canonical_id is unique in the input: a duplicated player
    is counted once per row per trial, so the ratio would exceed 1."""
    assert frame["canonical_id"].n_unique() == frame.height
    v = _stability(frame)["tier_stability"]
    assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0


# ─── the keying is load-bearing, so guard it ────────────────────────────────
def test_null_canonical_id_fails_loudly(frame):
    """§0.2's rule applied one level down: a null key would silently hand every
    unresolved player the same perturbation."""
    holed = frame.with_columns(
        pl.when(pl.col("canonical_id") == "QB-000")
        .then(None).otherwise(pl.col("canonical_id")).alias("canonical_id")
    )
    with pytest.raises(ValueError, match="canonical_id"):
        _stability(holed)
