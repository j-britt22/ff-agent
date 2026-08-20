"""MILESTONE 2 GATE (§11): recomputed scores vs ESPN's recorded scores.

Two layers, and the distinction matters:

  **Layer A** applies our rules to ESPN's OWN stat line. It must be 100% — a
  single failure means a scoring RULE is wrong, which would corrupt every
  downstream number. This is the hard gate.

  **Layer B** scores the nflverse stat line end to end. Residual failures here
  are stat-value disagreements between nflverse and ESPN, not rule errors.
  The known ones are pinned individually so a NEW disagreement fails the suite
  instead of hiding inside a tolerance.
"""
import polars as pl
import pytest

from ff_agent.config import LAST_SEASON
from ff_agent.scoring import validate as V

pytestmark = pytest.mark.espn

# Rows where nflverse and ESPN disagree about a STAT, not about a rule.
# Each was traced to a specific play; see the comments.
KNOWN_INPUT_DISAGREEMENTS: dict[int, set[tuple]] = {
    2023: set(),
    2024: {
        ("BAL", 4),   # nflverse credits the D/ST a TD ESPN does not
        ("LAC", 8),   # ESPN excludes 2 more points than we detect as conceded
    },
    2025: {
        ("Caleb Williams", 6),  # nflverse 3 rush yds over 4 plays; ESPN -2
        ("PHI", 14),            # ESPN counts 1 fumble recovery, nflverse 2
    },
}


@pytest.mark.parametrize("season", [2023, 2024, LAST_SEASON])
def test_layer_a_rules_are_exactly_right(season):
    """The hard gate. Our rules on ESPN's stat line must reproduce ESPN exactly.

    Covers two different rulesets: 2023-24 additionally scored completions and
    incompletions, both removed for 2025.
    """
    a = V.layer_a_rules_check(season)
    bad = a.filter(pl.col("delta").abs() > V.TOLERANCE)
    assert bad.height == 0, (
        f"{season}: {bad.height} rows where our RULES disagree with ESPN — "
        f"a scoring rule is wrong.\n{bad.head(10)}"
    )


@pytest.mark.parametrize("season", [2023, 2024, LAST_SEASON])
def test_layer_b_end_to_end_has_no_new_disagreements(season):
    players = V.compare_players(season)
    dst = V.compare_dst(season)
    known = KNOWN_INPUT_DISAGREEMENTS[season]

    found = set()
    for row in players.filter(pl.col("delta").abs() > V.TOLERANCE).to_dicts():
        found.add((row["name"], row["week"]))
    for row in dst.filter(pl.col("delta").abs() > V.TOLERANCE).to_dicts():
        found.add((row["team"], row["week"]))

    assert found == known, (
        f"{season}: scoring disagreements changed.\n"
        f"  new: {sorted(found - known)}\n"
        f"  fixed (remove from KNOWN_INPUT_DISAGREEMENTS): {sorted(known - found)}"
    )


def test_2023_is_exact_end_to_end():
    """A full season with zero disagreements — evidence the residual 2024/2025
    rows are data anomalies rather than a systematic bug."""
    p = V.compare_players(2023)
    d = V.compare_dst(2023)
    assert p.filter(pl.col("delta").abs() > V.TOLERANCE).height == 0
    assert d.filter(pl.col("delta").abs() > V.TOLERANCE).height == 0


def test_overall_accuracy_across_every_available_season():
    tot = ok = 0
    for season in (2023, 2024, LAST_SEASON):
        for df in (V.compare_players(season), V.compare_dst(season)):
            tot += df.height
            ok += df.filter(pl.col("delta").abs() <= V.TOLERANCE).height
    assert tot > 6000
    assert ok / tot > 0.999, f"end-to-end accuracy {ok}/{tot}"


def test_espn_breakdown_always_sums_to_its_own_total():
    """Sanity on the reference data itself — if this fails, ESPN's payload is
    inconsistent and no comparison against it means anything."""
    for season in (2023, 2024, LAST_SEASON):
        for df in (V.compare_players(season), V.compare_dst(season)):
            assert df.filter(pl.col("layerA_delta").abs() > V.TOLERANCE).height == 0
