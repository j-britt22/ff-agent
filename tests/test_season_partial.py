"""M10b's three changes to shipped season modules.

Every one of them has the same acceptance bar: **the preseason behaviour must be
unchanged**. M1-M10a all run through these functions, and a silent shift in the
board or the draft simulator would be far worse than the in-season feature is
worth.
"""
import numpy as np
import polars as pl
import pytest

from ff_agent.config import STARTER_SLOTS
from ff_agent.season import lineup as LU
from ff_agent.season import simulate as SIM
from ff_agent.season import strength as ST
from tests.conftest import SYNTH_TEAMS, brute_force_lineup


def roster(vals: dict[str, float] | None = None) -> pl.DataFrame:
    """A legal 15-man roster with distinct values, for lineup tests."""
    spec = [
        ("q1", "QB", 24.0), ("q2", "QB", 19.0), ("q3", "QB", 15.0),
        ("r1", "RB", 18.0), ("r2", "RB", 14.0), ("r3", "RB", 11.0), ("r4", "RB", 6.0),
        ("w1", "WR", 17.0), ("w2", "WR", 16.0), ("w3", "WR", 12.0), ("w4", "WR", 5.0),
        ("t1", "TE", 10.0), ("t2", "TE", 4.0),
        ("k1", "K", 8.0), ("d1", "DST", 7.0),
    ]
    df = pl.DataFrame(
        {"canonical_id": [s[0] for s in spec],
         "position": [s[1] for s in spec],
         "weekly_points": [s[2] for s in spec]}
    )
    if vals:
        df = df.with_columns(
            pl.col("canonical_id").replace_strict(vals, default=None)
            .fill_null(pl.col("weekly_points")).alias("weekly_points")
        )
    return df


# ─── 1. lineup pinning ──────────────────────────────────────────────────────
def test_unpinned_lineup_is_unchanged():
    """The whole point of the default argument. 10 starters, best of each slot."""
    lu = LU.optimal_lineup(roster())
    assert lu.height == sum(STARTER_SLOTS.values()) == 10
    got = dict(zip(lu["canonical_id"].to_list(), lu["slot"].to_list()))
    assert got == {
        "q1": "QB", "q2": "QB", "r1": "RB", "r2": "RB",
        "w1": "WR", "w2": "WR", "t1": "TE", "k1": "K", "d1": "DST",
        "w3": "FLEX",          # WR3 12.0 beats RB3 11.0 for the flex
    }
    assert LU.lineup_points(roster()) == pytest.approx(
        24 + 19 + 18 + 14 + 17 + 16 + 10 + 8 + 7 + 12
    )


def test_pinning_forces_a_slot_and_costs_what_it_should():
    """Pinning the WORST tight end is a real, quantified loss — not a no-op."""
    free = LU.lineup_points(roster())
    pinned = LU.lineup_points(roster(), pinned={"t2": "TE"})
    assert pinned == pytest.approx(free - 10.0 + 4.0)


def test_pinning_reroutes_rather_than_simply_dropping():
    """A pinned RB3 does not evict RB2 from the lineup — it moves him to the FLEX.

    Which is the whole reason pinning has to go through the solver rather than
    being applied afterwards: the cheapest way to absorb a pin is usually to
    rearrange around it, and only the assignment problem knows that.
    """
    lu = LU.optimal_lineup(roster(), pinned={"r3": "RB"})
    got = dict(zip(lu["canonical_id"].to_list(), lu["slot"].to_list()))
    assert got["r3"] == "RB"
    assert got["r1"] == "RB"          # the better back keeps the other RB slot
    assert got["r2"] == "FLEX"        # RB2 is displaced into the flex, not dropped
    assert "w3" not in got            # and the flex-bound WR is what actually falls out
    # The naive reading — "we lost RB2 (14) and gained RB3 (11)" — says the pin
    # costs 3. It costs ONE: RB2 keeps his points from the flex, and what
    # actually falls out is the flex-bound WR3 (12), replaced by RB3 (11).
    assert LU.lineup_points(roster(), pinned={"r3": "RB"}) == pytest.approx(
        LU.lineup_points(roster()) - 12.0 + 11.0
    )


def test_pinned_flex_is_honoured():
    lu = LU.optimal_lineup(roster(), pinned={"r4": "FLEX"})
    got = dict(zip(lu["canonical_id"].to_list(), lu["slot"].to_list()))
    assert got["r4"] == "FLEX"
    assert got["r1"] == "RB" and got["r2"] == "RB"


@pytest.mark.parametrize("bad,frag", [
    ({"q1": "FLEX"}, "cannot start at 'FLEX'"),
    ({"w1": "RB"}, "cannot start at 'RB'"),
    ({"nobody": "QB"}, "not on this roster"),
    ({"q1": "BENCH"}, "not a starting slot"),
    ({"r1": "RB", "r2": "RB", "r3": "RB"}, "only 2 starting spot"),
])
def test_impossible_pins_are_refused_loudly(bad, frag):
    """§10: a pin we cannot honour means our lineup and ESPN's disagree. That is
    a blocking error, because the alternative is advising on a spent slot."""
    with pytest.raises(LU.PinError, match=frag):
        LU.optimal_lineup(roster(), pinned=bad)


@pytest.mark.parametrize("seed", range(12))
def test_pinned_solver_matches_brute_force(seed):
    """The optimality argument is about the FLEX; this checks the CODE."""
    rng = np.random.default_rng(seed)
    small = pl.DataFrame({
        "canonical_id": ["a", "b", "c", "d", "e"],
        "position": ["RB", "RB", "WR", "WR", "TE"],
        "weekly_points": rng.uniform(1, 20, 5).round(2),
    })
    slots = {"RB": 2, "WR": 2, "FLEX": 1}
    for pins in ({}, {"a": "FLEX"}, {"e": "FLEX"}, {"b": "RB"}, {"c": "WR", "e": "FLEX"}):
        fast = sum(
            r["weekly_points"]
            for r in _solve(small, slots, pins).to_dicts()
        )
        slow = brute_force_lineup(small, slots, pinned=pins)
        assert fast == pytest.approx(slow), f"seed={seed} pins={pins}"


def _solve(df: pl.DataFrame, slots: dict[str, int], pins: dict[str, str]):
    """Run the real solver against a non-standard slot map."""
    import ff_agent.season.lineup as mod
    old_starter, old_strict = mod.STARTER_SLOTS, mod.STRICT_SLOTS
    mod.STARTER_SLOTS = slots
    mod.STRICT_SLOTS = {k: v for k, v in slots.items() if k != "FLEX"}
    try:
        return mod.optimal_lineup(df, pinned=pins)
    finally:
        mod.STARTER_SLOTS, mod.STRICT_SLOTS = old_starter, old_strict


# ─── 2. roster_strength is parameterised ────────────────────────────────────
def test_roster_strength_default_is_unchanged():
    rosters = pl.DataFrame({"fantasy_team": ["A"] * 15, "canonical_id": roster()["canonical_id"]})
    proj = roster().rename({"weekly_points": "blended_points"}).with_columns(
        pl.col("blended_points") * 17
    )
    out = ST.roster_strength(rosters, proj)
    assert out["A"] == pytest.approx(LU.lineup_points(roster()), abs=0.01)


def test_roster_strength_takes_remaining_games_and_a_ros_column():
    """In-season the numerator is REST-OF-SEASON points over REMAINING games."""
    rosters = pl.DataFrame({"fantasy_team": ["A"] * 15, "canonical_id": roster()["canonical_id"]})
    proj = roster().rename({"weekly_points": "ros_points"}).with_columns(
        pl.col("ros_points") * 6
    )
    out = ST.roster_strength(rosters, proj, games=6, value_col="ros_points")
    assert out["A"] == pytest.approx(LU.lineup_points(roster()), abs=0.01)


def test_roster_strength_refuses_a_missing_column_and_a_dead_season():
    rosters = pl.DataFrame({"fantasy_team": ["A"], "canonical_id": ["q1"]})
    proj = roster().rename({"weekly_points": "ros_points"})
    with pytest.raises(KeyError, match="blended_points"):
        ST.roster_strength(rosters, proj)
    with pytest.raises(ValueError, match="games must be positive"):
        ST.roster_strength(rosters, proj, games=0, value_col="ros_points")


# ─── 3. the simulator accepts completed weeks ───────────────────────────────
def _flat(means=130.0):
    return {t: means for t in SYNTH_TEAMS}


def test_preseason_simulation_is_untouched(synth_league):
    """No `completed` argument means byte-identical behaviour."""
    a = SIM.simulate(_flat(), n_sims=3000, seed=11)
    b = SIM.simulate(_flat(), n_sims=3000, seed=11, completed=None)
    assert np.array_equal(a.p_title, b.p_title)
    assert a.completed_weeks == ()
    # nine identical teams -> roughly 1/9 each
    assert a.p_title.sum() == pytest.approx(1.0, abs=0.01)
    assert a.p_title.max() < 0.16


def test_completed_weeks_are_deterministic_not_redrawn(synth_league):
    """A played week is not a random variable. Two different seeds must agree on
    the results of the weeks that already happened."""
    comp = _results(weeks=range(1, 7), winner="T1", loser="T9")
    a = SIM.simulate(_flat(), n_sims=2500, seed=1, completed=comp)
    b = SIM.simulate(_flat(), n_sims=2500, seed=99, completed=comp)
    assert a.completed_weeks == tuple(range(1, 7))
    # T1 won every completed game under both seeds, so its floor is identical
    ai = dict(zip(a.teams, a.mean_wins))
    bi = dict(zip(b.teams, b.mean_wins))
    assert abs(ai["T1"] - bi["T1"]) < 0.35


def test_a_hot_start_moves_title_odds_the_right_way(synth_league):
    """The whole reason the argument exists: a week-9 forecast that re-simulates
    weeks 1-8 is a forecast of a different season."""
    base = dict(zip(SYNTH_TEAMS, SIM.simulate(_flat(), n_sims=4000, seed=3).p_title))
    comp = _results(weeks=range(1, 7), winner="T1", loser="T9")
    after = SIM.simulate(_flat(), n_sims=4000, seed=3, completed=comp)
    got = dict(zip(after.teams, after.p_title))
    assert got["T1"] > base["T1"] * 1.5
    assert got["T9"] < base["T9"] * 0.2


@pytest.mark.parametrize("bad,frag", [
    ({"week": [1], "team": ["NOPE"], "points": [100.0]}, "does not know"),
    ({"week": [99], "team": ["T1"], "points": [100.0]}, "outside the"),
    ({"week": [1, 1], "team": ["T1", "T1"], "points": [1.0, 2.0]}, "duplicate"),
    ({"week": [1], "team": ["T1"], "points": [None]}, "null score"),
])
def test_bad_completed_rows_are_refused(synth_league, bad, frag):
    """A silently dropped result is a team carrying a simulated week it lost."""
    with pytest.raises(ValueError, match=frag):
        SIM.simulate(_flat(), n_sims=50, completed=pl.DataFrame(bad))


def test_completed_missing_columns_is_refused(synth_league):
    with pytest.raises(ValueError, match="missing"):
        SIM.simulate(_flat(), n_sims=50,
                     completed=pl.DataFrame({"week": [1], "team": ["T1"]}))


def test_bye_weeks_contribute_nothing_to_a_completed_week(synth_league):
    """A team on bye in a played week scored zero, not a leftover random draw."""
    comp = _results(weeks=[1], winner="T1", loser="T9")
    r = SIM.simulate(_flat(), n_sims=800, seed=5, completed=comp)
    bye_team = (
        synth_league.filter((pl.col("week") == 1) & (pl.col("opponent").is_null()))
        ["team"][0]
    )
    played = {row["team"] for row in comp.iter_rows(named=True)}
    assert bye_team not in played
    # its points-for must not include a phantom week-1 score
    pts = dict(zip(r.teams, r.mean_points))
    others = [v for t, v in pts.items() if t not in {"T1", "T9", bye_team}]
    assert pts[bye_team] < min(others)


def _results(weeks, winner: str, loser: str) -> pl.DataFrame:
    """Actual scores for whole weeks: one team dominant, one dreadful."""
    import tests.conftest as C

    sched = pl.DataFrame(C._round_robin(C.SYNTH_TEAMS, 14))
    rows = []
    for wk in weeks:
        wk_rows = sched.filter((pl.col("week") == wk) & (pl.col("opponent").is_not_null()))
        for t in wk_rows["team"].to_list():
            pts = 200.0 if t == winner else (60.0 if t == loser else 130.0)
            rows.append({"week": int(wk), "team": t, "points": pts})
    return pl.DataFrame(rows)
