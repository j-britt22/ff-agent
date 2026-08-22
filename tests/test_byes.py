"""§2.1 free-bye flag. Byes are derived from ABSENCE, not a bye column."""
import polars as pl
import pytest

from ff_agent.config import FREE_BYE_WEEKS, HISTORY_SEASONS, SEASON
from ff_agent.data import byes, crosswalk as cw, nflverse as nv


@pytest.fixture(scope="module")
def sched():
    return nv.schedules(offline=True)


@pytest.mark.parametrize("season", [*HISTORY_SEASONS, SEASON])
def test_every_team_has_exactly_one_bye(season, sched):
    b = byes.bye_weeks(season, schedules=sched)
    counts = b.group_by("team").len()
    assert counts.filter(pl.col("len") != 1).height == 0
    assert b.height == 32, f"{season}: expected 32 teams, got {b.height}"


def test_free_bye_flag_matches_my_fantasy_byes(sched):
    b = byes.bye_weeks(SEASON, schedules=sched)
    flagged = set(b.filter(pl.col("free_bye"))["bye_week"].unique().to_list())
    assert flagged <= FREE_BYE_WEEKS
    for wk in flagged:
        assert wk in (5, 14)


def test_week_14_byes_actually_exist_in_2026(sched):
    """If the NFL stopped scheduling week-14 byes, half of §2.1's edge vanishes
    and the board must not pretend otherwise. This asserts the fact we verified."""
    b = byes.bye_weeks(SEASON, schedules=sched)
    wk14 = b.filter(pl.col("bye_week") == 14)
    assert wk14.height > 0, (
        "No week-14 NFL byes in 2026 — §2.1's week-14 arbitrage does not exist. "
        "Update CLAUDE.md and the board's bye_adjustment before proceeding."
    )


def test_free_bye_pool_is_small_and_specific(sched):
    """2026 has materially fewer free-bye teams than 2025 (4 vs 8). Locking the
    number in means a schedule refresh that changes it is visible, not silent."""
    teams_2026 = set(byes.free_bye_teams(SEASON))
    assert len(teams_2026) == 4, f"free-bye pool changed: {sorted(teams_2026)}"
    assert teams_2026 == {"ARI", "CAR", "DAL", "KC"}


def test_bye_weeks_fall_in_a_plausible_range(sched):
    for season in (SEASON, 2025):
        b = byes.bye_weeks(season, schedules=sched)
        assert b["bye_week"].min() >= 4
        assert b["bye_week"].max() <= 14


# ─── historical schedule anomalies ───────────────────────────────────────────
def test_cancelled_game_is_not_mistaken_for_a_bye(sched):
    """2022 BUF/CIN week 17 was cancelled (Damar Hamlin) and never made up.

    Both teams have two no-game holes. The bye is the one in the bye window;
    week 17 must be reported as an anomaly, not silently treated as a bye.
    """
    b = byes.bye_weeks(2022, schedules=sched)
    assert b.filter(pl.col("team") == "BUF")["bye_week"][0] == 7
    assert b.filter(pl.col("team") == "CIN")["bye_week"][0] == 10

    a = byes.schedule_anomalies(2022, schedules=sched)
    assert set(a["team"].to_list()) == {"BUF", "CIN"}
    assert set(a["week"].to_list()) == {17}


def test_postponed_game_shifts_the_functional_bye(sched):
    """2017 MIA/TB: week 1 postponed for Hurricane Irma, replayed in week 11 —
    which had been their scheduled bye. Week 1 is therefore their real bye."""
    b = byes.bye_weeks(2017, schedules=sched)
    assert b.filter(pl.col("team") == "MIA")["bye_week"][0] == 1
    assert b.filter(pl.col("team") == "TB")["bye_week"][0] == 1
    assert byes.schedule_anomalies(2017, schedules=sched).height == 0


def test_no_unexplained_anomalies_in_the_history_window(sched):
    """Any NEW anomaly is a data-quality event worth a human look."""
    known = {(2022, "BUF", 17), (2022, "CIN", 17)}
    found = set()
    for season in [*HISTORY_SEASONS, SEASON]:
        a = byes.schedule_anomalies(season, schedules=sched)
        found |= {(r["season"], r["team"], r["week"]) for r in a.to_dicts()}
    assert found == known, f"unexpected schedule anomalies: {found ^ known}"


# ─── team spelling: nflverse "LA" vs everyone else's "LAR" ───────────────────
def test_bye_table_speaks_one_canonical_team_vocabulary(sched):
    """The bye table is joined against ESPN-derived rows, so it must not carry
    nflverse's spelling.

    nflverse spells the Rams ``LA``; ESPN, the crosswalk, the projections and the
    board all spell them ``LAR``. A left join on the raw abbreviation does not
    fail — it yields null — so this has bitten three separate times (M2 scoring,
    M7 draft pool, M4 board). Pinning the vocabulary at the source is what stops
    a fourth.
    """
    for season in [*HISTORY_SEASONS, SEASON]:
        b = byes.bye_weeks(season, schedules=sched)
        teams = set(b["team"].to_list())
        assert teams <= set(cw.CURRENT_TEAMS), (
            f"{season}: non-canonical team abbreviation(s) "
            f"{sorted(teams - set(cw.CURRENT_TEAMS))}"
        )
        assert len(teams) == 32
        assert "LAR" in teams and "LA" not in teams


def test_relocated_franchises_fold_into_their_current_spelling(sched):
    """2016 is the start of the history window and predates two relocations.

    Left raw, a 2016 backtest board would drop San Diego and Oakland the same way
    the 2026 board dropped the Rams. Folding them in is what keeps every season
    joinable against one vocabulary.
    """
    b2016 = set(byes.bye_weeks(2016, schedules=sched)["team"].to_list())
    assert {"SD", "OAK", "STL"} & b2016 == set()
    assert {"LAC", "LV", "LAR"} <= b2016


def test_a_non_canonical_team_is_a_blocking_error():
    """§0.2's culture, applied to team entities: fail loudly, never silently.

    The guard has to be shown to fire. A test that only reads today's clean data
    passes just as happily when the assertion is deleted.
    """
    bad = pl.DataFrame({"season": [2026], "team": ["LA"],
                        "bye_week": [11], "free_bye": [False]})
    with pytest.raises(byes.ByeWeekError, match="not canonical"):
        byes.assert_canonical_teams(bad, 2026)
