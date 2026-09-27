"""M10b foundations: the lock calendar, the value layer, ROS, and league state.

Everything here runs offline. §0.3 wants the pipeline to work without a network
and the in-season jobs are the part most tempting to test only against the live
league — and therefore the part most likely to end up untested.

The schedule fixture deliberately reproduces the REAL 2026 irregularities that
finding F9 is about, rather than a tidy Thursday/Sunday week. A calendar module
that only ever sees tidy weeks is not tested for the thing that breaks it.
"""
import datetime as dt

import numpy as np
import polars as pl
import pytest

from ff_agent.inseason import clock as CK
from ff_agent.inseason import ros as R
from ff_agent.inseason import state as ST
from ff_agent.inseason import value as V

ET = CK.ET


def _game(gid, wk, day, time, away, home):
    return {"game_id": gid, "season": 2026, "week": wk, "game_type": "REG",
            "gameday": day, "gametime": time,
            "weekday": dt.date.fromisoformat(day).strftime("%A"),
            "away_team": away, "home_team": home}


@pytest.fixture(scope="module")
def schedule() -> pl.DataFrame:
    """Three weeks with the real 2026 shapes, verified against nflverse."""
    return pl.DataFrame([
        # week 1 — opens on a WEDNESDAY, and Thursday is 20:35 not 20:15
        _game("w1a", 1, "2026-09-09", "20:20", "NE", "SEA"),
        _game("w1b", 1, "2026-09-10", "20:35", "SF", "LA"),      # nflverse spells it LA
        _game("w1c", 1, "2026-09-13", "13:00", "CHI", "CAR"),
        _game("w1d", 1, "2026-09-13", "16:25", "DEN", "LAC"),
        _game("w1e", 1, "2026-09-14", "20:15", "NYJ", "BUF"),
        # week 16 — my SEMIFINAL, with three Christmas Day games
        _game("w16a", 16, "2026-12-24", "20:15", "KC", "DAL"),
        _game("w16b", 16, "2026-12-25", "13:00", "GB", "DET"),
        _game("w16c", 16, "2026-12-25", "16:30", "PHI", "WAS"),
        _game("w16d", 16, "2026-12-25", "20:15", "SEA", "SF"),
        _game("w16e", 16, "2026-12-27", "13:00", "NYG", "MIN"),
        _game("w16f", 16, "2026-12-28", "20:15", "HOU", "TEN"),
        # week 5 — my fantasy bye
        _game("w5a", 5, "2026-10-08", "20:15", "ATL", "NO"),
        _game("w5b", 5, "2026-10-11", "13:00", "CLE", "PIT"),
    ])


@pytest.fixture(scope="module")
def kickoffs(schedule) -> pl.DataFrame:
    return CK.kickoff_table(2026, schedule)


# ─── F9: the lock calendar ──────────────────────────────────────────────────
def test_the_rams_are_normalised_at_the_nflverse_boundary(kickoffs):
    """nflverse spells them LA; ESPN and the crosswalk say LAR. The trap has bitten
    this project three times, and 2026's SECOND game is SF at LA on Thursday
    night — so an un-normalised join loses a lock time in week 1."""
    wk1 = kickoffs.filter(pl.col("week") == 1)
    assert "LAR" in wk1["team"].to_list()
    assert "LA" not in wk1["team"].to_list()


def test_week_one_opens_on_a_wednesday(kickoffs):
    """§9.1's fixed Thursday slot misses the first lock of the 2026 season."""
    windows = CK.lock_windows(1, None, 2026, kickoffs)
    first = windows[0].at.astimezone(ET)
    assert first.strftime("%A") == "Wednesday"
    assert first.strftime("%H:%M") == "20:20"
    # and Thursday is 20:35, not the 20:15 a hardcoded rule would assume
    assert windows[1].at.astimezone(ET).strftime("%A %H:%M") == "Thursday 20:35"


def test_the_semifinal_has_three_christmas_day_locks(kickoffs):
    """Week 16 is my semifinal. Four fixed weekly slots miss all three."""
    windows = CK.lock_windows(16, None, 2026, kickoffs)
    christmas = [w for w in windows
                 if w.at.astimezone(ET).date() == dt.date(2026, 12, 25)]
    assert len(christmas) == 3
    assert [w.at.astimezone(ET).strftime("%H:%M") for w in christmas] == [
        "13:00", "16:30", "20:15"]


def test_windows_are_restricted_to_teams_i_actually_roster(kickoffs):
    """A checkpoint for a window I have no player in is pure noise."""
    everything = CK.lock_windows(16, None, 2026, kickoffs)
    mine = CK.lock_windows(16, {"GB", "TEN"}, 2026, kickoffs)
    assert len(everything) > len(mine) == 2
    assert {t for w in mine for t in w.teams} == {"GB", "TEN"}


def test_a_partial_schedule_is_refused(schedule):
    """A lock calendar built from a schedule missing kickoff times silently
    drops those windows — which reads as 'no game' rather than as an error."""
    holed = schedule.with_columns(
        pl.when(pl.col("game_id") == "w1a").then(None)
        .otherwise(pl.col("gametime")).alias("gametime")
    )
    with pytest.raises(CK.ClockError, match="no kickoff time"):
        CK.kickoff_table(2026, holed)


def test_current_week_follows_the_schedule_not_the_calendar(kickoffs):
    before = dt.datetime(2026, 9, 8, 12, 0, tzinfo=ET)
    midweek = dt.datetime(2026, 9, 10, 9, 0, tzinfo=ET)     # after Wed, before Thu
    after1 = dt.datetime(2026, 9, 15, 12, 0, tzinfo=ET)
    assert CK.current_week(2026, before, kickoffs) == 1
    assert CK.current_week(2026, midweek, kickoffs) == 1
    assert CK.current_week(2026, after1, kickoffs) == 5     # next week WITH games
    assert CK.current_week(2026, dt.datetime(2027, 3, 1, tzinfo=ET), kickoffs) is None


def test_my_bye_weeks_refuse_to_produce_a_lineup():
    """§10 lists a lineup set for week 5 or 14 as a sign something is broken."""
    assert CK.is_my_bye(5) and CK.is_my_bye(14)
    assert not CK.is_my_bye(6)
    with pytest.raises(CK.ClockError, match="fantasy byes"):
        CK.assert_not_my_bye(14)


# ─── checkpoints ────────────────────────────────────────────────────────────
def test_checkpoints_cover_every_window_but_only_one_is_unconditional(kickoffs):
    """Checkpoints are not emails. Nine messages a week is the fatigue that
    stops a digest being read by October (§6.4)."""
    cps = CK.checkpoints_for_week(16, None, 2026, kickoffs)
    kinds = [c.kind for c in cps]
    assert kinds.count("advisory") == 1
    n_windows = len(CK.lock_windows(16, None, 2026, kickoffs))
    assert kinds.count("confirm") == n_windows
    assert kinds.count("inactives") == n_windows
    assert sum(c.unconditional for c in cps) == 1


def test_the_advisory_leads_the_first_lock_of_the_week(kickoffs):
    cps = CK.checkpoints_for_week(1, None, 2026, kickoffs)
    adv = next(c for c in cps if c.kind == "advisory")
    first = CK.lock_windows(1, None, 2026, kickoffs)[0]
    assert adv.window == first.at
    assert adv.runway == CK.ADVISORY_LEAD
    # ... which in week 1 is a TUESDAY, because the opener is Wednesday
    assert adv.due_at.astimezone(ET).strftime("%A") == "Tuesday"


def test_inactives_fire_at_kickoff_minus_75_not_at_a_fixed_hour(kickoffs):
    """Inactives drop at kickoff minus 90. §9.1's fixed 11:15 is fifteen minutes
    early for the 1pm slate and hours late for a 09:30 London game."""
    cps = CK.checkpoints_for_week(16, None, 2026, kickoffs)
    inact = sorted((c for c in cps if c.kind == "inactives"),
                   key=lambda c: c.due_at)
    for c in inact:
        assert c.window - c.due_at == dt.timedelta(minutes=75)
    xmas_late = [c for c in inact
                 if c.window.astimezone(ET).strftime("%m-%d %H:%M") == "12-25 20:15"]
    assert xmas_late and xmas_late[0].due_at.astimezone(ET).strftime("%H:%M") == "19:00"


def test_a_checkpoint_fires_exactly_once_across_consecutive_ticks(kickoffs):
    """The container wakes every fifteen minutes; a double-send is as bad as a
    miss, so the due window is half-open."""
    cps = CK.checkpoints_for_week(1, None, 2026, kickoffs)
    target = cps[0]
    fired = 0
    t = target.due_at - dt.timedelta(hours=1)
    while t <= target.due_at + dt.timedelta(hours=1):
        fired += len([c for c in CK.due(cps, t) if c is target])
        t += CK.TICK
    assert fired == 1


def test_a_quiet_tick_returns_nothing(kickoffs):
    cps = CK.checkpoints_for_week(1, None, 2026, kickoffs)
    quiet = dt.datetime(2026, 9, 1, 3, 0, tzinfo=ET)
    assert CK.due(cps, quiet) == []


def test_timezone_is_asserted_not_assumed(monkeypatch):
    """F7: a container defaults to UTC and nothing about that looks wrong."""
    monkeypatch.setenv("TZ", "America/New_York")
    assert CK.assert_timezone() == "America/New_York"
    monkeypatch.setenv("TZ", "UTC")
    with pytest.raises(CK.ClockError, match="daylight"):
        CK.assert_timezone()
    assert "WRONG" in CK.assert_timezone(strict=False)


# ─── the value layer ────────────────────────────────────────────────────────
def _roster(**over) -> pl.DataFrame:
    spec = [("q1", "QB", 24.0), ("q2", "QB", 19.0), ("q3", "QB", 15.0),
            ("r1", "RB", 18.0), ("r2", "RB", 14.0), ("r3", "RB", 11.0),
            ("w1", "WR", 17.0), ("w2", "WR", 16.0), ("w3", "WR", 12.0),
            ("t1", "TE", 10.0), ("k1", "K", 8.0), ("d1", "DST", 7.0)]
    df = pl.DataFrame({"canonical_id": [s[0] for s in spec],
                       "position": [s[1] for s in spec],
                       "weekly_points": [s[2] for s in spec],
                       "bye_week": [over.get(s[0], V.NO_BYE) for s in spec]})
    return df


def test_value_layer_agrees_with_the_polars_lineup_solver():
    """Two implementations of the same rule is one implementation too many
    unless they are checked against each other."""
    from ff_agent.season import lineup as LU
    r = _roster()
    assert V.roster_value(r, play_weeks=(1, 2, 3)).mean == pytest.approx(
        LU.lineup_points(r)
    )


def test_a_bye_free_roster_has_exactly_zero_bye_cost():
    """The sentinel for 'no bye' must not collide with the probe week used to
    price the bye — if it does, the cost comes back as the whole lineup."""
    assert V.roster_value(_roster(), play_weeks=(1, 2, 3)).bye_cost == 0.0


def test_a_bye_in_a_week_i_play_costs_a_real_measurable_amount():
    r = _roster(q1=2)
    assert V.roster_value(r, play_weeks=(1, 2, 3)).bye_cost == pytest.approx(
        (24.0 - 15.0) / 3
    )


def test_a_bye_in_MY_bye_week_costs_nothing_which_is_all_of_2_1():
    """§2.1's free-bye arbitrage, falling out of the arithmetic rather than
    being a hand-tuned nudge. Nobody's rankings price this because it is a
    property of MY schedule, not of the player."""
    assert V.roster_value(_roster(q1=5), play_weeks=(1, 2, 3)).bye_cost == 0.0
    assert V.roster_value(_roster(q1=14), play_weeks=(1, 2, 3, 14)).bye_cost > 0


def test_a_bench_upgrade_is_worth_exactly_zero():
    """§9.3's 'bench upgrades ≈ 0' and §3.2's 'don't hoard handcuffs', enforced
    by arithmetic rather than by a rule somebody has to remember."""
    before = _roster()
    after = before.with_columns(
        pl.when(pl.col("canonical_id") == "q3").then(16.5)
        .otherwise(pl.col("weekly_points")).alias("weekly_points")
    )
    assert V.weekly_delta(before, after, play_weeks=(1, 2, 3)) == 0.0


def test_a_starting_upgrade_is_worth_what_it_should_be():
    before = _roster()
    after = before.with_columns(
        pl.when(pl.col("canonical_id") == "t1").then(20.0)
        .otherwise(pl.col("weekly_points")).alias("weekly_points")
    )
    assert V.weekly_delta(before, after, play_weeks=(1, 2, 3)) == pytest.approx(10.0)


def test_an_unprojected_position_is_refused_not_approximated():
    """§10 and M9's IDP refusal: a confident wrong slotting beats no answer only
    if you never find out."""
    bad = _roster().with_columns(
        pl.when(pl.col("canonical_id") == "q1").then(pl.lit("LB"))
        .otherwise(pl.col("position")).alias("position")
    )
    with pytest.raises(ValueError, match="LB"):
        V.roster_value(bad)


def test_no_remaining_weeks_is_refused():
    with pytest.raises(ValueError, match="play_weeks is empty"):
        V.roster_value(_roster(), play_weeks=())


def test_posture_reports_both_crossovers_and_they_disagree_in_a_real_band():
    """M7 measured the title crossover at +15.9 and the top-2 one at +12.1, so
    there is a band where chasing the bye and chasing the title want opposite
    things. §2.4 states the rule without qualification and is right on one side."""
    mid = V.posture(130 + 13.0, 130)
    assert mid["chase_variance_for_title"] and not mid["chase_variance_for_top2"]


# ─── ROS ────────────────────────────────────────────────────────────────────
def _anchor() -> pl.DataFrame:
    return pl.DataFrame({
        "canonical_id": ["a", "b", "c"], "position": ["QB", "QB", "RB"],
        "team": ["BUF", "BAL", "SF"], "ros_points": [180.0, 170.0, 120.0],
        "games_remaining": [9, 9, 9],
        "sacks_over_expected_per_game": [1.5, -1.0, None],
    })


def test_the_sack_term_can_reorder_two_quarterbacks():
    """§3.3's edge, in points. Consensus prices sacks at zero — no other league
    scores them — and M3b measured SOE persisting at 0.434, four times TDOE's
    0.104. Two QBs ten points apart on ESPN's projection end up level."""
    out = R.build(_anchor(), from_week=9).frame
    got = dict(zip(out["canonical_id"].to_list(), out["ros_points"].to_list()))
    assert got["a"] == pytest.approx(180 - 0.434 * 1.5 * 9, abs=0.01)
    assert got["b"] == pytest.approx(170 + 0.434 * 1.0 * 9, abs=0.01)
    assert abs(got["a"] - got["b"]) < 1.0          # a ten-point gap, closed


def test_the_sack_term_is_quarterbacks_only():
    out = R.build(_anchor(), from_week=9).frame
    rb = out.filter(pl.col("canonical_id") == "c")
    assert rb["sack_correction"][0] == 0.0


def test_the_kicker_bucket_espn_cannot_express():
    """M3: ESPN merges 50-59 and 60+ into one bucket; §1 pays 5 and 6."""
    a = _anchor().with_columns(pl.Series("expected_fg_60_plus", [None, None, 3.0]))
    a = a.with_columns(pl.when(pl.col("canonical_id") == "c").then(pl.lit("K"))
                       .otherwise(pl.col("position")).alias("position"))
    out = R.build(a, from_week=9).frame
    assert out.filter(pl.col("canonical_id") == "c")["kicker_correction"][0] == 3.0


def test_model_weight_ships_at_zero_and_says_so():
    """M3's 0.12 was fitted for PRESEASON season-long projections on rank
    correlation. Carrying it to an in-season points task is the unjustified
    transfer this project keeps refusing to make."""
    p = R.build(_anchor(), from_week=9)
    assert p.model_weight == 0.0
    assert any("gate sets this" in n for n in p.notes)


def test_a_fan_out_in_the_model_join_is_refused():
    """§0.2 is not only about resolution. M7's validator found model.project
    emitting one row per (player, prior team), which put fifteen players on the
    board four times each."""
    model = pl.DataFrame({"canonical_id": ["a", "a"], "ros_points": [1.0, 2.0]})
    with pytest.raises(R.ROSError, match="row count"):
        R.blend(_anchor(), model, weight=0.5)


def test_duplicate_players_in_the_anchor_are_refused():
    dupe = pl.concat([_anchor(), _anchor().head(1)])
    with pytest.raises(R.ROSError, match="more than once"):
        R.build(dupe, from_week=9)


def test_weekly_rate_divides_by_GAMES_not_weeks():
    """M7 divided by 16 and inflated every player by 6.25%. The NFL plays 18
    weeks and 17 GAMES: the bye is the eighteenth week, not one of the seventeen."""
    out = R.build(_anchor(), from_week=9).frame
    row = out.filter(pl.col("canonical_id") == "c")
    assert row["weekly_points"][0] == pytest.approx(120.0 / 9)


# ─── league state ───────────────────────────────────────────────────────────
def test_starter_holes_finds_the_acute_2qb_case():
    assert ST.starter_holes({"QB": 2, "RB": 3, "WR": 3, "TE": 1, "K": 1, "DST": 1}) == []
    assert ST.starter_holes({"QB": 1, "RB": 3, "WR": 3, "TE": 1, "K": 1, "DST": 1}) == ["QB"]
    assert ST.starter_holes({"QB": 0, "RB": 3, "WR": 3, "TE": 1, "K": 1, "DST": 1}) == ["QB", "QB"]


def test_flex_hole_is_found_when_the_bodies_run_out():
    assert ST.starter_holes({"QB": 2, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}) == ["FLEX"]


def test_position_maxima_are_enforced():
    """Without them M7's failure returns: eight QBs, no TE, ~50 points a week."""
    assert not ST.can_add({"QB": 4}, "QB")
    assert ST.can_add({"QB": 3}, "QB")


def test_a_drop_that_breaks_the_lineup_is_not_offered():
    """§10 lists an unfillable starting lineup as a sign something is broken, so
    the waiver engine is never allowed to propose one."""
    minimal = pl.DataFrame({
        "canonical_id": list("abcdefghij"),
        "position": ["QB", "QB", "RB", "RB", "WR", "WR", "TE", "K", "DST", "RB"]})
    assert ST.droppable(minimal).height == 0
    plus = pl.concat([minimal, pl.DataFrame({"canonical_id": ["z"], "position": ["WR"]})])
    assert ST.droppable(plus)["canonical_id"].to_list() != []


def test_missing_results_for_a_played_week_block_the_digest():
    """§10: an optimizer silently running week-3 data in week 8 is worse than no
    optimizer. In-season that is a specific, checkable assertion."""
    have = pl.DataFrame({"week": [1, 2], "team": ["A", "A"], "points": [1.0, 2.0]})
    ST.assert_covers_completed_week(have, 2)
    with pytest.raises(ST.StateError, match=r"weeks \[3, 4\]"):
        ST.assert_covers_completed_week(have, 4)


def test_unresolved_players_are_named_not_counted():
    """A count is ignorable; a name is actionable. §0.2's in-season shape."""
    un = pl.DataFrame({"name": ["Practice Squad Guy"], "position": ["TE"],
                       "espn_id": ["999"]})
    note = ST.unresolved_note(un, "free agents")
    assert "Practice Squad Guy" in note and "override" in note
    assert ST.unresolved_note(un.head(0), "free agents") is None
