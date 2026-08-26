"""M10b-4 — the lineup sequence: locks, the Thursday counterfactual, F10.

The scenario is deliberately the one that breaks a naive implementation: a
Thursday player who must be committed before Sunday's information arrives.
"""
import datetime as dt

import polars as pl
import pytest

from ff_agent.inseason import availability as AV
from ff_agent.inseason import clock as CK
from ff_agent.inseason import lineup as L

ET = CK.ET


def _game(gid, wk, day, time, away, home):
    return {"game_id": gid, "season": 2026, "week": wk, "game_type": "REG",
            "gameday": day, "gametime": time,
            "weekday": dt.date.fromisoformat(day).strftime("%A"),
            "away_team": away, "home_team": home}


@pytest.fixture(scope="module")
def kickoffs():
    return CK.kickoff_table(2026, pl.DataFrame([
        _game("a", 7, "2026-10-22", "20:15", "BUF", "NYJ"),     # Thursday
        _game("b", 7, "2026-10-25", "13:00", "KC", "DEN"),      # Sunday early
        _game("c", 7, "2026-10-25", "16:25", "SF", "SEA"),      # Sunday late
        _game("d", 7, "2026-10-26", "20:15", "MIA", "LA"),      # Monday
    ]))


WEDNESDAY = dt.datetime(2026, 10, 21, 12, 0, tzinfo=ET)
FRIDAY = dt.datetime(2026, 10, 23, 9, 0, tzinfo=ET)             # after TNF


def roster(thu_pts=19.0, thu_pos="WR", thu_team="BUF"):
    """A REAL roster: 10 starting slots and a bench behind them.

    The depth is load-bearing rather than decorative. An earlier version of this
    fixture had exactly ten playable bodies, so benching the Thursday player left
    a slot literally unfillable — and the "option value of waiting" came out
    NEGATIVE, because waiting meant waiting for nobody. That is a real property
    of a thin roster, not a bug, but it is not the case these tests are about.
    """
    rows = [
        ("thu", "Thursday Guy", thu_pos, thu_team, thu_pts),
        ("q1", "QB One", "QB", "KC", 22.0), ("q2", "QB Two", "QB", "SF", 20.0),
        ("q3", "QB Three", "QB", "DEN", 13.0),
        ("r1", "RB One", "RB", "KC", 15.0), ("r2", "RB Two", "RB", "SEA", 13.0),
        ("r3", "RB Three", "RB", "SF", 10.5), ("r4", "RB Four", "RB", "DEN", 8.0),
        ("w1", "WR One", "WR", "DEN", 16.0), ("w2", "WR Two", "WR", "SF", 14.0),
        ("w3", "WR Three", "WR", "SEA", 11.0), ("w4", "WR Four", "WR", "KC", 9.5),
        ("t1", "TE One", "TE", "KC", 9.0), ("t2", "TE Two", "TE", "SEA", 6.0),
        ("k1", "K One", "K", "DEN", 8.0), ("d1", "DST One", "DST", "SEA", 7.0),
    ]
    return pl.DataFrame({
        "canonical_id": [r[0] for r in rows], "name": [r[1] for r in rows],
        "position": [r[2] for r in rows], "team": [r[3] for r in rows],
        "weekly_points": [float(r[4]) for r in rows]})


def injury(cid, status, practice="Limited Participation in Practice"):
    return pl.DataFrame({"gsis_id": [cid], "report_status": [status],
                         "practice_status": [practice]})


# ─── lock state ─────────────────────────────────────────────────────────────
def test_nothing_is_locked_before_the_first_kickoff(kickoffs):
    r = L.attach_locks(roster(), 7, kickoffs, WEDNESDAY)
    assert not r["locked"].any()
    assert r["plays_this_week"].all()


def test_the_thursday_player_locks_and_the_rest_do_not(kickoffs):
    r = L.attach_locks(roster(), 7, kickoffs, FRIDAY)
    locked = set(r.filter(pl.col("locked"))["canonical_id"].to_list())
    assert locked == {"thu"}


def test_a_player_on_an_nfl_bye_is_distinguished_from_a_locked_one(kickoffs):
    """Conflating the two lets a bye player sit "safely" in a starting slot."""
    r = L.attach_locks(roster(thu_team="ARI"), 7, kickoffs, FRIDAY)
    row = r.filter(pl.col("canonical_id") == "thu")
    assert not row["plays_this_week"][0]
    assert not row["locked"][0]


def test_my_bye_weeks_produce_no_lineup_at_all(kickoffs):
    """§10 lists a lineup set for week 5 or 14 as a sign something is broken."""
    with pytest.raises(CK.ClockError, match="fantasy byes"):
        L.build(roster(), 14, kickoffs, WEDNESDAY)


# ─── the Thursday counterfactual ────────────────────────────────────────────
@pytest.mark.parametrize("pts,expected", [(19.0, "START"), (6.0, "BENCH")])
def test_the_call_follows_the_counterfactual_not_the_ranking(kickoffs, pts, expected):
    plan = L.build(roster(pts), 7, kickoffs, WEDNESDAY)
    assert plan.decisions[0].call == expected


def test_the_bar_is_higher_than_beating_todays_best_alternative(kickoffs):
    """`E[p] > E[best Sunday alternative under SUNDAY information]` is a strictly
    higher bar than `E[p] > E[best alternative I can see today]`, because the
    Sunday choice gets to be made knowing more.

    With bench depth behind him, the Thursday player at 12.0 nominally beats the
    next flex body (WR3 at 11.0) by a full point. The counterfactual gives back
    part of that point, because locking the slot forfeits the chance to react.
    """
    plan = L.build(roster(12.0), 7, kickoffs, WEDNESDAY)
    d = plan.decisions[0]
    naive_edge = 12.0 - 11.0
    assert 0 < d.delta < naive_edge, (
        "the option cost has to make the bar higher than the naive comparison, "
        "or the whole counterfactual is decorative"
    )


def test_F10s_positional_asymmetry_actually_flips_the_decision(kickoffs):
    """The headline. Identical points, identical designation, opposite calls —
    because a Questionable QB sits 0.735 of the time against a Questionable RB's
    0.360, and in a 2-QB league that is where it matters."""
    qb = L.build(roster(18.0, "QB"), 7, kickoffs, WEDNESDAY,
                 injuries=injury("thu", "Questionable"))
    rb = L.build(roster(18.0, "RB"), 7, kickoffs, WEDNESDAY,
                 injuries=injury("thu", "Questionable"))
    assert qb.decisions[0].call == "BENCH"
    assert rb.decisions[0].call == "START"
    assert qb.decisions[0].p_out > 2 * rb.decisions[0].p_out


def test_an_out_player_is_never_committed(kickoffs):
    plan = L.build(roster(25.0), 7, kickoffs, WEDNESDAY,
                   injuries=injury("thu", "Out"))
    assert plan.decisions[0].call == "BENCH"
    assert plan.decisions[0].p_out == 1.0


def test_a_pinned_player_who_sits_wastes_the_slot_and_that_is_the_option_cost(kickoffs):
    """A pinned player who does not play scores zero AND spends the slot. A free
    player who does not play is simply passed over. That asymmetry IS the cost."""
    r = L.attach_locks(roster(18.0), 7, kickoffs, WEDNESDAY)
    r = AV.attach(r, injury("thu", "Questionable"))
    committed = L.commitment_value(r, {"thu": "WR"}, n_draws=8000)
    waited = L.commitment_value(r, {}, exclude={"thu"}, n_draws=8000)
    healthy = AV.attach(L.attach_locks(roster(18.0), 7, kickoffs, WEDNESDAY), None)
    committed_healthy = L.commitment_value(healthy, {"thu": "WR"}, n_draws=8000)
    # being questionable costs him value...
    assert committed < committed_healthy
    # ...and at 0.23 to sit, an 18-point WR is still worth committing over the
    # bench. The asymmetry shows up as a SHRUNK edge, not an inverted one.
    assert committed > waited
    assert (committed - waited) < (committed_healthy - waited)


def test_toss_ups_are_reported_as_toss_ups(kickoffs):
    """Advising a swap worth a tenth of a point is how a digest loses its reader."""
    d = L.Decision("x", "X", "WR", "BUF", "WR", WEDNESDAY, 100.0, 100.05, 0.02)
    assert d.call == "toss-up" and "option" in d.reason()


# ─── the resulting plan ─────────────────────────────────────────────────────
def test_locked_slots_are_carried_and_never_re_decided(kickoffs):
    """A locked slot cannot be advised on — the state machine makes it
    unrepresentable rather than merely discouraged."""
    r = roster(19.0).with_columns(
        pl.when(pl.col("canonical_id") == "thu").then(pl.lit("WR"))
        .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("lineup_slot"))
    plan = L.build(r, 7, kickoffs, FRIDAY)
    assert plan.pins.get("thu") == "WR"
    assert all(d.canonical_id != "thu" for d in plan.decisions)


def test_the_plan_reports_the_next_deadline(kickoffs):
    """Every email leads with this."""
    plan = L.build(roster(), 7, kickoffs, WEDNESDAY)
    assert plan.next_lock is not None
    assert plan.next_lock.at.astimezone(ET).strftime("%A %H:%M") == "Thursday 20:15"
    assert plan.runway.total_seconds() > 0
    assert plan.summary()["minutes_to_next_lock"] > 0


def test_a_starter_likely_to_sit_raises_an_alarm(kickoffs):
    """The kicker has no replacement on this roster, so he starts however likely
    he is to sit. That is exactly when the digest has to say so out loud: the
    recommendation cannot change, but knowing my kicker is not playing is worth
    a waiver claim on Tuesday."""
    plan = L.build(roster(), 7, kickoffs, WEDNESDAY,
                   injuries=injury("k1", "Doubtful"))
    assert any("to sit" in a for a in plan.alarms), plan.alarms
    assert "K One" in " ".join(plan.alarms)


def test_a_missing_injury_report_is_said_out_loud(kickoffs):
    """A silent default of 'everyone plays' during a week the report was
    unavailable is the stale-data failure §10 is about."""
    plan = L.build(roster(), 7, kickoffs, WEDNESDAY, injuries=None)
    assert any("no injury report" in n for n in plan.notes)


def test_a_lineup_arriving_after_its_deadline_raises_an_alarm(kickoffs):
    late = dt.datetime(2026, 10, 22, 20, 5, tzinfo=ET)          # 10 min to kickoff
    plan = L.build(roster(), 7, kickoffs, late)
    assert any("lands after its deadline" in a for a in plan.alarms)


def test_the_lineup_fills_every_slot_it_can(kickoffs):
    plan = L.build(roster(), 7, kickoffs, WEDNESDAY)
    from ff_agent.config import STARTER_SLOTS
    assert plan.starters.height == sum(STARTER_SLOTS.values())


# ─── the information effect, stated rather than implied ─────────────────────
def test_information_value_is_exactly_zero_under_expected_points():
    """A result, not an omission. The expected-points-optimal Sunday lineup does
    not depend on the Thursday realisation, so knowing it changes no decision.
    It becomes positive only under P(beat this opponent) — which is M10b-4's
    gate, and is refused rather than faked until then."""
    assert L.information_value("expected_points") == 0.0
    with pytest.raises(NotImplementedError, match="variance"):
        L.information_value("win_prob")


# ─── the availability model itself ──────────────────────────────────────────
def test_measured_rates_are_pinned_so_a_refit_is_visible():
    assert AV.QUESTIONABLE_BY_POSITION["QB"] == 0.735
    assert AV.QUESTIONABLE_BY_POSITION["RB"] == 0.360
    assert AV.BASELINE_ABSENCE == 0.149


def test_practice_participation_is_deliberately_not_a_third_scale():
    """Full (0.421) and limited (0.404) are inseparable, so adjusting on them
    would be noise dressed as precision. Only did-not-practice separates."""
    q_full = AV.p_out("Questionable", "WR", "Full Participation in Practice")
    q_lim = AV.p_out("Questionable", "WR", "Limited Participation in Practice")
    assert q_full == q_lim
    assert AV.p_out("Questionable", "WR", "Did Not Participate In Practice") > q_lim


def test_the_measurement_bias_is_subtracted_by_default():
    raw = AV.p_out("Questionable", "QB", corrected=False)
    corrected = AV.p_out("Questionable", "QB")
    assert raw == 0.735
    assert corrected == pytest.approx(0.735 - AV.BASELINE_ABSENCE)


def test_a_healthy_player_is_not_treated_as_risk_free():
    """People get hurt in warmups; a floor of exactly zero would let the
    optimizer treat a start as free."""
    assert 0 < AV.p_out(None, "WR") < 0.05


def test_an_injury_fan_out_is_refused():
    r = roster()
    dupe = pl.DataFrame({"gsis_id": ["thu", "thu"],
                         "report_status": ["Questionable", "Out"],
                         "practice_status": [None, None]})
    # unique() keeps one, so the join stays 1:1 — that is the point of the guard
    assert AV.attach(r, dupe).height == r.height
