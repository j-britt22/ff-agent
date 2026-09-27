"""M10b-5 — the two-sided trade search, the playoff view, and §2.2's week 14."""
import polars as pl
import pytest

from ff_agent.inseason import playoffs as P
from ff_agent.inseason import trades as T

WEEKS = tuple(range(6, 14))


def mk(ids, names, pos, teams, pts, anchor=None, byes=None, sack=None):
    return pl.DataFrame({
        "canonical_id": ids, "name": names, "position": pos, "team": teams,
        "weekly_points": [float(x) for x in pts],
        "bye_week": byes or [0] * len(ids),
        "anchor_points": [float(x) * 10 for x in (anchor or pts)],
        "ros_points": [float(x) * 10 for x in pts],
        "anchor_weekly": [float(x) for x in (anchor or pts)],
        "sack_correction": sack or [0.0] * len(ids),
        "kicker_correction": [0.0] * len(ids)})


@pytest.fixture
def me():
    """RB-rich with a fourth back on the bench, and thin at receiver."""
    return mk(["r1", "r2", "r3", "r4", "q1", "q2", "w1", "w2", "t1", "k1", "d1"],
              ["MyRB1", "MyRB2", "MyRB3", "MyRB4", "MyQB1", "MyQB2", "MyWR1",
               "MyWR2", "MyTE", "MyK", "MyD"],
              ["RB", "RB", "RB", "RB", "QB", "QB", "WR", "WR", "TE", "K", "DST"],
              ["SF"] * 11, [18, 16, 15, 14, 22, 20, 9, 8, 8, 8, 7])


@pytest.fixture
def them():
    """The mirror image: four receivers, the fourth benched, thin at back."""
    return mk(["tw1", "tw2", "tw3", "tw4", "tq1", "tq2", "tr1", "tr2", "tt1",
               "tk1", "td1"],
              ["TheirWR1", "TheirWR2", "TheirWR3", "TheirWR4", "TheirQB1",
               "TheirQB2", "TheirRB1", "TheirRB2", "TheirTE", "TheirK", "TheirD"],
              ["WR", "WR", "WR", "WR", "QB", "QB", "RB", "RB", "TE", "K", "DST"],
              ["KC"] * 11, [18, 16, 15, 14, 21, 19, 15, 8, 8, 8, 7])


def search(me, them, manager="Camden Sims"):
    return T.search_partner(me, them, WEEKS, WEEKS, "Hodor's Hodors", manager)


# ─── the search ─────────────────────────────────────────────────────────────
def test_a_mutual_fit_is_found(me, them):
    props = search(me, them)
    assert props
    assert all(p.acceptable for p in props)


def test_only_trades_positive_for_BOTH_sides_are_proposed(me, them):
    """A trade nobody would accept is not a recommendation."""
    for p in search(me, them):
        assert p.my_weekly_delta > 0 and p.their_weekly_delta > 0


def test_the_best_package_gives_from_my_bench_and_takes_a_starter(me, them):
    """My fourth back contributes exactly zero to my lineup; their WR1 replaces
    a 9-point starter. §9.4's 'depth is cheap in a nine-team league'."""
    best = search(me, them)[0]
    assert best.i_give == ["MyRB4"]
    assert best.i_get == ["TheirWR1"]
    assert best.my_weekly_delta == pytest.approx(10.0)


def test_consolidation_shows_up_without_a_rule_for_it(me, them):
    """§9.4: 'consolidating two mid pieces into one stud is usually favourable.'
    It falls out of the lineup solver, because two bench bodies contribute zero
    and one starter contributes."""
    props = search(me, them)
    assert any(len(p.i_give) == 2 and len(p.i_get) == 1 for p in props)


def test_their_side_is_scored_under_THEIR_value_function(me, them):
    """The mechanism, not a convenience. If both sides used our numbers there
    would be no trade to find; if both used consensus there would be no edge."""
    # make one of their receivers look far better to US than to THEM
    skewed = them.with_columns(
        pl.when(pl.col("canonical_id") == "tw1").then(3.0)
        .otherwise(pl.col("anchor_weekly")).alias("anchor_weekly"))
    props = search(me, skewed)
    got_tw1 = [p for p in props if "TheirWR1" in p.i_get]
    assert got_tw1, "a player they undervalue is exactly who we should be asking for"
    assert got_tw1[0].their_weekly_delta > 0


def test_an_illegal_package_is_never_proposed(me, them):
    """§1's maxima and starter feasibility. Without them M7's failure returns."""
    for p in search(me, them):
        assert len(p.i_give) <= T.MAX_PACKAGE
        assert len(p.i_get) <= T.MAX_PACKAGE


def test_a_double_up_opponent_is_flagged(me, them):
    """§2.3: strengthening one of the four costs me twice, in two of my twelve
    games."""
    assert search(me, them, "Camden Sims")[0].double_up
    assert not search(me, them, "Hanna Rogo")[0].double_up


def test_a_proposal_is_copy_pasteable_and_nothing_sends_it(me, them):
    """§0.1. I send it."""
    msg = search(me, them)[0].message()
    assert "would you do" in msg and "MyRB4" in msg


# ─── naming the wedge ───────────────────────────────────────────────────────
def test_the_sack_wedge_is_named_in_words(them):
    sacky = them.with_columns(
        pl.when(pl.col("canonical_id") == "tq1").then(-4.0)
        .otherwise(0.0).alias("sack_correction"))
    w = T.wedges(sacky.filter(pl.col("canonical_id") == "tq1"), {5, 14})
    assert w and "sack" in str(w[0]) and "no other league" in str(w[0])


def test_the_free_bye_wedge_is_named(them):
    free = them.with_columns(
        pl.when(pl.col("canonical_id") == "tw1").then(5)
        .otherwise(0).alias("bye_week"))
    w = T.wedges(free.filter(pl.col("canonical_id") == "tw1"), {5, 14})
    assert w and "free to me" in str(w[0])


def test_a_player_with_no_league_specific_wedge_produces_none(them):
    """'Our model likes him' is not a wedge. A proposal that cannot name one is
    usually a search artefact rather than a disagreement about football."""
    assert T.wedges(them.head(3), {5, 14}) == []


# ─── profiling ──────────────────────────────────────────────────────────────
def test_the_league_profile_reports_every_roster(me, them):
    prof = T.league_profile(
        {"Me": me, "Hodors": them},
        {"Me": WEEKS, "Hodors": WEEKS}, {"Hodors": "Camden Sims"})
    assert prof.height == 2
    assert set(prof.columns) >= {"team", "weekly_points", "holes", "double_up"}
    assert prof["weekly_points"].to_list() == sorted(
        prof["weekly_points"].to_list(), reverse=True)


def test_surplus_is_defined_against_the_STARTING_lineup(me):
    """A third good back on a team starting two plus a flex is surplus; a second
    QB hole in a 2-QB league is acute."""
    p = T.profile(me, WEEKS, "Me")
    assert p["surplus"]["name"].to_list() == ["MyRB4"]
    assert p["holes"] == []


# ─── playoffs and week 14 ───────────────────────────────────────────────────
def test_the_playoff_view_scores_weeks_15_to_17_only(me):
    v = P.view(me)
    assert v.summary()["playoff_weeks"] == [15, 16, 17]
    assert v.weekly_points > 0


def test_a_convenient_bye_is_worth_nothing_in_the_bracket(me):
    """NFL byes do not exist that late, so §2.1's term correctly contributes
    zero — a player carried by a convenient bye is worth just his points here."""
    byed = me.with_columns(
        pl.when(pl.col("canonical_id") == "r1").then(5)
        .otherwise(0).alias("bye_week"))
    assert P.view(byed).weekly_points == pytest.approx(P.view(me).weekly_points)
    assert P.view(byed).bye_cost == 0.0


def test_rest_risk_is_flagged_for_clinched_teams(me):
    """§2.4 lists this beside schedule strength as a real criterion."""
    v = P.view(me, clinched_teams={"SF"})
    assert v.rest_risk
    assert any("rest starters" in n for n in v.notes)


def test_week14_sets_no_lineup_at_all(me, them):
    """§2.2: my record is locked, there is no game to lose. §10 lists a lineup
    set for week 14 as a sign something is broken."""
    plan = P.week14(me, them.head(4))
    assert plan.summary()["sets_a_lineup"] is False
    assert P.sets_no_lineup(14) and P.sets_no_lineup(5)
    assert not P.sets_no_lineup(13)


def test_week14_ranks_purely_on_the_bracket(me, them):
    """A player useful in week 14 and useless in weeks 15-17 is worth nothing
    here; a high-variance stash I would never start live is worth everything."""
    plan = P.week14(me, them.head(4))
    assert plan.drops.height and plan.targets.height
    assert plan.drops["name"].to_list()[0] == "MyD"      # lowest bracket value
    assert any("no lineup, no game" in n for n in plan.notes)
