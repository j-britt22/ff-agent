"""M10b-3 — the waiver engine, plus D/ST and K streaming."""
import polars as pl
import pytest

from ff_agent.inseason import dst as D
from ff_agent.inseason import kicker as K
from ff_agent.inseason import waivers as W

PLAY_WEEKS = tuple(range(6, 14))


def roster() -> pl.DataFrame:
    return pl.DataFrame({
        "canonical_id": ["q1", "q2", "r1", "r2", "r3", "w1", "w2", "w3", "t1",
                         "k1", "d1", "bench1"],
        "name": ["QB1", "QB2", "RB1", "RB2", "RB3", "WR1", "WR2", "WR3", "TE1",
                 "K1", "DST1", "Bench"],
        "position": ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE", "K",
                     "DST", "WR"],
        "team": ["BUF", "BAL", "SF", "KC", "DEN", "MIA", "GB", "NYJ", "LV",
                 "DAL", "SEA", "CHI"],
        "weekly_points": [22., 20., 15., 13., 9., 16., 14., 10., 8., 8., 7., 4.],
        "bye_week": [0] * 12,
    })


def free_agents(**over) -> pl.DataFrame:
    base = pl.DataFrame({
        "canonical_id": ["fa1", "fa2", "fa3"],
        "name": ["Waiver WR", "Bye-5 RB", "Bench Guy"],
        "position": ["WR", "RB", "WR"],
        "team": ["TB", "ARI", "CLE"],
        "weekly_points": [13.5, 12.0, 2.0],
        "bye_week": [9, 5, 11],
    })
    return base.with_columns(**over) if over else base


def waiver_order(my_rank=5):
    return pl.DataFrame({"team_id": [1, 2, 3], "fantasy_team": ["Me", "A", "B"],
                         "waiver_rank": [my_rank, 1, 2]})


RIVALS = {
    2: pl.DataFrame({"position": ["QB", "RB", "RB", "WR", "WR", "TE", "K", "DST"]}),
    3: pl.DataFrame({"position": ["QB", "QB", "RB", "RB", "WR", "WR", "TE", "K", "DST"]}),
}


def build(**kw):
    kw.setdefault("my_roster", roster())
    kw.setdefault("free_agents", free_agents())
    kw.setdefault("play_weeks", PLAY_WEEKS)
    kw.setdefault("week", 6)
    kw.setdefault("waiver_order", waiver_order())
    kw.setdefault("rival_rosters", RIVALS)
    kw.setdefault("my_team_id", 1)
    return W.build(**kw)


# ─── the six things that fall out of §9.3's rule ────────────────────────────
def test_a_player_who_never_starts_is_worth_exactly_nothing():
    """§9.3's 'bench upgrades ≈ 0' and §3.2's 'don't hoard handcuffs', enforced
    by arithmetic. The 2.0-point WR never appears."""
    names = [c.add_name for c in build().claims]
    assert "Bench Guy" not in names


def test_add_and_drop_are_scored_as_one_action():
    """The dropped player's own remaining starts are the cost, so the engine
    picks WHICH drop, not just whether to add."""
    c = build().claims[0]
    assert c.drop_id is not None
    assert c.drop_name == "RB3"        # the worst startable body, not the bench


def test_a_drop_that_breaks_the_starting_lineup_is_never_proposed():
    """§10 lists an unfillable lineup as a sign something is broken."""
    for c in build().claims:
        assert c.drop_name not in {"K1", "DST1", "TE1"}


def test_position_maxima_are_respected():
    """Without them M7's failure returns: eight QBs, no tight end."""
    qb_heavy = pl.DataFrame({
        "canonical_id": [f"q{i}" for i in range(4)] + ["r1", "r2", "w1", "w2",
                                                       "t1", "k1", "d1"],
        "name": [f"QB{i}" for i in range(4)] + ["RB1", "RB2", "WR1", "WR2",
                                               "TE1", "K1", "DST1"],
        "position": ["QB"] * 4 + ["RB", "RB", "WR", "WR", "TE", "K", "DST"],
        "team": ["BUF"] * 11,
        "weekly_points": [22., 20., 8., 7., 15., 13., 16., 14., 8., 8., 7.],
        "bye_week": [0] * 11})
    fa = free_agents().with_columns(
        pl.Series("position", ["QB", "QB", "QB"]),
        pl.Series("weekly_points", [30.0, 29.0, 28.0]))
    out = W.build(my_roster=qb_heavy, free_agents=fa, play_weeks=PLAY_WEEKS, week=6)
    assert all(c.position != "QB" for c in out.claims), (
        "the roster already holds the maximum four quarterbacks"
    )


def test_2_1s_free_bye_is_a_weekly_edge_and_says_so():
    """A property of MY schedule, not of the player, so nobody's rankings price
    it — and it applies on the wire every week, not only on draft day."""
    c = next(c for c in build().claims if c.add_name == "Bye-5 RB")
    assert c.free_bye
    assert any("free weeks" in r for r in c.reasons)


def test_a_low_probability_claim_stays_on_the_list():
    """§9.3: 'if you're 7th and three teams ahead share the gap, the claim likely
    fails — which costs nothing.' The last clause is the important half."""
    out = build(waiver_order=waiver_order(my_rank=9))
    assert out.claims
    longshots = [c for c in out.claims if c.p_success < 0.35]
    assert longshots
    assert any("costs nothing" in r for r in longshots[0].reasons)


def test_the_list_is_ordered_and_names_what_to_burn_priority_on():
    """ESPN processes claims in MY order and priority drops only on a success,
    so a single claim is strictly worse than a ranked list."""
    out = build()
    assert len(out.claims) >= 2
    deltas = [c.weekly_delta for c in out.claims]
    assert deltas == sorted(deltas, reverse=True)
    assert out.burn_priority_on is out.claims[0]


# ─── P(claim succeeds) and option value ─────────────────────────────────────
def test_rivals_ahead_of_me_with_a_matching_hole_lower_the_odds():
    holes = {2: ["WR"], 3: []}
    ranks = {2: 1, 3: 2}
    wants = W.p_claim_succeeds("WR", 5, holes, ranks)
    idle = W.p_claim_succeeds("TE", 5, holes, ranks)
    assert wants < idle


def test_rivals_behind_me_are_irrelevant():
    assert W.p_claim_succeeds("WR", 1, {2: ["WR"]}, {2: 5}) == 1.0


def test_option_value_is_cheap_in_a_nine_team_league():
    """§9.3 argues the queue is only nine long so you climb back fast. Written
    out, that becomes a number instead of an assertion."""
    at_one = W.option_value(8, 1, best_available_now=4.0)
    at_last = W.option_value(8, 9, best_available_now=4.0)
    assert at_last == 0.0
    assert at_one < 4.0, "holding priority must not be worth more than the player"


def test_holding_number_one_priority_is_called_out():
    out = build(waiver_order=waiver_order(my_rank=1))
    assert any("spend it more freely" in n for n in out.notes)


def test_an_empty_wire_says_so_rather_than_producing_filler():
    out = build(free_agents=free_agents().with_columns(
        pl.Series("weekly_points", [1.0, 1.0, 1.0])))
    assert out.claims == []
    assert any("bench-upgrades" in n for n in out.notes)


def test_missing_waiver_priority_is_stated_not_assumed():
    out = build(waiver_order=None, rival_rosters=None, my_team_id=None)
    assert any("waiver priority unavailable" in n for n in out.notes)


# ─── D/ST: the trap ADD-§F names ────────────────────────────────────────────
RULES = {"YA100": 5, "YA199": 3, "YA299": 2, "YA399": -1, "YA449": -3,
         "YA499": -5, "YA549": -6, "YA550": -7, "PA0": 5, "PA1": 4, "PA7": 3,
         "PA14": 1, "PA28": -1, "PA35": -3, "PA46": -5, "DEFSK": 1,
         "DEFINT": 2, "DEFFR": 2}


def test_two_defences_facing_the_same_scoring_split_on_VOLUME():
    """ADD-§F's whole point. Identical implied totals; a five-point gap that
    every other league's model scores as a tie, because they project points
    allowed and this league pays for YARDS."""
    methodical = D.score_option("SEA", "MET", 70, 5.9, 17, rules=RULES)
    three_and_out = D.score_option("NYJ", "TAO", 58, 4.6, 17, rules=RULES)
    assert methodical.points_bucket_points == three_and_out.points_bucket_points
    assert methodical.yards_bucket_points < three_and_out.yards_bucket_points
    assert three_and_out.total - methodical.total > 4.0


def test_the_trap_is_flagged_in_words_not_only_in_the_number():
    o = D.score_option("SEA", "MET", 70, 5.9, 17, rules=RULES)
    assert any("stalls" in w for w in o.warnings)


def test_the_zero_buckets_really_are_zero():
    """M2 verified all three of §1's zero-valued buckets are genuinely zero
    rather than transcription slips. ESPN omits zero rules from its payload, so
    'absent' and 'zero' look identical from the outside."""
    assert D._bucket_value(320, D.YARDS_ALLOWED_BUCKETS, RULES) == 0.0
    assert D._bucket_value(20, D.POINTS_ALLOWED_BUCKETS, RULES) == 0.0
    assert D._bucket_value(25, D.POINTS_ALLOWED_BUCKETS, RULES) == 0.0


def test_the_worst_case_is_reachable_and_is_minus_seven():
    o = D.score_option("X", "Y", 75, 7.5, 31, rules=RULES)
    assert o.yards_bucket_points == -7


def test_pace_converts_to_plays():
    assert D.plays_from_pace(26.0) == pytest.approx(3600 * 0.5 / 26.0)
    with pytest.raises(ValueError):
        D.plays_from_pace(0)


# ─── K: the bucket ESPN cannot express ──────────────────────────────────────
KRULES = {"FG0": 3, "FG40": 4, "FG50": 5, "FG60": 6, "FGM": -1, "PAT": 1}


def test_the_sixty_plus_bucket_is_worth_exactly_one_point_more():
    """M3: ESPN merges 50-59 and 60+ into one bucket; §1 pays 5 and 6."""
    assert K.long_range_premium(KRULES) == 1.0


def test_a_big_leg_beats_a_short_one_at_equal_volume():
    long_leg = K.score_week("Long", "DAL", "NYG", {"FG0": 1, "FG50": 1, "FG60": 0.5},
                            rules=KRULES)
    short_leg = K.score_week("Short", "GB", "CHI", {"FG0": 2.5}, rules=KRULES)
    assert long_leg.total > short_leg.total


def test_a_miss_costs_the_same_at_any_distance():
    """§1's flat -1 makes range strictly valuable and accuracy separately so."""
    near = K.score_week("A", "T", "O", {"FG0": 2}, miss_rate=0.0, rules=KRULES)
    near_miss = K.score_week("A", "T", "O", {"FG0": 2}, miss_rate=0.5, rules=KRULES)
    far = K.score_week("B", "T", "O", {"FG60": 2}, miss_rate=0.0, rules=KRULES)
    far_miss = K.score_week("B", "T", "O", {"FG60": 2}, miss_rate=0.5, rules=KRULES)
    assert (near.total - near_miss.total) == pytest.approx(
        (far.total - far_miss.total) - 2 * 0.5 * (6 - 3)
    )
