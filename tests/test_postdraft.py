"""Milestone 10 — post-draft analysis (§11 for a draft that already happened).

The tests worth reading first are the regressions, because each one pins a wrong
answer this module actually produced before it was fixed:

* ``test_grading_is_not_seduced_by_a_third_quarterback`` — M7's finding, arriving
  from a new direction. Scoring pairs by VOR graded round one as a wall of Ds for
  not drafting quarterbacks nobody could start.
* ``test_the_counterfactual_never_drafts_a_kicker_early`` — with §3.5's guard
  lifted, the best pair at nine consecutive picks came back "Texans D/ST +
  Brandon Aubrey".
* ``test_projected_finish_ranks_on_win_percentage`` — §2.5. Ranking on raw wins
  put a 13-game team above a 12-game team it trailed on every other measure, so
  the projected order contradicted the playoff odds printed beside it.
* ``test_every_team_gets_its_own_play_weeks`` — polars hands ``group_by`` a TUPLE
  key, so the schedule join silently matched nothing and gave all nine teams my
  bye weeks.
* ``test_equal_regret_picks_share_a_grade`` — ordinal ranking broke ties on row
  order and split identical picks between A+ and C.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ff_agent.config import ARTIFACTS_DIR, N_TEAMS, REGULAR_SEASON_WEEKS, ROSTER_TOTAL
from ff_agent.draft import pool as PL
from ff_agent.draft import surrogate as SUR
from ff_agent.postdraft import finish as FI
from ff_agent.postdraft import grade as GR
from ff_agent.postdraft import predict as PR
from ff_agent.postdraft import results as RS
from ff_agent.postdraft import teams as TM

pytestmark = pytest.mark.espn


@pytest.fixture(scope="module")
def pool():
    cached = PL.load_pool(ARTIFACTS_DIR)
    return cached if cached is not None else PL.build_pool(2026, N_TEAMS)


@pytest.fixture(scope="module")
def record():
    path = RS.latest_log(2026)
    if path is None:
        pytest.skip("no 2026 draft session log to analyse")
    return RS.load(2026, source="log", log_path=path)


@pytest.fixture(scope="module")
def weeks(record):
    w, _, _ = TM.play_weeks_by_manager(record, 2026)
    return w


@pytest.fixture(scope="module")
def rosters(record, weeks):
    return FI.project_remaining(record, weeks)[0]


@pytest.fixture(scope="module")
def graded(record, weeks, rosters):
    return GR.analyse(record, play_weeks=weeks, rosters=rosters)


# ── §0.1, which this milestone must not break either ─────────────────────────
FORBIDDEN = (
    ".post(", ".put(", ".patch(", ".delete(",
    "requests.post", "httpx.post", "urlopen",
    "add_to_roster", "submit_pick", "place_claim", "propose_trade",
)


def test_postdraft_package_cannot_write_to_espn():
    """§0.1 is absolute and applies to every package, not just the live one."""
    offenders = [
        f"{f}: {t}"
        for f in sorted(Path("ff_agent/postdraft").rglob("*.py"))
        for t in FORBIDDEN
        if t in f.read_text()
    ]
    assert not offenders, "§0.1 violated:\n" + "\n".join(offenders)


# ── reading the draft back ───────────────────────────────────────────────────
def test_undone_picks_are_replayed_away():
    recs = [
        {"kind": "pick", "overall": 1, "team": "A", "player": "x",
         "canonical_id": "1", "position": "RB"},
        {"kind": "pick", "overall": 2, "team": "B", "player": "y",
         "canonical_id": "2", "position": "WR"},
        # the human undid pick 2 and took somebody else
        {"kind": "pick", "overall": 2, "team": "B", "player": "z",
         "canonical_id": "3", "position": "TE"},
    ]
    hist = RS._replay(recs)
    assert [h["canonical_id"] for h in hist] == ["1", "3"]


def test_a_gap_in_the_log_is_a_blocking_error():
    """A missing pick makes every roster after it wrong, silently."""
    recs = [
        {"kind": "pick", "overall": 1, "team": "A", "player": "x",
         "canonical_id": "1", "position": "RB"},
        {"kind": "pick", "overall": 5, "team": "B", "player": "y",
         "canonical_id": "2", "position": "WR"},
    ]
    with pytest.raises(ValueError, match="missing"):
        RS._replay(recs)


def test_the_same_player_cannot_be_drafted_twice(record, pool):
    """§0.2: two fantasy teams cannot own the same person."""
    doubled = pl.concat([record.picks.head(1), record.picks]).with_columns(
        pl.int_range(1, record.picks.height + 2).alias("overall")
    )
    with pytest.raises(ValueError, match="more than once"):
        RS.attach(doubled.drop("pool_index", "nfl_team"), pool, 2026)


def test_every_pick_resolves_to_exactly_one_pool_row(record):
    """§0.2's hard gate, on the population that matters most."""
    assert record.picks["pool_index"].null_count() == 0
    assert record.picks["canonical_id"].n_unique() == record.picks.height


def test_off_board_picks_do_not_inherit_the_null_vor_sentinel(record):
    """``_to_pool`` fills a null VOR with -999, which would rank any player
    drafted from outside the board as the worst pick in the draft."""
    vor = record.pool.vor
    assert float(np.nanmin(vor[record.picks["pool_index"].to_numpy()])) > -900


# ── grading ──────────────────────────────────────────────────────────────────
def test_regret_is_never_negative(graded):
    """The manager cannot beat the best pair available to them."""
    assert (graded["regret_weekly_points"] >= 0).all()


def test_grades_are_a_within_round_curve(graded):
    """Every round should show a spread; the whole draft should show every letter."""
    per_round = graded.group_by("round").agg(pl.col("grade").n_unique().alias("n"))
    assert per_round["n"].max() > 1
    assert graded["grade"].n_unique() >= 4


def test_equal_regret_picks_share_a_grade(graded):
    """Ordinal ranking split identical picks between A+ and C on row order."""
    for (_, _), g in graded.group_by(["round", "regret_ratio"]):
        assert g["grade"].n_unique() == 1, g.select("name", "grade", "regret_ratio")


def test_grading_is_not_seduced_by_a_third_quarterback(pool, weeks):
    """M7, from the other side.

    VOR prices QB3 as though he would start, so a VOR-summed counterfactual
    graded round one as a wall of Ds for passing on quarterbacks. Lineup value
    must see that a third QB adds almost nothing to a roster that starts two.
    """
    qbs = np.flatnonzero(pool.pos_idx == PL.POS_INDEX["QB"])[:3]
    rbs = np.flatnonzero(pool.pos_idx == PL.POS_INDEX["RB"])[:7]
    wrs = np.flatnonzero(pool.pos_idx == PL.POS_INDEX["WR"])[:6]
    tes = np.flatnonzero(pool.pos_idx == PL.POS_INDEX["TE"])[:1]

    # a roster that already starts two quarterbacks, then ONE extra player —
    # the same roster size either way, so only the added player differs
    base = np.concatenate([qbs[:2], rbs[:6], wrs, tes])
    plus_qb = np.concatenate([base, [qbs[2]]])

    b0 = SUR.lineup_stats(pool, base[None, :]).mean[0]
    gain_qb = SUR.lineup_stats(pool, plus_qb[None, :]).mean[0] - b0
    vor_qb = float(pool.vor[qbs[2]])

    assert vor_qb > 100, (
        "the board really does price QB3 in triple digits — that is the trap M7 "
        "found, and if it stops being true this test stops testing anything"
    )
    assert gain_qb < 0.6 * vor_qb / SUR.GAMES, (
        f"QB3 realised {gain_qb:.2f} weekly points against the "
        f"{vor_qb / SUR.GAMES:.2f} his VOR implies. A grader scoring pairs by VOR "
        "would repeat M7's four-QB error."
    )
    # What he DOES realise is real and worth knowing: a third quarterback is not
    # worthless, he is worth the weeks the starters are on bye (§2.1). That is
    # the whole of M8's "§2.1's QB COUNT IS THREE" argument, visible in one line.
    assert gain_qb > 0.0


def test_the_counterfactual_never_drafts_a_kicker_early(graded):
    """§3.5: stream K and D/ST; never spend an early pick owning the best one."""
    early = graded.filter(
        (pl.col("round") < ROSTER_TOTAL - 2) & pl.col("better_pair").is_not_null()
    )
    bad = early.filter(
        pl.col("better_pair").str.contains("D/ST")
        | pl.col("better_pair").str.contains(r"(?i)kicker")
    )
    assert bad.height == 0, bad.select("overall", "round", "better_pair")


def test_a_pick_is_scored_against_the_managers_own_next_pick(graded, record):
    """The pair is (this pick, my next one) — not this pick and the next overall."""
    picks = record.picks.sort("overall").to_dicts()
    nxt = GR._next_pick_index(picks)
    for i, j in enumerate(nxt):
        if j is not None:
            assert picks[j]["manager"] == picks[i]["manager"]
            assert picks[j]["overall"] > picks[i]["overall"]


# ── finishing an unfinished draft ────────────────────────────────────────────
def test_projected_picks_never_duplicate_a_player(record, weeks):
    rosters, _ = FI.project_remaining(record, weeks)
    flat = [i for r in rosters.values() for i in r]
    assert len(flat) == len(set(flat)), "two managers were projected the same player"


def test_projected_rosters_reach_full_size_and_can_field_a_lineup(record, weeks):
    rosters, projected = FI.project_remaining(record, weeks)
    if record.complete:
        assert set(projected.values()) == {0}
        return
    for m, idx in rosters.items():
        assert len(idx) == record.rounds, f"{m} has {len(idx)} of {record.rounds}"
        counts = {p: 0 for p in PL.POSITIONS}
        for i in idx:
            counts[PL.POSITIONS[record.pool.pos_idx[i]]] += 1
        assert counts["K"] >= 1 and counts["DST"] >= 1, (m, counts)
        assert counts["QB"] >= 2, f"{m} cannot start two quarterbacks: {counts}"


def test_padding_is_a_no_op_on_a_complete_draft(record, weeks):
    if not record.complete:
        pytest.skip("draft still in progress")
    rosters, projected = FI.project_remaining(record, weeks)
    assert set(projected.values()) == {0}
    for m in record.managers:
        assert rosters[m] == list(record.roster_indices(m))


# ── team ratings ─────────────────────────────────────────────────────────────
def test_every_team_gets_its_own_play_weeks(record, weeks):
    """polars' group_by key is a TUPLE; keying on the bare name matched nothing
    and quietly handed all nine teams MY bye weeks."""
    spans = {m: sorted(set(REGULAR_SEASON_WEEKS) - set(w)) for m, w in weeks.items()}
    assert len({tuple(v) for v in spans.values()}) > 1, (
        "every team was given identical bye weeks — the schedule join failed"
    )
    assert sum(len(v) for v in spans.values()) == len(REGULAR_SEASON_WEEKS), (
        "an odd team count must produce exactly one bye per week"
    )


def test_roster_shape_reports_real_picks_not_projected_ones(record, weeks):
    rosters, projected = FI.project_remaining(record, weeks)
    r, _ = TM.rate(record, 2026, rosters=rosters, projected=projected)
    for row in r.to_dicts():
        assert row["picks"] == len(record.roster_indices(row["manager"]))


def test_team_strength_is_a_lineup_not_a_roster_sum(record, weeks):
    """A 17-man roster starting 10 must score less than the sum of its parts."""
    rosters, projected = FI.project_remaining(record, weeks)
    r, _ = TM.rate(record, 2026, rosters=rosters, projected=projected)
    for row in r.to_dicts():
        idx = rosters[row["manager"]]
        total = float(np.nansum(record.pool.points[idx])) / SUR.GAMES
        assert row["weekly_points"] < total


# ── predictions ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def predicted(record, weeks, rosters, graded):
    proj = {m: len(rosters[m]) - len(record.roster_indices(m)) for m in rosters}
    r, _ = TM.rate(record, 2026, rosters=rosters, projected=proj)
    r = TM.draft_efficiency(r, graded)
    return PR.predict(r, season=2026, n_sims=4000)


def test_projected_finish_ranks_on_win_percentage(predicted):
    """§2.5, confirmed: the league seeds on PCT, and five teams play 12 games
    against four playing 13. Ranking on raw wins contradicted the odds beside it."""
    table, _ = predicted
    order = table.sort("projected_finish")
    pct = order["mean_win_pct"].to_list()
    assert pct == sorted(pct, reverse=True)
    assert set(order["games"].to_list()) == {12, 13}


def test_playoff_odds_agree_with_the_projected_order(predicted):
    table, _ = predicted
    order = table.sort("projected_finish")
    assert order["p_playoffs"][0] == pytest.approx(
        order["p_playoffs"].max(), abs=0.02
    )


def test_the_bracket_is_conserved(predicted):
    from ff_agent.config import FIRST_ROUND_BYES, PLAYOFF_TEAMS

    table, _ = predicted
    assert float(table["p_title"].sum()) == pytest.approx(1.0, abs=0.01)
    assert float(table["p_playoffs"].sum()) == pytest.approx(PLAYOFF_TEAMS, abs=0.05)
    assert float(table["p_top2_seed"].sum()) == pytest.approx(
        FIRST_ROUND_BYES, abs=0.05
    )


def test_the_forecast_ships_its_own_measured_ceiling(predicted):
    """M6 scored this exact task at -0.21 with a +0.52 ceiling. A standings
    projection that does not say so is being read as more than it is."""
    _, meta = predicted
    cal = meta["calibration"]
    assert cal["preseason_projections_spearman"] == -0.21
    assert cal["perfect_simulator_ceiling_spearman"] == 0.52
    assert "projections" in cal["binding_constraint"]


def test_the_two_sources_agree_on_the_same_draft():
    """The session log and ESPN's export are independent readings of one draft.

    They disagree about spelling in both directions — the log records MANAGERS,
    the export records TEAM NAMES, and on draft night one of those names changed
    mid-draft — so agreement here is a real cross-check on the crosswalk, the
    snake-order reconstruction and the team_id join, not a tautology.
    """
    from ff_agent.postdraft import build as PB

    log = RS.latest_log(2026)
    if log is None:
        pytest.skip("no 2026 session log")
    a = PB.build(2026, source="log", log_path=log, predict_season=False)
    if not a.record.complete:
        pytest.skip("draft still in progress; ESPN will not match a partial log")
    try:
        b = PB.build(2026, source="espn", predict_season=False)
    except Exception as exc:                       # offline or uncached
        pytest.skip(f"ESPN draft export unavailable: {exc}")

    assert a.record.n_picks == b.record.n_picks
    assert (
        a.picks.sort("overall")["canonical_id"].to_list()
        == b.picks.sort("overall")["canonical_id"].to_list()
    ), "the log and ESPN disagree about who was drafted, or in what order"

    ra = a.ratings.sort("team").select("team", "weekly_points")
    rb = b.ratings.sort("team").select("team", "weekly_points")
    assert ra.to_dicts() == rb.to_dicts(), (
        "the same rosters scored differently depending on which source read them"
    )


def test_all_four_double_up_managers_are_matched(predicted, record, weeks, rosters, graded):
    """§1 spells one manager "R. Sharrett" and ESPN says "Rayne Sharrett"."""
    table, _ = predicted
    proj = {m: 0 for m in rosters}
    r, _ = TM.rate(record, 2026, rosters=rosters, projected=proj)
    view = PR.my_schedule_view(table, r)
    assert not view["unmatched_double_up_managers"], view
    assert len(view["faced_twice"]) == 4
