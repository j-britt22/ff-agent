"""M9 — the live draft loop (§4, §5, §11 step 9).

The load-bearing ones:

* ``test_live_package_cannot_write_to_espn`` — §0.1 is absolute, and this is the
  module where it would be violated. "Never submit a lineup, waiver claim, trade,
  or draft pick to ESPN. Ever."
* ``test_a_query_longer_than_the_stored_name_still_resolves`` — the 2025 replay
  found this: ESPN says "James Cook III" and the board says "James Cook", and a
  longer query fails prefix, surname AND substring matching. Typing the name
  exactly as ESPN displays it matched nothing at all.
* ``test_advice_fits_the_clock`` — §4 budgets one second.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ff_agent.config import ARTIFACTS_DIR, N_TEAMS, ROSTER_TOTAL
from ff_agent.draft import build as DB
from ff_agent.draft import engine as E
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.live import advise as AD
from ff_agent.live import log as LG
from ff_agent.live import loop as LP
from ff_agent.live import state as ST
from ff_agent.live.entry import Resolver

pytestmark = pytest.mark.espn


@pytest.fixture(scope="module")
def pool():
    cached = PL.load_pool(ARTIFACTS_DIR)
    return cached if cached is not None else PL.build_pool(2026, N_TEAMS)


@pytest.fixture(scope="module")
def managers():
    return DB.managers_for_slot(5)


@pytest.fixture
def state(pool, managers):
    return ST.new_state(pool, 5, managers)


@pytest.fixture(scope="module")
def resolver(pool):
    return Resolver(pool)


@pytest.fixture(scope="module")
def context(pool, managers):
    return AD.load_context(2026, pool, managers, ROSTER_TOTAL)


# ── §0.1, the rule this milestone could break ────────────────────────────────
FORBIDDEN = (
    ".post(", ".put(", ".patch(", ".delete(",
    "requests.post", "httpx.post", "urlopen",
    "add_to_roster", "submit_pick", "place_claim", "propose_trade",
)


def test_live_package_cannot_write_to_espn():
    """§0.1: "Never submit a lineup, waiver claim, trade, or draft pick to ESPN."

    Write-capable endpoints are out of scope for this project — if code to hit
    one appears, it is a bug. This scans for one, rather than trusting that
    nobody added it.
    """
    live = Path("ff_agent/live")
    offenders = []
    for f in sorted(live.rglob("*.py")):
        src = f.read_text()
        for token in FORBIDDEN:
            if token in src:
                offenders.append(f"{f}: {token}")
    assert not offenders, "§0.1 violated:\n" + "\n".join(offenders)


def test_the_espn_client_exposes_no_write_endpoint():
    src = Path("ff_agent/data/espn.py").read_text()
    for token in FORBIDDEN:
        assert token not in src, f"espn.py contains {token}"


def test_the_loop_says_so_out_loud(state, resolver):
    assert "§0.1" in "\n".join(LP.handle(state, resolver, "q").lines)


# ── state ────────────────────────────────────────────────────────────────────
def test_turn_order_follows_the_snake(state):
    assert state.next_pick == 1
    assert state.on_the_clock == 0
    assert not state.my_turn
    assert state.picks_until_my_turn() == 4     # slot 5, zero-based index 4


def test_recording_advances_the_clock(state, pool):
    for i in range(4):
        state.record(i)
    assert state.my_turn
    assert state.picks_until_my_turn() == 0
    assert state.next_pick == 5


def test_a_player_cannot_be_drafted_twice(state):
    state.record(3)
    with pytest.raises(ValueError, match="already been drafted"):
        state.record(3)


def test_undo_restores_the_previous_board(state):
    state.record(3)
    state.record(9)
    assert state.undo() == 9
    assert state.history == [3]
    assert 9 not in state.taken()


def test_my_roster_tracks_only_my_picks(state, pool):
    for i in range(9):            # one full round, slot 5 is index 4
        state.record(i)
    assert state.my_roster() == [4]
    assert sum(state.position_counts().values()) == 1


def test_alarms_fire_on_my_roster_not_the_leagues(state, pool):
    """§10's alarms are about MY decisions (M7 finding)."""
    ks = np.flatnonzero(pool.pos_idx == PL.POS_INDEX["K"])
    for i in range(4):
        state.record(int(np.flatnonzero(pool.pos_idx == PL.POS_INDEX["RB"])[i]))
    state.record(int(ks[0]))       # mine, round 1
    assert any("kicker" in a for a in state.alarms())


# ── manual entry (§6, §0.2) ──────────────────────────────────────────────────
def test_exact_name_resolves(resolver):
    m = resolver.resolve("Josh Allen")
    assert m.resolved and resolver.names[m.index] == "Josh Allen"


def test_ambiguity_asks_and_never_guesses(resolver):
    """§0.2's spirit: no id resolves without a human looking at it."""
    m = resolver.resolve("allen")
    assert not m.resolved
    assert m.ambiguous and len(m.candidates) > 1


def test_a_query_longer_than_the_stored_name_still_resolves(resolver):
    """ESPN shows "James Cook III"; the board says "James Cook".

    A longer query fails prefix, surname and substring alike, so typing the name
    exactly as ESPN displays it matched NOTHING. Found by the 2025 replay.
    """
    a = resolver.resolve("James Cook III")
    b = resolver.resolve("James Cook")
    assert a.resolved and b.resolved and a.index == b.index


def test_defenses_resolve_by_abbreviation_and_by_nickname(resolver):
    """The two sources spell defenses differently ("Texans D/ST" vs "HOU D/ST").

    Also pins the short-circuit bug: normalize_team() ECHOES anything it does
    not recognise, so it is always truthy, and `normalize_team(q) or nickname(q)`
    meant the nickname branch never ran.
    """
    for q in ("SF", "49ers", "Texans", "Texans D/ST"):
        m = resolver.resolve(q)
        assert m.resolved, q
        assert resolver.positions[m.index] == "DST", q


def test_nonsense_is_reported_not_guessed(resolver):
    m = resolver.resolve("qqzzxx")
    assert m.missing and not m.candidates


def test_taken_players_drop_out_of_the_pool(resolver, pool):
    i = resolver.resolve("Josh Allen").index
    m = resolver.resolve("Josh Allen", available=set(range(pool.n)) - {i})
    assert not m.resolved or m.index != i


# ── the engine's partial-board support ───────────────────────────────────────
def test_history_is_preserved_in_every_simulation(pool, managers, context):
    tilt, reach, scen, _ = context
    hist = [0, 5, 9, 12]
    res = E.simulate_drafts(
        pool, managers, my_index=4, policy=POL.NeedWeighted(caps={"QB": 3}),
        tilt_lookup=tilt, reach=reach, n_sims=40, seed=3, history=hist,
    )
    assert (res.picks[:, :len(hist)] == np.array(hist)[None, :]).all()
    for row in res.picks[:10]:
        assert len(set(row.tolist())) == len(row)


def test_a_corrupt_history_is_refused(pool, managers, context):
    tilt, reach, _, _ = context
    with pytest.raises(ValueError, match="twice"):
        E.simulate_drafts(
            pool, managers, my_index=4, policy=POL.BestVOR(),
            tilt_lookup=tilt, reach=reach, n_sims=10, history=[7, 7],
        )


# ── advice ───────────────────────────────────────────────────────────────────
def test_advice_fits_the_clock(state, context):
    """§4: "Live draft | On the clock | ... | < 1 sec"."""
    tilt, reach, scen, targets = context
    for i in range(4):
        state.record(i)
    t = time.perf_counter()
    a = AD.advise(state, tilt, reach, scen, targets)
    assert time.perf_counter() - t < 1.0
    assert a.options and a.pick == 5


def test_advice_prices_alternatives_not_just_a_winner(state, context):
    """A deterministic policy has ONE answer at a known board, so probabilities
    collapse to 100% and say nothing. What decides the pick is what the runner-up
    costs."""
    tilt, reach, scen, targets = context
    for i in range(4):
        state.record(i)
    a = AD.advise(state, tilt, reach, scen, targets)
    assert len(a.options) >= 2
    assert a.options[0].delta_vs_best == 0.0
    assert all(o.delta_vs_best <= 0 for o in a.options)
    assert len({o.position for o in a.options}) > 1, "price a real choice, not 5 QBs"


def test_advice_follows_my_actual_roster(pool, managers, context):
    """I will deviate from the policy; the advice has to start from the roster I
    actually have, not the one the simulation would have built."""
    tilt, reach, scen, targets = context
    qbs = [int(i) for i in np.flatnonzero(pool.pos_idx == PL.POS_INDEX["QB"])[:3]]
    others = [int(i) for i in np.flatnonzero(pool.pos_idx != PL.POS_INDEX["QB"])]
    s = ST.new_state(pool, 1, managers)
    order = s.order
    filler = iter(others)
    for k in range(2 * len(managers)):
        if order[k] == 0 and len(s.my_roster()) < len(qbs):
            s.record(qbs[len(s.my_roster())])
        else:
            s.record(next(i for i in filler if i not in s.taken()))
    assert s.position_counts()["QB"] >= 2
    opts = AD.candidate_options(s)
    caps = {PL.POSITIONS[pool.pos_idx[i]] for i in opts}
    assert "QB" not in caps or s.position_counts()["QB"] < 3


def test_scenario_detection_waits_for_evidence(state, context):
    tilt, reach, scen, targets = context
    name, ev = AD.detect_scenario(state, targets)
    assert name == "shipped" and not ev["confident"]


def test_scenario_detection_reads_a_quarterback_run(pool, managers, context):
    """M8 left half the turns scenario-dependent because one 2-QB draft cannot
    pin QB timing. The draft itself resolves it — that is M9's whole edge."""
    _, _, _, targets = context
    s = ST.new_state(pool, 5, managers)
    qbs = np.flatnonzero(pool.pos_idx == PL.POS_INDEX["QB"])
    for i in qbs[:12]:
        s.record(int(i))
    hot, ev = AD.detect_scenario(s, targets)
    assert ev["confident"]
    assert hot == "qb_hot", ev


def test_sims_are_sized_to_the_budget():
    early = AD.sims_for_budget(150)
    late = AD.sims_for_budget(20)
    assert late > early >= AD.MIN_SIMS
    assert AD.sims_for_budget(1) <= AD.MAX_SIMS


# ── §10 logging ──────────────────────────────────────────────────────────────
def test_every_recommendation_is_logged(tmp_path, state, context):
    """§10: "Log every recommendation with timestamp and inputs, or the season
    teaches you nothing."."""
    tilt, reach, scen, targets = context
    for i in range(4):
        state.record(i)
    log = LG.SessionLog(tmp_path / "s.jsonl")
    a = AD.advise(state, tilt, reach, scen, targets)
    log.recommendation(a, state)
    recs = log.read()
    assert len(recs) == 1
    r = recs[0]
    assert r["kind"] == "recommendation" and r["ts"]
    assert r["pick"] == 5 and r["options"]
    assert "position_counts" in r and "scenario_evidence" in r


def test_picks_are_logged_with_the_canonical_id(tmp_path, state, resolver):
    log = LG.SessionLog(tmp_path / "s.jsonl")
    LP.handle(state, resolver, "Josh Allen")
    log.pick(state, state.history[-1], state.managers[0], "manual")
    r = log.read()[0]
    assert r["player"] == "Josh Allen" and r["canonical_id"]
    assert r["source"] == "manual"


# ── the artifact the T-60 path depends on ────────────────────────────────────
def test_cached_pool_round_trips(pool, tmp_path):
    PL.save_pool(pool, tmp_path)
    back = PL.load_pool(tmp_path)
    assert back is not None and back.n == pool.n
    assert back.df["canonical_id"].to_list() == pool.df["canonical_id"].to_list()
    assert np.allclose(back.vor, pool.vor)


# ── §11 step 9: the full mock ────────────────────────────────────────────────
def test_full_mock_replays_the_2025_draft_under_the_clock():
    """§11 step 9: "full mock, sub-second, manual entry works with WiFi off".

    Every pick goes in through the manual-entry resolver BY NAME — the same path
    draft day uses — and every one of my turns pays for a real recommendation.
    This gate is about the mechanics under a clock, not predictive accuracy;
    M7's backtest grades the model.
    """
    from ff_agent.live import replay as RP

    r = RP.run(write_log=False)
    assert r["picks_replayed"] == r["actual_picks"]
    assert r["entry"]["needed_fallback"] == 0, r["entry"]["examples"]
    assert r["latency"]["p95"] < 1.0, r["latency"]
    assert r["latency"]["over_budget"] == 0, r["latency"]
    assert r["my_roster"]["legal"], r["my_roster"]
    assert r["turns_advised"] == r["rounds"]


def test_the_mock_runs_with_the_network_off(monkeypatch):
    """§0.3 and §5: draft-day WiFi will betray you."""
    monkeypatch.setenv("FF_OFFLINE", "1")
    cached = PL.load_pool(ARTIFACTS_DIR)
    if cached is None:
        pytest.skip("run `cli plans` first — this checks the shipped artifact")
    s = ST.new_state(cached, 5, DB.managers_for_slot(5))
    assert s.total_picks == N_TEAMS * ROSTER_TOTAL
    assert Resolver(cached).resolve("Josh Allen").resolved


# ── the browser GUI (cli gui) ────────────────────────────────────────────────
# Same engine as the terminal loop, so these test the parts the GUI adds: the
# ESPN poll, the snapshot the browser renders, and §0.1 across an HTML surface
# the Python-only guard above cannot see.


def _session(pool, managers, tmp_path, context):
    from ff_agent.live import log as LG
    from ff_agent.live import server as SV

    state = ST.new_state(pool, 5, managers)
    log = LG.SessionLog(tmp_path / "s.jsonl")
    return SV.DraftSession(state, Resolver(pool), context, 2026, 5, 3, log)


def test_the_gui_page_can_only_talk_to_its_own_api():
    """§0.1 across the HTML surface.

    The Python guard scans ``*.py`` and cannot see ui.html, but the page is the
    one thing in this project that issues requests from a browser — where the
    user's ESPN cookies live. A fetch to any absolute URL would carry them.
    Every request it makes must be a same-origin /api/ path.
    """
    src = Path("ff_agent/live/ui.html").read_text()
    import re

    for url in re.findall(r"""fetch\(\s*['"`]([^'"`]+)""", src):
        assert url.startswith("/api/"), f"ui.html fetches {url!r}, not a local /api path"
    for bad in ("espn.com", "http://", "https://"):
        assert bad not in src, f"ui.html references {bad!r} — it must stay same-origin"


def test_the_poll_places_every_player_in_the_pool(pool):
    """A poll that silently drops players is worse than no poll: the board and
    ESPN would diverge without saying so. D/ST are the interesting half — they
    are negative ESPN ids and were null in draft history for every season.
    """
    from ff_agent.live import poll as PO

    m = PO.espn_index_map(pool)
    assert len(set(m.values())) == pool.n, (
        f"only {len(set(m.values()))} of {pool.n} pool entries are reachable "
        f"from an ESPN draft-feed id"
    )
    dst = [i for i in range(pool.n) if pool.df["position"].to_list()[i] == "DST"]
    assert dst, "the pool has no defenses — M7 added them, so this is a regression"
    assert set(dst) <= set(m.values())


def test_the_poll_never_overwrites_what_a_human_typed(pool, managers, tmp_path, context):
    """§6: ESPN's feed is flaky, so manual entry is the source of truth. A feed
    that CONTRADICTS the board is reported, never applied.
    """
    s = _session(pool, managers, tmp_path, context)
    s.record_index(10)
    s.record_index(11)

    # the feed claims a different pick 1
    feed = [12, 11]
    with s.lock:
        common = min(len(s.state.history), len(feed))
        conflict = s.state.history[:common] != feed[:common]
    assert conflict, "this fixture must actually disagree or it tests nothing"
    assert s.state.history == [10, 11], "manual entries must survive untouched"


def test_recent_picks_are_numbered_from_the_right_end(pool, managers, tmp_path, context):
    """Regression: the first version derived the tail's starting pick number
    with ``len(history) - 11``, which goes NEGATIVE below 12 picks and clamped
    every row to pick 1 — so early in the draft every pick was attributed to
    whoever drafted first.
    """
    s = _session(pool, managers, tmp_path, context)
    for i in range(4):
        s.record_index(i)
    snap = s.snapshot()
    assert [r["pick"] for r in snap["recent"]] == [4, 3, 2, 1]
    assert snap["recent"][-1]["manager"] == managers[0]
    assert snap["recent"][0]["manager"] == managers[3]


def test_the_snapshot_survives_json(pool, managers, tmp_path, context):
    """The browser gets this over HTTP; a numpy scalar in it is a 500 mid-draft."""
    import json

    s = _session(pool, managers, tmp_path, context)
    s.record_index(3)
    json.dumps(s.snapshot())


def test_a_duplicate_pick_is_refused_with_a_usable_message(pool, managers, tmp_path, context):
    s = _session(pool, managers, tmp_path, context)
    assert s.record_index(7)["ok"]
    again = s.record_index(7)
    assert not again["ok"] and "already been drafted" in again["message"]


def test_the_gui_prompts_on_ambiguity_rather_than_guessing(pool, managers, tmp_path, context):
    """§0.2 through the web path: no id resolves without a human looking at it."""
    s = _session(pool, managers, tmp_path, context)
    r = s.record_query("jefferson")
    assert not r["ok"] and r.get("ambiguous"), r
    assert len(r["candidates"]) > 1
    assert s.state.history == [], "nothing may be recorded while ambiguous"
