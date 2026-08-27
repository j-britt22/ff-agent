"""The layer that was missing: live state -> engine -> a real digest.

Every engine was built and tested against synthetic frames, and nothing
assembled real league state, ran an engine over it and rendered the result — so
`cli monitor --job waivers` checked that ESPN was reachable and printed a
placeholder. These tests exist so that cannot silently be true again.

ESPN is mocked at the fetcher boundary rather than at the HTTP one: the point is
to prove the WIRING, and a fake `espn_api.League` would only test my idea of
what espn_api returns.
"""
import datetime as dt

import polars as pl
import pytest

from ff_agent.inseason import builders as B
from ff_agent.inseason import clock as CK
from ff_agent.inseason import lineup as LN
from ff_agent.inseason import ros as ROS
from ff_agent.inseason import state as ST

ET = CK.ET
WEEK = 9
FILLER_TEAMS = (
    "Clearing the Fields", "Gibbs Me My Money", "A Chane Reaction",
    "Unsolicited Dak Pics", "Nothing Beats a JJett 2 Holiday", "leah's team",
)
TEAMS = ("First Down Syndrome", "Hodor's Hodors", "Personality Hires") + FILLER_TEAMS
"""Nine teams, because the league has nine and the bracket seats six.

An earlier version of this fixture had three, and the simulator raised an
IndexError from inside numpy while indexing seed 6 of a 3-team league. That is
now a clear refusal in simulate() — but the fixture also has to be the real
shape, or every probability it produces is answering a different league's
question.
"""

NFL_TEAMS = (
    "BUF", "BAL", "SF", "KC", "DEN", "MIA", "GB", "NYJ", "LV", "DAL", "SEA",
    "CIN", "PHI", "MIN", "DET", "TB", "CAR", "CLE", "PIT", "ATL", "NO", "CHI",
    "HOU", "TEN", "IND", "JAX", "LAC", "ARI", "WAS", "NYG", "LAR", "NE",
)

ROSTER_SPEC = {
    "First Down Syndrome": [
        ("q1", "My QB1", "QB", "BUF"), ("q2", "My QB2", "QB", "BAL"),
        ("r1", "My RB1", "RB", "SF"), ("r2", "My RB2", "RB", "KC"),
        ("r3", "My RB3", "RB", "DEN"), ("r4", "My RB4", "RB", "MIA"),
        ("w1", "My WR1", "WR", "GB"), ("w2", "My WR2", "WR", "NYJ"),
        ("t1", "My TE1", "TE", "LV"), ("k1", "My K", "K", "DAL"),
        ("d1", "My DST", "DST", "SEA"),
    ],
    "Hodor's Hodors": [
        ("hq1", "H QB1", "QB", "CIN"), ("hq2", "H QB2", "QB", "PHI"),
        ("hw1", "H WR1", "WR", "MIN"), ("hw2", "H WR2", "WR", "DET"),
        ("hw3", "H WR3", "WR", "TB"), ("hw4", "H WR4", "WR", "CAR"),
        ("hr1", "H RB1", "RB", "CLE"), ("hr2", "H RB2", "RB", "PIT"),
        ("ht1", "H TE1", "TE", "ATL"), ("hk1", "H K", "K", "NO"),
        ("hd1", "H DST", "DST", "CHI"),
    ],
    "Personality Hires": [
        ("pq1", "P QB1", "QB", "HOU"), ("pq2", "P QB2", "QB", "TEN"),
        ("pr1", "P RB1", "RB", "IND"), ("pr2", "P RB2", "RB", "JAX"),
        ("pw1", "P WR1", "WR", "LAC"), ("pw2", "P WR2", "WR", "ARI"),
        ("pt1", "P TE1", "TE", "WAS"), ("pk1", "P K", "K", "NYG"),
        ("pd1", "P DST", "DST", "LAR"),
    ],
}
SHAPE = ("QB", "QB", "RB", "RB", "WR", "WR", "TE", "K", "DST")
for _i, _team in enumerate(FILLER_TEAMS):
    ROSTER_SPEC[_team] = [
        (f"f{_i}{_j}", f"F{_i} {_pos}{_j}", _pos, NFL_TEAMS[(_i * 4 + _j) % 32])
        for _j, _pos in enumerate(SHAPE)
    ]

FREE_AGENTS = [
    ("fa1", "Waiver WR", "WR", "IND"),      # good — upgrades my WR2
    ("fa2", "Bye-14 RB", "RB", "ARI"),      # §2.1 free bye
    ("fa3", "Deep Bench", "WR", "TEN"),     # never starts
]
PTS = {
    "q1": 22.0, "q2": 20.0, "r1": 16.0, "r2": 14.0, "r3": 10.0, "r4": 7.0,
    "w1": 15.0, "w2": 8.0, "t1": 9.0, "k1": 8.0, "d1": 7.0,
    "hq1": 21.0, "hq2": 19.0, "hw1": 17.0, "hw2": 15.0, "hw3": 13.0,
    "hw4": 12.0, "hr1": 9.0, "hr2": 6.0, "ht1": 8.0, "hk1": 8.0, "hd1": 7.0,
    "pq1": 18.0, "pq2": 17.0, "pr1": 13.0, "pr2": 11.0, "pw1": 14.0,
    "pw2": 12.0, "pt1": 8.0, "pk1": 8.0, "pd1": 7.0,
    "fa1": 13.0, "fa2": 11.0, "fa3": 2.0,
}
for _team in FILLER_TEAMS:
    for _cid, _n, _pos, _nfl in ROSTER_SPEC[_team]:
        PTS[_cid] = {"QB": 18.0, "RB": 12.0, "WR": 12.0, "TE": 8.0,
                     "K": 8.0, "DST": 7.0}[_pos]
BYES = {"ARI": 14, "SEA": 5}

MANAGERS = {
    # §2.3's four DOUBLE-UP opponents, spelled as config.py spells them. Using
    # placeholder names here meant the double-up penalty could never fire, and
    # the test asserting it silently passed on a league where nobody was played
    # twice.
    "Hodor's Hodors": "Camden Sims",
    "Personality Hires": "Kylie Leahy",
    "Clearing the Fields": "R. Sharrett",
    "Gibbs Me My Money": "Matthew Benca",
    "First Down Syndrome": "Jordan Britt",
}


@pytest.fixture
def espn(monkeypatch):
    """Mock every fetcher state.load reads, and the resolution step."""
    from ff_agent.data import byes as BY
    from ff_agent.data import espn as E
    from ff_agent.season import schedule as SCH

    roster_rows, tid = [], {}
    for i, (team, players) in enumerate(ROSTER_SPEC.items(), start=1):
        tid[team] = i
        for cid, name, pos, nfl in players:
            roster_rows.append({
                "team_id": i, "fantasy_team": team,
                "manager": MANAGERS.get(team, f"M{i}"),
                "espn_id": cid, "name": name, "position": pos, "team": nfl,
                "lineup_slot": None, "injury_status": None})
    rosters = pl.DataFrame(roster_rows)

    proj_rows = []
    for cid, name, pos, nfl in (
        [p for ps in ROSTER_SPEC.values() for p in ps] + list(FREE_AGENTS)
    ):
        for wk in range(1, 15):
            bye = BYES.get(nfl)
            proj_rows.append({
                "espn_id": cid, "name": name, "position": pos, "team": nfl,
                "week": wk, "projected_points": 0.0 if wk == bye else PTS[cid],
                "actual_points": None, "injury_status": None, "lineup_slot": None,
                "on_team_id": None, "percent_owned": 50.0, "source": "x",
                # The SEASON projection is the rest-of-season anchor and the
                # per-week numbers serve the weekly call. The fixture carries
                # both, or every wiring test here would exercise the fallback
                # branch instead of the shipped one. 17 games, so the implied
                # rate is PTS[cid] and every expectation below is unchanged.
                "season_projected_points": PTS[cid] * 17,
                # Consistent with WEEK: this player has already banked the games
                # he has played. An earlier version left this at 0.0 while the
                # fixture sat at week 9, which said "projected for 17 games,
                # scored nothing in 8" — and the anchor correctly answered that
                # the remaining 187 points were all still to come, at nearly
                # double the rate. A fixture that is not internally consistent
                # measures the fixture.
                "season_actual_points":
                    PTS[cid] * ((WEEK - 1) - (1 if bye and bye < WEEK else 0)),
                "season_projected_avg": PTS[cid]})
    projections = pl.DataFrame(proj_rows)

    completed = pl.DataFrame([
        {"week": w, "team": t, "points": 120.0}
        for w in range(1, WEEK) for t in TEAMS])
    waivers = pl.DataFrame({
        "team_id": [tid[t] for t in TEAMS],
        "fantasy_team": list(TEAMS),
        # I sit fifth of nine — §9.3's "nine teams makes priority cheap" is only
        # interesting from somewhere in the middle of the queue.
        "waiver_rank": [5, 1, 2, 3, 4, 6, 7, 8, 9]})

    sched_rows = []
    for wk in range(1, 15):
        for t in TEAMS:
            opp = next(o for o in TEAMS if o != t) if wk % 2 else None
            sched_rows.append({"season": 2026, "week": wk, "team": t,
                               "opponent": opp, "team_id": tid[t]})
    schedule = pl.DataFrame(sched_rows)

    byes_df = pl.DataFrame({
        "team": sorted({p[3] for ps in ROSTER_SPEC.values() for p in ps}
                       | {f[3] for f in FREE_AGENTS}),
    }).with_columns(
        pl.col("team").replace_strict(BYES, default=None).cast(pl.Int64).alias("bye_week"))

    monkeypatch.setattr(E, "current_rosters", lambda season=2026, **kw: rosters)
    monkeypatch.setattr(E, "player_projections", lambda season=2026, **kw: projections)
    monkeypatch.setattr(E, "weekly_results", lambda season=2026, **kw: completed)
    monkeypatch.setattr(E, "waiver_order", lambda season=2026, **kw: waivers)
    monkeypatch.setattr(BY, "bye_weeks", lambda season, **kw: byes_df)
    monkeypatch.setattr(SCH, "league_schedule", lambda season=2026, **kw: schedule)
    monkeypatch.setattr(
        ST, "resolve",
        lambda df, label, canonical=None: (
            df.with_columns(pl.col("espn_id").alias("canonical_id"),
                            pl.lit("mock").alias("match_method")),
            df.head(0).with_columns(pl.lit(None, dtype=pl.Utf8).alias("canonical_id"),
                                    pl.lit(None, dtype=pl.Utf8).alias("match_method")),
        ))
    return {"rosters": rosters, "projections": projections}


@pytest.fixture
def loaded(espn):
    st = ST.load(2026, week=WEEK, my_team="First Down Syndrome")
    proj = ST._normalize(espn["projections"]).with_columns(
        pl.col("espn_id").alias("canonical_id"))
    byes = pl.DataFrame({"team": list(BYES), "bye_week": list(BYES.values())})
    ros = ROS.from_espn(proj, from_week=WEEK, byes=byes, season=2026)
    return st, ros.frame


# ─── state.load ─────────────────────────────────────────────────────────────
def test_load_builds_a_real_league_state(loaded):
    st, _ = loaded
    assert st.week == WEEK
    assert set(st.managers) == set(TEAMS)
    assert st.my_roster.height == len(ROSTER_SPEC["First Down Syndrome"])
    assert st.free_agents.height == len(FREE_AGENTS)
    assert st.completed.height == (WEEK - 1) * len(TEAMS)


def test_the_free_agent_pool_is_everyone_not_on_a_roster(loaded):
    st, _ = loaded
    assert set(st.free_agents["canonical_id"].to_list()) == {f[0] for f in FREE_AGENTS}


def test_byes_reach_the_roster_so_2_1_can_be_priced(loaded):
    st, _ = loaded
    fa = st.free_agents.filter(pl.col("canonical_id") == "fa2")
    assert fa["bye_week"][0] == 14        # one of MY free weeks


def test_an_unresolvable_ROSTERED_player_is_fatal(espn, monkeypatch):
    """A rostered player we cannot price silently distorts MY starting lineup —
    unlike a free agent, he cannot simply be skipped."""
    monkeypatch.setattr(
        ST, "resolve",
        lambda df, label, canonical=None: (
            df.head(0).with_columns(pl.lit(None, dtype=pl.Utf8).alias("canonical_id")),
            df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("canonical_id")),
        ))
    with pytest.raises(ST.StateError, match="ROSTERED"):
        ST.load(2026, week=WEEK)


def test_an_unresolvable_FREE_AGENT_is_reported_not_fatal(espn, monkeypatch):
    """§0.2's in-season shape: refusing to send a Tuesday claim list because a
    practice-squad tight end was promoted is the tool failing at its job."""
    real = ST.resolve

    def selective(df, label, canonical=None):
        ok, bad = real(df, label, canonical)
        if label == "free agents":
            return ok.head(0), ok
        return ok, bad

    monkeypatch.setattr(ST, "resolve", selective)
    st = ST.load(2026, week=WEEK)
    assert st.free_agents.height == 0
    assert any("could not be resolved" in n for n in st.notes)


def test_a_stale_results_gap_blocks_the_digest(espn, monkeypatch):
    """§10: an optimizer silently running week-3 data in week 8 is worse than
    no optimizer."""
    from ff_agent.data import espn as E
    monkeypatch.setattr(E, "weekly_results", lambda season=2026, **kw: pl.DataFrame(
        {"week": [1], "team": ["First Down Syndrome"], "points": [100.0]}))
    with pytest.raises(ST.StateError, match="have been played"):
        ST.load(2026, week=WEEK)


# ─── ros.from_espn ──────────────────────────────────────────────────────────
def test_ros_extends_espns_season_projection_over_the_window(loaded):
    """Rest-of-season value is ESPN's SEASON projection spread over the games
    that are left, not the published weeks summed — see the block at the foot of
    this file for why the second reading was dangerous."""
    _, ros = loaded
    row = ros.filter(pl.col("canonical_id") == "q1")
    assert row["anchor_points"][0] == pytest.approx(22.0 * 6)   # weeks 9-14


def test_a_bye_in_the_window_reduces_games_not_just_points(loaded):
    """The NFL plays 18 weeks and 17 GAMES. M7 paid 6.25% for the other error."""
    _, ros = loaded
    fa2 = ros.filter(pl.col("canonical_id") == "fa2")
    assert fa2["games_remaining"][0] == 5                       # bye week 14
    assert fa2["weekly_points"][0] == pytest.approx(11.0)


def test_players_with_no_games_left_are_dropped_not_divided_by_zero(espn):
    proj = ST._normalize(espn["projections"]).with_columns(
        pl.col("espn_id").alias("canonical_id"))
    byes = pl.DataFrame({"team": ["SEA"], "bye_week": [14]})
    out = ROS.from_espn(proj, from_week=14, byes=byes, season=2026)
    assert "d1" not in out.frame["canonical_id"].to_list()
    assert any("no games left" in n for n in out.notes)


# ─── the builders produce REAL digests ──────────────────────────────────────
def test_waivers_digest_names_an_actual_claim(loaded):
    st, ros = loaded
    d, fp = B.waivers_digest(st, ros, n_sims=400)
    assert not d.is_empty
    body = "\n".join(ln for _t, lines in d.sections for ln in lines)
    assert "Waiver WR" in body, body
    assert "pts/wk" in body
    assert fp["claims"], "fingerprint must carry the decision"


def test_the_worthless_free_agent_never_appears(loaded):
    """§9.3's bench-upgrades-are-worth-nothing, end to end."""
    st, ros = loaded
    d, _ = B.waivers_digest(st, ros, n_sims=400)
    body = "\n".join(ln for _t, lines in d.sections for ln in lines)
    assert "Deep Bench" not in body


def test_the_free_bye_reason_survives_all_the_way_to_the_email(loaded):
    """§2.1 is a property of MY schedule, and the digest has to say so or the
    number looks arbitrary."""
    st, ros = loaded
    d, _ = B.waivers_digest(st, ros, n_sims=400)
    body = "\n".join(ln for _t, lines in d.sections for ln in lines)
    if "Bye-14 RB" in body:
        assert "free weeks" in body


def test_lineup_digest_renders_a_full_starting_lineup(loaded):
    from ff_agent.config import STARTER_SLOTS
    st, ros = loaded
    kk = CK.kickoff_table(2026, pl.DataFrame([{
        "game_id": f"g{i}", "season": 2026, "week": WEEK, "game_type": "REG",
        "gameday": "2026-11-01", "gametime": "13:00", "weekday": "Sunday",
        "away_team": a, "home_team": h}
        for i, (a, h) in enumerate([
            ("BUF", "BAL"), ("SF", "KC"), ("DEN", "MIA"), ("GB", "NYJ"),
            ("LV", "DAL"), ("SEA", "CIN"), ("PHI", "MIN"), ("DET", "TB"),
            ("CAR", "CLE"), ("PIT", "ATL"), ("NO", "CHI"), ("HOU", "TEN"),
            ("IND", "JAX"), ("LAC", "ARI"), ("WAS", "NYG"), ("LAR", "NE")])]))
    d, fp = B.lineup_digest(st, ros, now=dt.datetime(2026, 10, 30, 9, 0, tzinfo=ET),
                            kickoffs=kk)
    starters = next(lines for t, lines in d.sections if t == "Starting lineup")
    assert len(starters) == sum(STARTER_SLOTS.values())
    assert fp["starters"]


def test_trades_digest_profiles_every_rival_even_with_no_trade(loaded):
    st, ros = loaded
    d, _ = B.trades_digest(st, ros, n_sims=400)
    profile = next(lines for t, lines in d.sections if t == "Every roster, ranked")
    assert len(profile) == len(TEAMS)
    assert any("DOUBLE-UP" in ln for ln in profile)


def test_week14_digest_sets_no_lineup(loaded):
    st, ros = loaded
    d, _ = B.week14_digest(st, ros)
    assert d.week == 14
    assert "No lineup to set" in d.headline


def test_team_means_covers_the_whole_league(loaded):
    st, ros = loaded
    weeks = B.remaining_play_weeks(2026, WEEK)
    means = B.team_means(st, ros, weeks)
    assert set(means) == set(TEAMS)
    assert all(v > 0 for v in means.values())


def test_remaining_play_weeks_is_per_team_not_global(loaded):
    """§2.1 and §2.2 are properties of MY schedule. M10a gave all nine teams my
    own byes by keying this wrong."""
    weeks = B.remaining_play_weeks(2026, WEEK)
    assert all(min(w) >= WEEK for w in weeks.values())


# ─── the stub can never come back ───────────────────────────────────────────
def test_no_cli_job_is_a_placeholder():
    """`cli monitor --job waivers` used to check ESPN was reachable and print a
    status line. This is the guard that it never silently does that again."""
    from pathlib import Path
    src = Path("ff_agent/cli.py").read_text()
    assert "_run_state_job" not in src
    assert "league state reachable" not in src
    for job in ("waivers", "trades", "week14"):
        assert f"B.{job}_digest" in src or f"cmd_{job}" in src


def test_every_real_job_actually_calls_a_builder():
    from pathlib import Path
    src = Path("ff_agent/cli.py").read_text()
    for fn in ("waivers_digest", "trades_digest", "week14_digest", "lineup_digest"):
        assert fn in src, f"cli never calls {fn}"


# ─── the collision guard must count PEOPLE, not rows ────────────────────────
# Found on the first live run: `monitor --job waivers` refused to start,
# reporting "553 canonical id(s) claimed by more than one ESPN player" and
# printing Philip Rivers fourteen times. 553 x 14 = 7,742, which was the row
# count — every player "colliding" with himself once per week, because the
# per-week projection table is LONG and the guard counted rows.
def _long_projection_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "espn_id": ["5529"] * 14 + ["8439"] * 14,
        "name": ["Philip Rivers"] * 14 + ["Aaron Rodgers"] * 14,
        "position": ["QB"] * 28,
        "team": ["LAC"] * 14 + ["NYJ"] * 14,
        "week": list(range(1, 15)) * 2,
        "projected_points": [18.0] * 28,
    })


@pytest.fixture
def passthrough_resolve(monkeypatch):
    ids = {"5529": "00-0022942", "8439": "00-0023459"}
    monkeypatch.setattr(ST, "resolve", lambda df, label, canonical=None: (
        df.with_columns(
            pl.col("espn_id").replace_strict(ids, default=None).alias("canonical_id"),
            pl.lit("mock").alias("match_method")),
        df.head(0).with_columns(pl.lit(None, dtype=pl.Utf8).alias("canonical_id")),
    ))


def test_a_long_frame_is_not_a_mass_collision(passthrough_resolve):
    """Fourteen weeks of Philip Rivers is not fourteen Philip Riverses."""
    long = _long_projection_frame()
    out, bad = ST.resolve_long(long, "projections")
    assert out.height == 28
    assert out["canonical_id"].n_unique() == 2
    assert bad.is_empty()


def test_resolve_long_asks_once_per_person_not_once_per_row(monkeypatch):
    """Also a performance property: 553 players x 14 weeks was 7,742 lookups."""
    seen = {}

    def counting(df, label, canonical=None):
        seen["rows"] = df.height
        return (df.with_columns(pl.col("espn_id").alias("canonical_id"),
                                pl.lit("mock").alias("match_method")),
                df.head(0).with_columns(pl.lit(None, dtype=pl.Utf8).alias("canonical_id")))

    monkeypatch.setattr(ST, "resolve", counting)
    ST.resolve_long(_long_projection_frame(), "projections")
    assert seen["rows"] == 2, "resolution must be asked once per PERSON"


def test_a_real_collision_is_still_fatal(monkeypatch):
    """Two DIFFERENT espn_ids on one canonical id silently merges two people,
    which is worse than not resolving one. The looser row count must not have
    loosened this."""
    from ff_agent.data import crosswalk as CW

    merged = pl.DataFrame({
        "espn_id": ["111", "222"], "name": ["Player A", "Player B"],
        "position": ["WR", "WR"], "team": ["SF", "SF"],
        "canonical_id": ["00-X", "00-X"], "match_method": ["m", "m"]})
    monkeypatch.setattr(CW, "resolve_players", lambda df, canonical=None: merged)
    with pytest.raises(ST.StateError, match="more than one ESPN player"):
        ST.resolve(merged.drop("canonical_id", "match_method"), "test")


def test_the_collision_report_lists_each_player_once(monkeypatch):
    """The false alarm printed 7,742 rows. A real one has to stay readable."""
    from ff_agent.data import crosswalk as CW

    merged = pl.DataFrame({
        "espn_id": ["111"] * 5 + ["222"] * 5,
        "name": ["Player A"] * 5 + ["Player B"] * 5,
        "position": ["WR"] * 10, "team": ["SF"] * 10,
        "canonical_id": ["00-X"] * 10, "match_method": ["m"] * 10})
    monkeypatch.setattr(CW, "resolve_players", lambda df, canonical=None: merged)
    with pytest.raises(ST.StateError) as exc:
        ST.resolve(merged.drop("canonical_id", "match_method"), "test")
    assert str(exc.value).count("Player A") == 1


# ─── the four jobs that used to share one generic stub ──────────────────────
def test_freeagents_names_who_cleared(loaded, tmp_path, monkeypatch):
    """§9.3 asks the tool to say 'this one will clear — grab him Wednesday'.
    Tuesday makes the prediction; this is the half that CHECKS it."""
    from ff_agent.inseason import log as LOG
    monkeypatch.setattr(LOG, "log_path", lambda season=2026: tmp_path / "l.jsonl")
    st, ros = loaded
    LOG.write("waivers", season=2026, week=WEEK,
              decision={"claims": ["fa1"], "grabs": ["fa2"]})
    d, fp = B.freeagents_digest(st, ros, added_ids=set())
    assert "fa2" in fp["cleared"]
    body = "\n".join(ln for _t, lines in d.sections for ln in lines)
    assert "Bye-14 RB" in body
    assert any("accuracy" in n for n in d.notes)


def test_freeagents_reports_a_wrong_prediction_as_wrong(loaded, tmp_path, monkeypatch):
    """A prediction that only ever reports its hits is not a measurement."""
    from ff_agent.inseason import log as LOG
    monkeypatch.setattr(LOG, "log_path", lambda season=2026: tmp_path / "l.jsonl")
    st, ros = loaded
    LOG.write("waivers", season=2026, week=WEEK, decision={"grabs": ["fa2"]})
    d, fp = B.freeagents_digest(st, ros, added_ids={"fa2"})
    assert fp["taken"] == ["fa2"]
    assert any("was claimed" in t for t, _ in d.sections)


def test_freeagents_says_when_it_cannot_tell_who_was_taken(loaded, tmp_path, monkeypatch):
    from ff_agent.inseason import log as LOG
    monkeypatch.setattr(LOG, "log_path", lambda season=2026: tmp_path / "l.jsonl")
    st, ros = loaded
    d, _ = B.freeagents_digest(st, ros, added_ids=None)
    assert any("no transaction log" in n for n in d.notes)


def test_injuries_flags_a_questionable_starter_with_the_measured_rate(loaded):
    """F10: the word 'questionable' alone is not actionable — a Questionable QB
    sits roughly twice as often as a Questionable skill player."""
    st, ros = loaded
    inj = pl.DataFrame({"gsis_id": ["q1"], "report_status": ["Questionable"],
                        "practice_status": ["Limited Participation in Practice"]})
    d, fp = B.injuries_digest(st, ros, injuries=inj)
    assert "q1" in fp["flagged"]
    body = "\n".join(ln for _t, lines in d.sections for ln in lines)
    assert "% to sit" in body
    assert "twice as often" in body, "the QB asymmetry must be stated"


def test_injuries_distinguishes_no_report_from_a_healthy_roster(loaded):
    """'Everyone reads as healthy' is a statement about the DATA."""
    st, ros = loaded
    d, _ = B.injuries_digest(st, ros, injuries=None)
    assert any("statement about the DATA" in n for n in d.notes)
    clean, _ = B.injuries_digest(st, ros, injuries=pl.DataFrame({
        "gsis_id": ["q1"], "report_status": [None], "practice_status": [None]}))
    assert "nobody flagged" in clean.subject


def test_the_scoring_tripwire_is_silent_when_clean(monkeypatch):
    """F2. A clean run says nothing — §6.4."""
    from ff_agent.scoring import validate as VAL
    monkeypatch.setattr(VAL, "layer_a_rules_check", lambda season, weeks=None:
                        pl.DataFrame({"week": [1, 1], "name": ["A", "B"],
                                      "position": ["QB", "RB"],
                                      "espn_points": [20.0, 10.0],
                                      "rules_points": [20.0, 10.0],
                                      "delta": [0.0, 0.0]}))
    alarms, notes = B.scoring_tripwire(2026, 1)
    assert alarms == []
    assert any("clean" in n for n in notes)


def test_the_scoring_tripwire_alarms_on_a_rule_change(monkeypatch):
    """This league changed its scoring once already (PC and INC removed after
    2024). The tripwire turns 'found in January' into 'found next Tuesday'."""
    from ff_agent.scoring import validate as VAL
    monkeypatch.setattr(VAL, "layer_a_rules_check", lambda season, weeks=None:
                        pl.DataFrame({"week": [1], "name": ["A"],
                                      "position": ["QB"], "espn_points": [20.0],
                                      "rules_points": [23.5], "delta": [3.5]}))
    alarms, notes = B.scoring_tripwire(2026, 1)
    assert alarms and "SCORING MAY HAVE CHANGED" in alarms[0]
    assert any("delta" in n or "+3.50" in n for n in notes)


def test_refresh_is_silent_when_nothing_is_wrong(loaded, monkeypatch):
    monkeypatch.setattr(B, "scoring_tripwire", lambda s, w: ([], ["clean"]))
    st, ros = loaded
    d, _ = B.refresh_digest(st, ros)
    assert d.is_empty, "a clean refresh must send nothing (§6.4)"


def test_refresh_speaks_when_the_tripwire_fires(loaded, monkeypatch):
    monkeypatch.setattr(B, "scoring_tripwire",
                        lambda s, w: (["SCORING MAY HAVE CHANGED: 5 rows"], ["x"]))
    st, ros = loaded
    d, _ = B.refresh_digest(st, ros)
    assert not d.is_empty and d.urgent


def test_most_added_rebuilds_the_control_from_transactions():
    """F4: player_owned_espn is null in-season, so the control is rebuilt from
    what the other managers actually DID."""
    from ff_agent.inseason import audit as AU
    txns = pl.DataFrame({
        "scoring_period": [1] * 5, "fantasy_team": ["A", "B", "C", "D", "E"],
        "action": ["WAIVER ADDED"] * 4 + ["FA ADDED"],
        "espn_id": ["x", "x", "x", "y", "y"],
        "name": ["X"] * 3 + ["Y", "Y"]})
    assert AU.most_added(txns, 1) == "x"
    assert AU.most_added(pl.DataFrame(schema={
        "scoring_period": pl.Int64, "fantasy_team": pl.Utf8, "action": pl.Utf8,
        "espn_id": pl.Utf8, "name": pl.Utf8}), 1) is None


def test_the_season_verdict_says_UNMEASURED_rather_than_passing():
    """M7's precedent: a number with no control beside it should not be
    believed, so 'no control' must not read as success."""
    from ff_agent.inseason import audit as AU
    unmeasured = pl.DataFrame({"week": [1, 2], "measured": [False, False],
                               "edge_vs_control": [None, None]})
    assert "UNMEASURED" in AU.season_verdict(unmeasured)
    assert "nothing logged" in AU.season_verdict(pl.DataFrame())
    losing = pl.DataFrame({"week": [1], "measured": [True],
                           "edge_vs_control": [-2.0]})
    assert "LOSES" in AU.season_verdict(losing)
    assert "ship the control" in AU.season_verdict(losing)


def test_the_backtest_gate_still_cannot_pass_on_no_data():
    from ff_agent.inseason import backtest as BT
    assert not BT.run().passed


# ─── the six bugs the first live multi-job run exposed ──────────────────────
def test_espn_publishing_only_one_week_does_not_shrink_everyone(loaded):
    """THE CRITICAL ONE. ESPN publishes only the weeks it has projected —
    often just the upcoming one. Summing the window and treating unpublished
    weeks as zero made Justin Herbert 1.6 pts/wk and a full starting lineup
    10.3 instead of ~130."""
    proj = pl.DataFrame([
        {"canonical_id": "h", "name": "QB", "position": "QB", "team": "BUF",
         "week": wk, "projected_points": 22.4 if wk == 1 else None}
        for wk in range(1, 15)
    ])
    out = ROS.from_espn(proj, from_week=1, season=2026,
                        byes=pl.DataFrame({"team": ["BUF"], "bye_week": [7]}))
    row = out.frame.filter(pl.col("canonical_id") == "h")
    assert row["weekly_points"][0] == pytest.approx(22.4)
    assert row["ros_points"][0] == pytest.approx(22.4 * 13)
    assert row["weeks_projected"][0] == 1


def test_partial_publishing_does_not_rank_by_espns_schedule(loaded):
    """Worse than uniformly wrong, the old bug was UNEVEN: an identical player
    with two published weeks outranked one with a single published week."""
    rows = []
    for cid, n_pub in (("one", 1), ("two", 2)):
        for wk in range(1, 15):
            rows.append({"canonical_id": cid, "name": cid, "position": "RB",
                         "team": "SF", "week": wk,
                         "projected_points": 15.0 if wk <= n_pub else None})
    out = ROS.from_espn(pl.DataFrame(rows), from_week=1, season=2026).frame
    vals = dict(zip(out["canonical_id"].to_list(), out["weekly_points"].to_list()))
    assert vals["one"] == pytest.approx(vals["two"]), (
        "two identical players must not be separated by ESPN's publishing order"
    )


def test_coverage_is_stated_not_hidden_in_a_total(loaded):
    """'ESPN projected one of fourteen weeks' and 'this player scores little'
    look identical in a total."""
    proj = pl.DataFrame([
        {"canonical_id": "h", "name": "QB", "position": "QB", "team": "BUF",
         "week": wk, "projected_points": 20.0 if wk == 1 else None}
        for wk in range(1, 15)
    ])
    out = ROS.from_espn(proj, from_week=1, season=2026)
    assert any("median of 1 of the 14" in n for n in out.notes)


def test_a_player_with_no_projection_is_dropped_not_priced_at_zero(loaded):
    rows = [{"canonical_id": "real", "name": "R", "position": "RB", "team": "SF",
             "week": wk, "projected_points": 12.0} for wk in range(1, 15)]
    rows += [{"canonical_id": "blank", "name": "B", "position": "RB", "team": "SF",
              "week": wk, "projected_points": None} for wk in range(1, 15)]
    out = ROS.from_espn(pl.DataFrame(rows), from_week=1, season=2026)
    assert "blank" not in out.frame["canonical_id"].to_list()
    assert any("no projection at all" in n for n in out.notes)


def test_no_projections_at_all_is_refused_not_priced_at_zero(loaded):
    proj = pl.DataFrame([
        {"canonical_id": "h", "name": "QB", "position": "QB", "team": "BUF",
         "week": wk, "projected_points": None} for wk in range(1, 15)])
    with pytest.raises(ROS.ROSError, match="not the same as projecting zero"):
        ROS.from_espn(proj, from_week=1, season=2026)


def test_the_sack_note_is_not_printed_twice(loaded):
    """The caller knows WHY there is no input — 'it is week 2' reads very
    differently from 'the cache is missing'. Two sentences saying it vaguely is
    worse than one saying it precisely."""
    st, ros = loaded
    out = ROS.from_espn(
        ST._normalize(pl.DataFrame([
            {"espn_id": "h", "canonical_id": "h", "name": "QB", "position": "QB",
             "team": "BUF", "week": wk, "projected_points": 20.0}
            for wk in range(1, 15)])),
        from_week=1, season=2026, soe=None)
    sack_notes = [n for n in out.notes if "sacks-over-expected" in n]
    assert len(sack_notes) == 0, "from_espn must stay silent; the caller explains"


def test_a_notes_only_digest_still_reaches_the_caller(tmp_path, monkeypatch):
    """Three jobs looked like they had done nothing, because a digest whose
    entire content was a note got discarded before the CLI could show it."""
    from ff_agent.inseason import jobs as J
    from ff_agent.inseason import log as LOG
    from ff_agent.inseason.notify import Digest, MemoryNotifier

    monkeypatch.setattr(LOG, "STATE_DIR", tmp_path / "s")
    monkeypatch.setattr(LOG, "log_path", lambda season=2026: tmp_path / "l.jsonl")
    n = MemoryNotifier()
    res = J.run("injuries",
                lambda: (Digest(job="injuries", subject="no injury report",
                                notes=["no injury report available"]), {}),
                n, week=1, skip_gate=True)
    assert res.digest is not None, "the caller must still be able to SHOW it"
    assert res.sent is False and n.sent == [], "but it must not be SENT (§6.4)"
    assert "nothing to say" in res.detail


def test_the_lineup_headline_separates_locked_from_recommended(loaded):
    """'1 locked' on a Wednesday when nothing has kicked off meant '1 we
    suggest you commit'. Two different facts, one number."""
    kk = CK.kickoff_table(2026, pl.DataFrame([{
        "game_id": "g1", "season": 2026, "week": WEEK, "game_type": "REG",
        "gameday": "2026-11-01", "gametime": "13:00", "weekday": "Sunday",
        "away_team": a, "home_team": h}
        for a, h in [("BUF", "BAL"), ("SF", "KC"), ("DEN", "MIA"), ("GB", "NYJ"),
                     ("LV", "DAL"), ("SEA", "CIN")]]))
    st, ros = loaded
    d, _ = B.lineup_digest(st, ros, now=dt.datetime(2026, 10, 30, 9, 0, tzinfo=ET),
                           kickoffs=kk)
    assert "already locked by kickoff" in d.headline
    assert "recommended to commit" in d.headline


# ─── "drop Brian Thomas Jr. for a kicker" ───────────────────────────────────
# The live run recommended cutting a real starting receiver for Chris Boswell.
# Two independent halves, both now fixed:
#   a) ESPN published his week-1 projection as 0.0 (out), and requiring `> 0`
#      threw that away as though it were missing data, so he vanished from the
#      priced frame entirely;
#   b) with_values then silently filled him to 0.0, making him the "cheapest"
#      thing on the roster and therefore the obvious drop.
def test_a_published_zero_is_data_not_a_missing_value():
    """A published 0.0 is what ESPN says about a player who is out. Discarding
    it is discarding information, not noise."""
    rows = [{"canonical_id": "out", "name": "Out Guy", "position": "WR",
             "team": "JAX", "week": wk,
             "projected_points": 0.0 if wk == 1 else None} for wk in range(1, 15)]
    out = ROS.from_espn(pl.DataFrame(rows), from_week=1, season=2026,
                        byes=pl.DataFrame({"team": ["JAX"], "bye_week": [8]}))
    assert "out" in out.frame["canonical_id"].to_list(), (
        "a player ESPN priced at zero must stay in the frame, priced at zero"
    )
    assert out.frame.filter(pl.col("canonical_id") == "out")["weeks_projected"][0] == 1


def test_the_bye_week_is_excluded_from_the_rate_not_counted_as_a_zero():
    """games_remaining already excludes the bye — counting it in the rate too
    would charge for the same bye twice."""
    rows = [{"canonical_id": "p", "name": "P", "position": "RB", "team": "SF",
             "week": wk, "projected_points": 0.0 if wk == 7 else 12.0}
            for wk in range(1, 15)]
    out = ROS.from_espn(pl.DataFrame(rows), from_week=1, season=2026,
                        byes=pl.DataFrame({"team": ["SF"], "bye_week": [7]}))
    row = out.frame.filter(pl.col("canonical_id") == "p")
    assert row["weekly_points"][0] == pytest.approx(12.0), "bye must not dilute the rate"
    assert row["games_remaining"][0] == 13


def test_an_unpriced_rostered_player_is_flagged_not_silently_zeroed(loaded):
    """`priced` separates 'ESPN says he scores nothing' from 'we have no number
    for him'. Both arrive as zero and mean opposite things."""
    st, ros = loaded
    thinned = ros.filter(pl.col("canonical_id") != "w1")
    out = ST.with_values(st.my_roster, thinned, "my roster")
    row = out.filter(pl.col("canonical_id") == "w1")
    assert row["priced"][0] is False
    assert row["weekly_points"][0] == 0.0        # still usable in the lineup math
    assert ST.unpriced(out)["canonical_id"].to_list() == ["w1"]


def test_an_unpriced_player_is_never_offered_as_a_drop(loaded):
    """Not knowing a player's value is a reason for caution, not a licence to
    cut him — and a zero always looks like the cheapest thing to lose."""
    from ff_agent.inseason import waivers as WV
    st, ros = loaded
    thinned = ros.filter(pl.col("canonical_id") != "r1")   # my best RB, now blind
    mine = ST.align(ST.with_values(st.my_roster, thinned, "mine"))
    pool = ST.align(ST.with_values(st.free_agents, ros, "fa"))
    pairs = WV.candidate_pairs(mine, pool, tuple(range(WEEK, 15)))
    assert all(drop != "r1" for _add, drop, _v in pairs), (
        "the unpriced player must not be the drop the engine reaches for first"
    )


def test_the_digest_names_unpriced_roster_players(loaded):
    st, ros = loaded
    thinned = ros.filter(pl.col("canonical_id") != "w1")
    d, _ = B.waivers_digest(st, thinned, n_sims=400)
    assert any("NO PROJECTION" in n for n in d.notes)
    assert any("My WR1" in n for n in d.notes)


# ─────────────────────────────────────────────────────────────────────────────
# The rest-of-season anchor is ESPN's SEASON projection, not one published week
#
# Found on the first live in-season run, twice. The first version summed the
# remaining weeks and treated unpublished ones as zero, so everyone read as a
# fraction of himself. The second took the mean of the published weeks and
# extended it, which fixed the scale and not the substance: with a median of ONE
# of fourteen weeks published, a "rest-of-season" number was this week's
# projection multiplied by fourteen — this week's opponent, this week's snap
# count and this week's injury designation included. A starting receiver ESPN
# had projected at 0.0 for week 1 because he was out became worth zero for the
# SEASON, and therefore the cheapest thing on the roster to cut.
# ─────────────────────────────────────────────────────────────────────────────
def _proj_rows(cid, name, pos, team, week_one, season_proj, weeks=range(1, 15)):
    return [{
        "canonical_id": cid, "name": name, "position": pos, "team": team,
        "week": wk,
        "projected_points": (week_one if wk == 1 else None),
        "season_projected_points": season_proj,
        "season_actual_points": 0.0,
        "season_projected_avg": (season_proj / 17 if season_proj is not None else None),
    } for wk in weeks]


def _one_week_league():
    """ESPN as it actually is at week 1: one week published, seasons for all."""
    rows = []
    rows += _proj_rows("out_wr", "Brian Thomas Jr.", "WR", "SF", 0.0, 210.0)
    rows += _proj_rows("qb1", "Justin Herbert", "QB", "KC", 20.6, 350.0)
    rows += _proj_rows("kicker", "Chris Boswell", "K", "SF", 8.5, 140.0)
    return pl.DataFrame(rows)


def test_a_week_one_zero_does_not_write_off_the_season():
    """The Brian Thomas Jr. bug, pinned by name.

    ESPN says 0.0 for week 1 (he is out) and 210 for the season. Those are both
    true and they answer different questions. Pricing the season off the week
    made him worth nothing and offered him as a drop for a kicker.
    """
    out = ROS.from_espn(_one_week_league(), from_week=1, season=2026)
    row = out.frame.filter(pl.col("canonical_id") == "out_wr").to_dicts()[0]

    assert row["anchor_source"] == "season"
    # 210 over 17 NFL games, extended across the 14-week fantasy window.
    assert row["weekly_points"] == pytest.approx(210 / 17, abs=0.01)
    # And he is worth more than the kicker, which is the whole point.
    kick = out.frame.filter(pl.col("canonical_id") == "kicker").to_dicts()[0]
    assert row["weekly_points"] > kick["weekly_points"]


def test_the_rescued_players_are_named_in_the_digest():
    out = ROS.from_espn(_one_week_league(), from_week=1, season=2026)
    assert any("Brian Thomas Jr." in n for n in out.notes), out.notes


def test_this_weeks_projection_is_kept_for_the_weekly_call():
    """Both numbers survive: the season one prices him, the weekly one starts him.

    Dropping him and benching him are opposite calls and the engine has to be
    able to make them at the same time.
    """
    out = ROS.from_espn(_one_week_league(), from_week=1, season=2026)
    row = out.frame.filter(pl.col("canonical_id") == "out_wr").to_dicts()[0]
    assert row["espn_projection"] == 0.0
    swapped = LN.this_week_value(out.frame)
    got = swapped.filter(pl.col("canonical_id") == "out_wr")["weekly_points"][0]
    assert got == 0.0
    # ...while the free-agent tail, which has no weekly number at all, keeps its
    # season rate rather than falling to zero.
    tail = ROS.from_espn(
        pl.DataFrame(_proj_rows("tail", "Deep Guy", "WR", "SF", None, 40.0)),
        from_week=1, season=2026)
    assert LN.this_week_value(tail.frame)["weekly_points"][0] > 0


def test_the_season_rate_is_divided_by_nfl_games_not_fantasy_weeks():
    """17 games, not the 14-week fantasy window.

    M7 paid 6.25% for dividing by 16 on the opposite reasoning. Dividing a
    17-game projection by a 14-week window inflates every player by 21%.
    """
    out = ROS.from_espn(
        pl.DataFrame(_proj_rows("p", "P", "RB", "SF", 10.0, 170.0)),
        from_week=1, season=2026)
    assert out.frame["weekly_points"][0] == pytest.approx(10.0, abs=0.01)


def test_points_already_scored_are_not_projected_again():
    """In week 9 the season projection still spans weeks 1-18. What is LEFT is
    the projection minus what is already banked, or a player who started hot is
    projected to score his whole season all over again."""
    rows = _proj_rows("p", "P", "RB", "SF", 10.0, 170.0, weeks=range(9, 15))
    for r in rows:
        r["season_actual_points"] = 100.0
    out = ROS.from_espn(pl.DataFrame(rows), from_week=9, season=2026)
    # 70 points left over 9 remaining NFL games.
    assert out.frame["weekly_points"][0] == pytest.approx(70 / 9, abs=0.01)


def test_a_seventeen_times_disagreement_is_reported_not_absorbed():
    """M3's "read literally, every projection is 17x wrong" trap, in its new home.

    ESPN ships appliedAverage beside appliedTotal, so if the total were secretly
    a per-game number the ratio would sit near 17. That has to be loud: every
    rest-of-season number in the digest depends on which one it is.
    """
    rows = _proj_rows("p", "P", "RB", "SF", 10.0, 170.0)
    for r in rows:
        r["season_projected_avg"] = 170.0  # as if the total were per-game
    out = ROS.from_espn(pl.DataFrame(rows), from_week=1, season=2026)
    assert any("per-game average" in n for n in out.notes), out.notes


def test_a_pre_migration_table_still_works_and_says_so():
    """A cache written before the season columns existed is incomplete, not
    stale — the TTL cannot see it. It still produces a board; it says loudly
    that the board is one week doing a season's job."""
    rows = [{"canonical_id": "p", "name": "P", "position": "RB", "team": "SF",
             "week": wk, "projected_points": (10.0 if wk == 1 else None)}
            for wk in range(1, 15)]
    out = ROS.from_espn(pl.DataFrame(rows), from_week=1, season=2026)
    assert out.frame["anchor_source"][0] == "week"
    assert any("predates the season-projection columns" in n for n in out.notes)
