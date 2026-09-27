"""M10b-1's gate: the container comes up correctly, and the backtest has teeth."""
import re
from pathlib import Path

import polars as pl
import pytest

from ff_agent.inseason import backtest as B

ROOT = Path(__file__).resolve().parent.parent
DOCKER = ROOT / "docker"


# ─── the container ──────────────────────────────────────────────────────────
def test_every_container_file_exists():
    for name in ("Dockerfile", "compose.yml", "crontab", "entrypoint.sh"):
        assert (DOCKER / name).exists(), name


def test_the_timezone_is_set_in_the_image_and_asserted_at_startup():
    """F7: a container defaults to UTC, and a job written as 11:15 then fires at
    06:15 or 07:15 ET depending on daylight saving — four hours before the
    inactives it exists to read."""
    df = (DOCKER / "Dockerfile").read_text()
    assert "ENV TZ=America/New_York" in df
    entry = (DOCKER / "entrypoint.sh").read_text()
    assert 'if [ "${TZ:-}" != "America/New_York" ]' in entry
    assert "exit 1" in entry


def test_preflight_runs_before_the_scheduler():
    """A container that comes up broken must say so immediately, not at 08:00
    on Tuesday."""
    entry = (DOCKER / "entrypoint.sh").read_text()
    pre = entry.index("--job preflight")
    cron = entry.index("supercronic")
    assert pre < cron
    assert "set -euo pipefail" in entry


def test_the_container_does_not_run_as_root():
    df = (DOCKER / "Dockerfile").read_text()
    assert "USER ff" in df
    assert df.index("USER ff") < df.index("ENTRYPOINT")


def test_secrets_are_mounted_never_baked_into_the_image():
    df = (DOCKER / "Dockerfile").read_text()
    assert "COPY .env" not in df
    compose = (DOCKER / "compose.yml").read_text()
    assert "../.env" in compose and "../.env.notify" in compose


def test_mail_credentials_live_in_a_separate_file_from_the_league_login():
    """One leak must not be two."""
    compose = (DOCKER / "compose.yml").read_text()
    assert "../.env\n" in compose or "- ../.env\n" in compose
    assert "../.env.notify" in compose
    example = (ROOT / ".env.notify.example").read_text()
    assert "ESPN_S2" not in example and "SMTP_PASSWORD" in example


def test_the_crontab_parses_and_has_a_fifteen_minute_tick():
    """F9: the lineup has no fixed time, so the crontab is a dumb tick and
    clock.py decides."""
    lines = [
        ln for ln in (DOCKER / "crontab").read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert lines
    for ln in lines:
        fields = ln.split(None, 5)
        assert len(fields) == 6, f"not a 5-field cron line: {ln}"
        assert "ff_agent.cli monitor --job" in fields[5]
    assert any(ln.split()[0] == "*/15" for ln in lines), "no tick"


def test_the_crontab_has_no_fixed_lineup_or_inactives_entry():
    """§9.1's four fixed slots would miss the Wednesday opener, all three
    Christmas games, and the whole Sunday late slate."""
    cron = (DOCKER / "crontab").read_text()
    jobs = re.findall(r"--job (\w+)", cron)
    assert "tick" in jobs
    assert "lineup" not in jobs and "inactives" not in jobs


def test_the_cache_and_logs_persist_across_restarts():
    compose = (DOCKER / "compose.yml").read_text()
    for vol in ("data/cache", "logs", "state"):
        assert vol in compose, vol


def test_setup_documents_the_step_that_makes_email_buzz():
    """§6.3's honest problem: email does not buzz a phone by default, and for
    the Sunday job that is the difference between working and existing."""
    doc = (ROOT / "SETUP_MONITOR.md").read_text()
    assert "[FDS URGENT]" in doc
    assert "High priority only" in doc or "VIP" in doc


# ─── the gate ───────────────────────────────────────────────────────────────
def test_a_gate_with_no_data_does_not_pass():
    """A gate that can only pass is not a gate. Unmeasured is not passing."""
    g = B.run()
    assert not g.passed
    assert all(a.verdict == "UNMEASURED" for a in g.arms)


def test_the_caveats_are_printed_before_the_numbers():
    """M5/M6's habit: state what the measurement cannot support first."""
    report = B.run().report()
    assert report.index("CAVEATS") < report.index("ARM")
    assert "EIGHT-team" in report
    assert "ONE season" in report or "One season" in report


def test_every_arm_names_its_control():
    for a in B.run().arms:
        assert a.control_name and a.control_name != a.name


def test_losing_to_the_control_says_ship_the_control():
    """M7's precedent: a policy with zero board edge captured +29.9 against the
    full model's +32.1. Without a control the whole 32 gets banked as skill."""
    lose = B.ArmResult("waivers", 5, 1.0, 2.0, "most-added")
    win = B.ArmResult("waivers", 5, 3.0, 2.0, "most-added")
    assert lose.verdict == "SHIP THE CONTROL"
    assert win.verdict == "BEATS CONTROL"


def test_the_thursday_arm_exists_because_it_justifies_the_machinery():
    """If the counterfactual cannot beat naively starting the higher projection,
    §5.3's simulation is decoration and the naive rule ships."""
    arm = B.thursday_arm([
        {"our_pick": "a", "naive_pick": "b", "actual_points": {"a": 18.0, "b": 12.0}},
        {"our_pick": "c", "naive_pick": "d", "actual_points": {"c": 9.0, "d": 11.0}},
    ])
    assert arm.n == 2
    assert arm.ours == pytest.approx(13.5)
    assert arm.control == pytest.approx(11.5)
    assert arm.verdict == "BEATS CONTROL"


def test_the_lineup_replay_solves_on_positions_and_never_fakes_our_projection(monkeypatch):
    """Two bugs, one path. load_season used slot_position AS position, so a
    benched player ("BE") could never be started and FLEX/D-ST could never be
    filled; and it filled OUR projection from ESPN's, so ours and the control
    were one column and the arm read a measured edge of exactly zero."""
    from ff_agent.config import MY_TEAM_NAME
    from ff_agent.data import espn as ESPN

    rows = [  # (id, position, slot, espn proj, actual)
        ("q1", "QB", "QB", 20, 20), ("q2", "QB", "QB", 18, 18),
        ("r1", "RB", "RB", 15, 10), ("r2", "RB", "RB", 14, 9),
        ("r3", "RB", "BE", 6, 30),                     # the one who should start
        ("w1", "WR", "WR", 13, 12), ("w2", "WR", "WR", 12, 11),
        ("w3", "WR", "RB/WR/TE", 11, 8),
        ("t1", "TE", "TE", 9, 7), ("k1", "K", "K", 8, 8),
        ("d1", "D/ST", "D/ST", 7, 5),
    ]
    box = pl.DataFrame([
        {"week": 4, "fantasy_team": MY_TEAM_NAME, "espn_id": i, "name": i,
         "position": p, "slot_position": s, "points": float(a),
         "projected_points": float(e)} for i, p, s, e, a in rows])
    monkeypatch.setattr(ESPN, "started_lineup", lambda season, wk: box)
    monkeypatch.setattr(ESPN, "transactions", lambda season, wk: pl.DataFrame())

    data = B.load_season(2025, (4,))
    roster = data["lineup_weeks"][0]["roster"]
    assert set(roster["position"]) == {"QB", "RB", "WR", "TE", "K", "DST"}
    assert "weekly_points" not in roster.columns, "ours must never be ESPN's"
    assert B.lineup_arm(data["lineup_weeks"]).verdict == "UNMEASURED"

    # once a replay supplies our number, the bench RB can be started
    wk = dict(data["lineup_weeks"][0])
    wk["roster"] = roster.with_columns(
        pl.col("canonical_id").replace_strict(
            {i: float(a) for i, _, _, _, a in rows}).alias("weekly_points"))
    arm = B.lineup_arm([wk])
    assert arm.n == 1
    assert "r3" in B.LN.LU.optimal_lineup(wk["roster"])["canonical_id"].to_list()
    assert B.LN.LU.optimal_lineup(wk["roster"]).height == 10   # FLEX and DST filled
    assert arm.ours > arm.control


def test_an_unmeasured_arm_says_so_rather_than_passing_quietly():
    arm = B.thursday_arm([])
    assert arm.verdict == "UNMEASURED"
    assert any("not the same as passing" in n for n in arm.notes)


def test_claim_calibration_is_scored_against_an_uninformative_prior():
    """Calibration, not a contest — the honest baseline for a probability nobody
    has measured before is a coin flip."""
    good = B.claim_calibration([0.9, 0.1, 0.8], [True, False, True])
    bad = B.claim_calibration([0.1, 0.9, 0.2], [True, False, True])
    assert good.edge > 0
    assert bad.edge < 0
