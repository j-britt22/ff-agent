"""M10b-1 — delivery: the notifier, the digest, dedupe, and §0.1's scan.

The scan is the most important test in this package. M1-M10a all ran with a
human at the keyboard; this one runs unattended on a schedule, which makes it the
first place a §0.1 violation could happen with nobody watching.
"""
import datetime as dt
from pathlib import Path

import pytest

from ff_agent.inseason import digest as DG
from ff_agent.inseason import jobs as J
from ff_agent.inseason import log as LOG
from ff_agent.inseason.clock import ET
from ff_agent.inseason.notify import (
    Digest, MemoryNotifier, NullNotifier, SecretLeak, assert_no_secrets,
)

NOW = dt.datetime(2026, 10, 25, 11, 46, tzinfo=ET)
DEADLINE = dt.datetime(2026, 10, 25, 13, 0, tzinfo=ET)


def sample(**kw) -> Digest:
    base = dict(
        job="inactives", subject="Nacua OUT. 1 swap", week=7, urgent=True,
        headline="4 slots open, 3 locked",
        sections=[("Swap", ["WR2 Puka Nacua (OUT) -> Jayden Reed: +4.1 pts"])],
        deadline=DEADLINE)
    base.update(kw)
    return Digest(**base)


# ─── §0.1 — the scan that matters most ──────────────────────────────────────
FORBIDDEN = (
    ".post(", ".put(", ".patch(", ".delete(",
    "requests.post", "httpx.post", "urlopen",
    "add_to_roster", "submit_pick", "place_claim", "propose_trade",
    ".add_player(", ".drop_player(", ".set_lineup(", ".submit(",
)
"""Call-shaped, exactly as ``test_live.py`` matches them.

Bare words would be wrong here in a way they are not in ``live/``: this package
DESCRIBES the rule in prose that reaches the reader — every digest ends with
"nothing here has been submitted to ESPN, and nothing in this tool can" — so a
substring match on "submit" flags the sentence asserting the rule as a violation
of it. Which is precisely what happened the first time this test ran.
"""

PACKAGE = Path(__file__).resolve().parent.parent / "ff_agent" / "inseason"


def test_the_inseason_package_cannot_write_to_espn():
    """§0.1 is absolute and asserted, not trusted.

    M9 recorded that this rule needed a SECOND guard once a browser was involved.
    An unattended container on a schedule is the third place it could break, and
    the first where nobody would notice for weeks.
    """
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        src = path.read_text()
        for token in FORBIDDEN:
            if token in src:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, (
        "§0.1: write-capable ESPN calls found in the in-season package. "
        f"{offenders}"
    )


def test_the_espn_data_layer_still_exposes_no_write_path():
    """The package can only be as read-only as what it reads through."""
    src = (Path(__file__).resolve().parent.parent / "ff_agent" / "data"
           / "espn.py").read_text()
    for token in FORBIDDEN:
        assert token not in src, f"data/espn.py contains {token}"


def test_the_only_outbound_path_is_smtp_and_it_is_scanned(monkeypatch):
    """An outbound email is the only way a credential could leave the machine.
    Every other component reads."""
    monkeypatch.setenv("ESPN_S2", "AEBb7RiVpX%2FQm3kZ9nT4uLc%2BdW1yH0jGf8sOaPzE%3D")
    with pytest.raises(SecretLeak, match="ESPN_S2"):
        assert_no_secrets("debug: cookie=AEBb7RiVpX%2FQm3kZ9nT4uLc%2BdW1yH0jGf8sOaPzE%3D")


def test_a_swid_shaped_string_is_caught_even_if_it_is_not_in_the_environment():
    """Catches a value that reached the digest from a cached ESPN payload rather
    than from the environment — checking only os.environ would miss it."""
    with pytest.raises(SecretLeak, match="SWID"):
        assert_no_secrets("owner {1A2B3C4D-5E6F-7A8B-9C0D-1E2F3A4B5C6D} added a player")


def test_a_short_value_is_not_treated_as_a_secret(monkeypatch):
    """A two-character league id would otherwise flag every digest."""
    monkeypatch.setenv("ESPN_S2", "ab")
    assert_no_secrets("the letters ab appear in this sentence")


def test_the_scan_runs_on_every_send(monkeypatch):
    monkeypatch.setenv("ESPN_SWID", "{1A2B3C4D-5E6F-7A8B-9C0D-1E2F3A4B5C6D}")
    n = MemoryNotifier()
    d = sample(sections=[("Oops", ["{1A2B3C4D-5E6F-7A8B-9C0D-1E2F3A4B5C6D}"])])
    with pytest.raises(SecretLeak):
        n.send(d, DG.to_text(d), DG.to_html(d))
    assert n.sent == []


# ─── the digest ─────────────────────────────────────────────────────────────
def test_every_message_leads_with_the_deadline():
    """§5.3: a recommendation that arrives after its deadline is worse than none."""
    text = DG.to_text(sample(), NOW)
    assert text.splitlines()[3].startswith("NEXT LOCK")
    assert "in 74 min" in text


def test_a_passed_deadline_says_so_rather_than_counting_backwards():
    late = DG.to_text(sample(), dt.datetime(2026, 10, 25, 13, 30, tzinfo=ET))
    assert "PASSED 30 min ago" in late


def test_urgent_and_routine_are_distinguishable_in_the_subject():
    """Email only buzzes if a rule says it should, and the rule keys on this."""
    assert sample(urgent=True).full_subject.startswith("[FDS URGENT]")
    assert sample(urgent=False).full_subject.startswith("[FDS]")
    assert "wk 7" in sample().full_subject


def test_the_html_is_self_contained():
    """No external images or stylesheets — half of them are blocked by default
    and the Sunday message has to be readable in ninety seconds on a phone."""
    html = DG.to_html(sample(), NOW)
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html
    assert "<img" not in html


def test_the_html_escapes_content():
    d = sample(sections=[("X", ["<script>alert(1)</script>"])])
    assert "<script>" not in DG.to_html(d)
    assert "&lt;script&gt;" in DG.to_html(d)


def test_every_message_states_that_nothing_was_submitted():
    assert "nothing here has been submitted" in DG.to_text(sample(), NOW)
    assert "§0.1" in DG.to_html(sample(), NOW)


def test_an_empty_digest_is_empty():
    """§6.4: no-action days send NOTHING. A digest that arrives daily saying
    'no action' stops being read by week four."""
    assert Digest(job="x", subject="nothing").is_empty
    assert not sample().is_empty


# ─── the job runner ─────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(LOG, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(LOG, "log_path", lambda season=2026: tmp_path / "log.jsonl")


def _build(claims):
    return lambda: (
        Digest(job="waivers", subject=f"{len(claims)} claims", week=6,
               sections=[("Claims", claims)]),
        {"claims": claims},
    )


def test_an_unchanged_answer_is_not_re_sent():
    """§6.4: every checkpoint after the weekly advisory speaks only when the
    answer MOVED. Nine emails a week is the fatigue that kills a digest."""
    n = MemoryNotifier()
    first = J.run("waivers", _build(["Ekeler"]), n, week=6, skip_gate=True)
    second = J.run("waivers", _build(["Ekeler"]), n, week=6, skip_gate=True)
    assert first.sent and first.detail == "sent"
    assert not second.sent and "unchanged" in second.detail
    assert len(n.sent) == 1


def test_a_changed_answer_is_sent():
    n = MemoryNotifier()
    J.run("waivers", _build(["Ekeler"]), n, week=6, skip_gate=True)
    moved = J.run("waivers", _build(["Ekeler", "Hall"]), n, week=6, skip_gate=True)
    assert moved.sent and len(n.sent) == 2


def test_the_weekly_advisory_is_unconditional():
    n = MemoryNotifier()
    for _ in range(2):
        J.run("lineup", _build(["same"]), n, week=6, unconditional=True, skip_gate=True)
    assert len(n.sent) == 2


def test_a_dry_run_does_not_advance_the_dedupe_state():
    """A dry run that recorded state would silently suppress the next real send."""
    dry = NullNotifier()
    J.run("waivers", _build(["Ekeler"]), dry, week=6, skip_gate=True)
    real = MemoryNotifier()
    res = J.run("waivers", _build(["Ekeler"]), real, week=6, skip_gate=True)
    assert res.sent and len(real.sent) == 1


def test_a_job_that_raises_reports_rather_than_disappearing():
    """An unattended job that dies silently is indistinguishable from one with
    nothing to say."""
    def boom():
        raise ValueError("projection source is empty")
    n = MemoryNotifier()
    res = J.run("waivers", boom, n, week=6, skip_gate=True)
    assert not res.ok
    assert n.sent and n.sent[0][0].urgent
    assert "projection source is empty" in n.sent[0][1]


def test_every_run_is_logged_with_its_inputs():
    """§10: 'or the season teaches you nothing.'"""
    J.run("waivers", _build(["Ekeler"]), MemoryNotifier(), week=6, skip_gate=True)
    recs = LOG.read()
    assert recs and recs[-1]["kind"] == "waivers"
    assert recs[-1]["week"] == 6 and "Claims" in recs[-1]["sections"]


def test_the_fingerprint_ignores_the_countdown_not_the_decision():
    """Hashing the rendered text would defeat the dedupe entirely, because the
    countdown changes every tick."""
    a = LOG.fingerprint({"claims": ["Ekeler"]})
    b = LOG.fingerprint({"claims": ["Ekeler"]})
    c = LOG.fingerprint({"claims": ["Hall"]})
    assert a == b != c


# ─── gate and heartbeat ─────────────────────────────────────────────────────
def test_missing_cookies_send_the_fix_instead_of_a_digest(monkeypatch):
    """Expired cookies are the number-one way an unattended monitor dies quietly."""
    monkeypatch.setattr("ff_agent.config.have_espn_credentials", lambda: False)
    monkeypatch.setattr("ff_agent.inseason.jobs.have_espn_credentials", lambda: False,
                        raising=False)
    n = MemoryNotifier()
    res = J.run("waivers", _build(["Ekeler"]), n, week=6)
    assert not res.ok
    assert n.sent and n.sent[0][0].urgent
    assert "Re-grab" in n.sent[0][1]


def test_the_heartbeat_says_nothing_when_all_is_well(monkeypatch):
    """A heartbeat that emails daily gets filtered — and then the one that
    matters is filtered too."""
    monkeypatch.setattr(J, "gate", lambda season=2026: None)
    monkeypatch.setattr("ff_agent.inseason.clock.assert_timezone",
                        lambda strict=True: "America/New_York")
    res = J.heartbeat(now=dt.datetime(2026, 10, 1, 7, 0, tzinfo=ET))
    assert res.ok and res.digest is None


def test_the_deadman_fires_when_nothing_has_succeeded(monkeypatch):
    """Silence must mean 'nothing to do', never 'it died'. This is the one case
    where the absence of news has to generate news."""
    monkeypatch.setattr(J, "gate", lambda season=2026: None)
    monkeypatch.setattr("ff_agent.inseason.clock.assert_timezone",
                        lambda strict=True: "America/New_York")
    LOG.record_sent("waivers", 2026, 6, "fp")
    stale = dt.datetime.now(tz=ET) + dt.timedelta(hours=48)
    res = J.heartbeat(now=stale)
    assert not res.ok
    assert any("has not succeeded" in a for _t, lines in res.digest.sections
               for a in lines)
