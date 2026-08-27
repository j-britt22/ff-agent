"""The cron entry points: preflight -> build -> decide -> log -> notify.

Every job runs the same four steps and nothing else touches the outside world.
Three of the four are guardrails rather than work, because an unattended agent
breaks differently from one with a human at the keyboard:

  * **Expired cookies are the number-one operational failure.** An agent whose
    cookies expire in week 3 silently stops working until November. So every job
    pre-flights authentication, and on failure sends the re-grab instructions
    INSTEAD of a digest rather than alongside one.
  * **Stale data blocks the send.** §10: an optimizer silently running week-3 data
    in week 8 is worse than no optimizer.
  * **Silence must mean "nothing to do", never "it died."** No-action days send
    nothing, the heartbeat speaks only when something is wrong, and the deadman
    fires if no job has succeeded in 36 hours — the one case where the absence of
    news has to generate news.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ff_agent.config import SEASON
from ff_agent.inseason import clock as CK
from ff_agent.inseason import digest as DG
from ff_agent.inseason import log as LOG
from ff_agent.inseason.notify.base import Digest, Notifier, NullNotifier, SendResult

JOBS = ("preflight", "refresh", "waivers", "freeagents", "tick", "injuries",
        "trades", "week14", "heartbeat", "audit")

DEADMAN = dt.timedelta(hours=36)
"""If nothing has succeeded in this long, that IS something wrong."""


@dataclass
class JobResult:
    job: str
    ok: bool
    sent: bool = False
    digest: Digest | None = None
    detail: str = ""
    send: SendResult | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "job": self.job, "ok": self.ok, "sent": self.sent,
            "subject": None if not self.digest else self.digest.full_subject,
            "detail": self.detail,
            "latency_s": None if not self.send else self.send.latency_s,
            "notes": self.notes,
        }


# ─── Pre-flight ──────────────────────────────────────────────────────────────
def preflight(strict: bool = True, check_smtp: bool = True) -> JobResult:
    """Run at container start. Emails and exits non-zero on any failure.

    A container that comes up broken says so immediately rather than at 08:00 on
    Tuesday.
    """
    problems: list[str] = []
    notes: list[str] = []

    try:
        notes.append(f"timezone {CK.assert_timezone(strict=strict)}")
    except CK.ClockError as e:
        problems.append(str(e))

    from ff_agent.config import have_espn_credentials
    if not have_espn_credentials():
        problems.append(
            "ESPN credentials are missing. Every job reads the league; none can "
            "run. See SETUP.md §3."
        )
    else:
        try:
            from ff_agent.data.espn import verify_credentials
            v = verify_credentials()
            if not v.get("ok"):
                problems.append(v.get("detail", "ESPN rejected the credentials"))
            else:
                notes.append(f"ESPN ok — {v.get('league_name')}, {v.get('n_teams')} teams")
        except Exception as exc:
            problems.append(f"ESPN check failed: {type(exc).__name__}: {exc}")

    if check_smtp:
        try:
            from ff_agent.inseason.notify.email import EmailNotifier
            r = EmailNotifier().preflight()
            (notes if r.ok else problems).append(f"SMTP: {r.detail}")
        except Exception as exc:
            problems.append(f"email not configured: {exc}")

    ok = not problems
    d = None
    if not ok:
        d = Digest(
            job="preflight", subject="pre-flight FAILED — the monitor is not running",
            urgent=True,
            headline="The container came up but cannot do its job. Nothing else "
                     "will run until this is fixed.",
            sections=[("Problems", problems)], notes=notes,
        )
    return JobResult("preflight", ok, digest=d,
                     detail="; ".join(problems) or "all checks passed", notes=notes)


# ─── Cookie / staleness gate shared by every real job ────────────────────────
def gate(season: int = SEASON) -> JobResult | None:
    """Returns a JobResult to send INSTEAD of a digest, or None to proceed."""
    from ff_agent.config import have_espn_credentials
    if not have_espn_credentials():
        return JobResult(
            "gate", False,
            digest=Digest(
                job="gate", subject="ESPN cookies are gone — the monitor is blind",
                urgent=True,
                headline="Every job reads the league and none can run.",
                sections=[("Fix", [
                    "Re-grab espn_s2 and SWID from the browser (SETUP.md §3).",
                    "Keep the % characters in ESPN_S2 and the braces on ESPN_SWID.",
                    "Then: docker compose restart ff-monitor",
                ])]),
            detail="missing credentials")
    try:
        from ff_agent.data.espn import verify_credentials
        v = verify_credentials(season)
    except Exception as exc:
        return JobResult("gate", False, detail=f"{type(exc).__name__}: {exc}")
    if not v.get("ok"):
        return JobResult(
            "gate", False,
            digest=Digest(
                job="gate", subject="ESPN rejected the cookies — they have expired",
                urgent=True,
                headline="This is the number-one way an unattended monitor dies "
                         "quietly. It has not died quietly.",
                sections=[("What ESPN said", [v.get("detail", "")])]),
            detail="auth failed")
    return None


# ─── Heartbeat / deadman ─────────────────────────────────────────────────────
def heartbeat(season: int = SEASON, now: dt.datetime | None = None) -> JobResult:
    """Speaks ONLY when something is wrong. §6.4.

    A heartbeat that emails daily to say everything is fine is a heartbeat that
    gets filtered, and then the one that matters is filtered too.
    """
    now = now or CK.now_et()
    problems: list[str] = []

    g = gate(season)
    if g is not None:
        problems.append(g.detail or "ESPN unreachable")

    try:
        CK.assert_timezone()
    except CK.ClockError as e:
        problems.append(str(e).splitlines()[0])

    stalest = None
    for job in ("waivers", "tick", "refresh"):
        last = LOG.last_success(job, season)
        if last is None:
            continue
        age = now - last.astimezone(now.tzinfo)
        stalest = age if stalest is None else max(stalest, age)
        if age > DEADMAN:
            problems.append(
                f"job {job!r} has not succeeded in {age.total_seconds() / 3600:.0f} "
                f"hours. Silence should mean 'nothing to do', not 'it died'."
            )
    if not problems:
        return JobResult("heartbeat", True, detail="all quiet — nothing sent")
    return JobResult(
        "heartbeat", False,
        digest=Digest(job="heartbeat", subject="the monitor needs attention",
                      urgent=True, sections=[("Problems", problems)]),
        detail="; ".join(problems))


# ─── Running one job ─────────────────────────────────────────────────────────
def run(
    job: str,
    build_digest,
    notifier: Notifier | None = None,
    season: int = SEASON,
    week: int | None = None,
    unconditional: bool = False,
    now: dt.datetime | None = None,
    skip_gate: bool = False,
) -> JobResult:
    """The four steps, in order, with the guardrails between them.

    ``build_digest`` is a callable returning ``(Digest, fingerprint_payload)``.
    ``unconditional`` is true only for the weekly advisory — everything else
    speaks only when the answer moved (§6.4).
    """
    notifier = notifier or NullNotifier()
    now = now or CK.now_et()

    if not skip_gate:
        blocked = gate(season)
        if blocked is not None:
            if blocked.digest is not None:
                _deliver(blocked, notifier, now)
            return blocked

    try:
        d, fingerprint_payload = build_digest()
    except Exception as exc:
        res = JobResult(job, False, detail=f"{type(exc).__name__}: {exc}")
        res.digest = Digest(
            job=job, subject=f"{job} failed to build", urgent=True, week=week,
            headline="The job raised rather than producing a digest. No "
                     "recommendation was made.",
            sections=[("Error", [f"{type(exc).__name__}: {exc}"])])
        _deliver(res, notifier, now)
        return res

    # The DECISION payload is logged beside the rendered text, because §10's
    # "log every recommendation with its inputs" is only half useful if nothing
    # can read it back. audit.py scores against this, and the Wednesday
    # free-agent sweep checks last night's predictions against it — both need
    # ids, not prose.
    fp = LOG.fingerprint(fingerprint_payload)
    LOG.write(job, season=season, week=week, subject=d.full_subject,
              sections={t: lines for t, lines in d.sections},
              alarms=d.alarms, notes=d.notes, decision=fingerprint_payload,
              fingerprint=fp)
    if d.is_empty and not unconditional:
        return JobResult(job, True, detail="nothing to say — not sent")
    if not unconditional and LOG.already_sent(job, season, week, fp):
        return JobResult(job, True, detail="unchanged since the last send")

    res = JobResult(job, True, digest=d)
    _deliver(res, notifier, now)
    if res.sent:
        LOG.record_sent(job, season, week, fp, subject=d.full_subject)
        res.detail = res.detail or "sent"
    elif notifier.dry_run:
        res.detail = res.detail or "rendered (dry run — state not advanced)"
    return res


def _deliver(res: JobResult, notifier: Notifier, now: dt.datetime) -> None:
    if res.digest is None:
        return
    text = DG.to_text(res.digest, now)
    html = DG.to_html(res.digest, now)
    res.send = notifier.send(res.digest, text, html)
    res.sent = res.send.ok and not notifier.dry_run
    if not res.send.ok:
        res.detail = f"{res.detail} | send failed: {res.send.detail}".strip(" |")
