"""The Digest, the Notifier interface, and the one guardrail that must not fail.

**An outbound email is the only path in this project by which a credential could
leave the machine.** Every other component reads. So the secret scan lives here,
runs on every send, and is a hard failure rather than a warning — a digest that
leaks the ESPN cookies is worse than no digest at all, and unlike every other
failure mode in this package it cannot be undone once it has happened.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime

ROUTINE_PREFIX = "[FDS]"
URGENT_PREFIX = "[FDS URGENT]"
"""§6.3's convention. A Gmail filter on the urgent prefix, plus high-priority-only
phone notifications, is what turns email into something that buzzes. Two minutes
at setup, documented in SETUP_MONITOR.md."""


class SecretLeak(RuntimeError):
    """A rendered digest contains something that must never leave the box."""


@dataclass
class Digest:
    """One message. Rendered to text and HTML by ``digest.py``."""

    job: str
    subject: str
    week: int | None = None
    urgent: bool = False
    headline: str = ""
    sections: list[tuple[str, list[str]]] = field(default_factory=list)
    """(title, lines). Lines are plain sentences — the renderer adds structure."""
    alarms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    deadline: datetime | None = None
    created_at: datetime | None = None

    @property
    def full_subject(self) -> str:
        prefix = URGENT_PREFIX if self.urgent else ROUTINE_PREFIX
        wk = f" wk {self.week}" if self.week else ""
        return f"{prefix}{wk} — {self.subject}"

    @property
    def is_empty(self) -> bool:
        """No-action days send NOTHING (§6.4).

        A digest that arrives daily saying "no action" stops being read by week
        four, at which point the tool has failed regardless of how good its
        numbers are.
        """
        return not (self.sections or self.alarms or self.headline)


@dataclass
class SendResult:
    ok: bool
    channel: str
    detail: str = ""
    latency_s: float | None = None
    """Logged on every send. §6.3 promised the Sunday email's timeliness would be
    measurable rather than argued about; this is the measurement."""


# ─── The guardrail ───────────────────────────────────────────────────────────
SECRET_ENV = ("ESPN_S2", "ESPN_SWID", "SMTP_PASSWORD", "ANTHROPIC_API_KEY")
MIN_SECRET_LEN = 8
"""Below this a value is too short to be a credential and matching it would be
noise — a two-character league id would flag every digest."""

_SWID_RE = re.compile(r"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
                      r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}")


def assert_no_secrets(*payloads: str) -> None:
    """Refuse to send anything containing a live credential.

    Checks two ways, because either alone is insufficient: the exact values from
    the environment (catches an accidental interpolation) and the SWID's literal
    shape (catches a value that reached the digest from somewhere other than the
    environment, such as a cached ESPN payload echoed into a debug line).
    """
    blob = "\n".join(p for p in payloads if p)
    if not blob:
        return
    for name in SECRET_ENV:
        val = (os.environ.get(name) or "").strip()
        if len(val) >= MIN_SECRET_LEN and val in blob:
            raise SecretLeak(
                f"the rendered digest contains the value of {name}. Refusing to "
                f"send.\n"
                f"  An outbound email is the only path in this project by which a "
                f"credential could leave the machine, which is why this is fatal "
                f"rather than a warning."
            )
    if _SWID_RE.search(blob):
        raise SecretLeak(
            "the rendered digest contains something shaped exactly like an ESPN "
            "SWID cookie. Refusing to send."
        )


# ─── The interface ───────────────────────────────────────────────────────────
class Notifier:
    """Send a digest. Subclasses implement ``_send``.

    ``dry_run`` separates "the notifier accepted it" from "it actually went out",
    which matters because the dedupe state must only advance on a REAL send. A
    dry run that recorded state would silently suppress the next real one.
    """

    name = "notifier"
    dry_run = False

    def send(self, digest: Digest, text: str, html: str | None = None) -> SendResult:
        if digest.is_empty:
            return SendResult(True, self.name, "nothing to say — not sent")
        assert_no_secrets(digest.full_subject, text, html or "")
        import time
        t0 = time.time()
        res = self._send(digest, text, html)
        return SendResult(res.ok, res.channel, res.detail, round(time.time() - t0, 3))

    def _send(self, digest: Digest, text: str, html: str | None) -> SendResult:
        raise NotImplementedError


class NullNotifier(Notifier):
    """Renders and validates, sends nothing. What ``--dry-run`` uses.

    The first thing anybody does with a tool that emails is run it once without
    emailing, so that path is a first-class object rather than an if-statement.
    """

    name = "null"
    dry_run = True

    def __init__(self):
        self.sent: list[tuple[Digest, str, str | None]] = []

    def _send(self, digest, text, html):
        self.sent.append((digest, text, html))
        return SendResult(True, self.name, "dry run — not sent")


class MemoryNotifier(Notifier):
    """Behaves like a real backend but keeps the mail. For tests.

    Distinct from NullNotifier precisely because it is NOT a dry run: it advances
    the dedupe state, which is the behaviour under test.
    """

    name = "memory"
    dry_run = False

    def __init__(self):
        self.sent: list[tuple[Digest, str, str | None]] = []

    def _send(self, digest, text, html):
        self.sent.append((digest, text, html))
        return SendResult(True, self.name, "captured")
