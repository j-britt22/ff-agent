"""SMTP. The only backend shipped (§1's Channel decision).

Configuration lives in ``.env.notify``, never in ``.env``'s ESPN block, so a
leaked mail password cannot also be a leaked league login.

**This file has to load its own env file.** ``ff_agent/config.py`` loads ``.env``
on first use (``_ensure_env``), but that loader only ever reads ``.env`` — it has
no reason to know ``.env.notify`` exists, and mixing the two back together would
undo the whole point of keeping them apart. Docker Compose's ``env_file:``
directive populates the container's environment before Python ever runs, which
made this invisible in the one place it was tested. Running the CLI directly
(``uv run python -m ff_agent.cli monitor ...``, which is also the fastest way to
test any of this before standing up the container) has no Compose in the loop,
so without this loader ``.env.notify`` sits on disk, fully filled in, and does
nothing.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

from dotenv import load_dotenv

from ff_agent.config import ROOT
from ff_agent.inseason.notify.base import Digest, Notifier, SendResult

REQUIRED = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "DIGEST_TO")

_ENV_LOADED = False


def _ensure_env() -> None:
    """Load ``.env.notify`` once. Mirrors ``ff_agent.config._ensure_env`` exactly,
    against the other file, on purpose — one mechanism, two separate files."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(ROOT / ".env.notify")
        _ENV_LOADED = True


class EmailConfigError(RuntimeError):
    """Mail is not configured. A clear fix, never a stack trace (§10)."""


def config() -> dict[str, str]:
    _ensure_env()
    vals = {k: (os.environ.get(k) or "").strip() for k in REQUIRED}
    missing = [k for k, v in vals.items() if not v]
    if missing:
        raise EmailConfigError(
            "email is not configured: " + ", ".join(missing) + ".\n"
            "  Set them in .env.notify (see SETUP_MONITOR.md). With Gmail this\n"
            "  is an APP PASSWORD, not the account password.\n"
            "  Keep them out of the ESPN .env so one leak is not two."
        )
    vals["SMTP_PORT"] = os.environ.get("SMTP_PORT", "587").strip()
    # ``.get(key, default)`` only falls back when the key is ABSENT — but the
    # example file ships "DIGEST_FROM=" (blank, marked optional), and dotenv
    # sets a blank line to "" rather than leaving it unset. So a literal copy of
    # the template sent mail with an empty From: header until this was written
    # as `or`, which treats blank the same as absent.
    vals["DIGEST_FROM"] = (os.environ.get("DIGEST_FROM") or vals["SMTP_USER"]).strip()
    return vals


class EmailNotifier(Notifier):
    name = "email"

    def __init__(self, cfg: dict[str, str] | None = None):
        self.cfg = cfg or config()

    def _send(self, digest: Digest, text: str, html: str | None) -> SendResult:
        msg = EmailMessage()
        msg["Subject"] = digest.full_subject
        msg["From"] = self.cfg["DIGEST_FROM"]
        msg["To"] = self.cfg["DIGEST_TO"]
        if digest.urgent:
            # Some clients surface this; the Gmail filter in SETUP_MONITOR.md is
            # what actually makes the phone buzz.
            msg["X-Priority"] = "1"
            msg["Importance"] = "high"
        msg.set_content(text)
        if html:
            msg.add_alternative(html, subtype="html")

        port = int(self.cfg["SMTP_PORT"])
        ctx = ssl.create_default_context()
        try:
            if port == 465:
                with smtplib.SMTP_SSL(self.cfg["SMTP_HOST"], port, context=ctx) as s:
                    s.login(self.cfg["SMTP_USER"], self.cfg["SMTP_PASSWORD"])
                    s.send_message(msg)
            else:
                with smtplib.SMTP(self.cfg["SMTP_HOST"], port) as s:
                    s.starttls(context=ctx)
                    s.login(self.cfg["SMTP_USER"], self.cfg["SMTP_PASSWORD"])
                    s.send_message(msg)
        except Exception as exc:
            return SendResult(False, self.name, f"{type(exc).__name__}: {exc}")
        return SendResult(True, self.name, f"sent to {self.cfg['DIGEST_TO']}")

    def preflight(self) -> SendResult:
        """Prove SMTP accepts a connection without sending. Part of entrypoint."""
        port = int(self.cfg["SMTP_PORT"])
        try:
            if port == 465:
                with smtplib.SMTP_SSL(self.cfg["SMTP_HOST"], port,
                                      context=ssl.create_default_context()) as s:
                    s.login(self.cfg["SMTP_USER"], self.cfg["SMTP_PASSWORD"])
            else:
                with smtplib.SMTP(self.cfg["SMTP_HOST"], port) as s:
                    s.starttls(context=ssl.create_default_context())
                    s.login(self.cfg["SMTP_USER"], self.cfg["SMTP_PASSWORD"])
        except Exception as exc:
            return SendResult(False, self.name, f"{type(exc).__name__}: {exc}")
        return SendResult(True, self.name, "SMTP accepted the login")
