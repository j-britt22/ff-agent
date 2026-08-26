"""SMTP. The only backend shipped (§1's Channel decision).

Configuration is environment-only and never touches ``.env``'s ESPN block, so a
leaked mail password cannot also be a leaked league login.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

from ff_agent.inseason.notify.base import Digest, Notifier, SendResult

REQUIRED = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "DIGEST_TO")


class EmailConfigError(RuntimeError):
    """Mail is not configured. A clear fix, never a stack trace (§10)."""


def config() -> dict[str, str]:
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
    vals["DIGEST_FROM"] = os.environ.get("DIGEST_FROM", vals["SMTP_USER"]).strip()
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
