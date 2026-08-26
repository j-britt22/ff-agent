"""§10: "Log every recommendation with timestamp and inputs, or the season
teaches you nothing."

One JSON object per line, appended, never rewritten — the same shape
``live/log.py`` uses for the draft, so ``audit.py`` can replay either.

The log is also what makes §6.4's dedupe possible: a state file per (job, week)
records what was already said, so re-running Tuesday's job twice produces one
email rather than two.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ff_agent.config import LOGS_DIR, SEASON

STATE_DIR = LOGS_DIR.parent / "state"


def log_path(season: int = SEASON) -> Path:
    return LOGS_DIR / f"inseason_{season}.jsonl"


def write(kind: str, season: int = SEASON, **payload) -> dict:
    """Append one record. Returns it, so callers can log and use in one step."""
    rec = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "kind": kind,
        **payload,
    }
    p = log_path(season)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def read(season: int = SEASON) -> list[dict]:
    p = log_path(season)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ─── Dedupe state ────────────────────────────────────────────────────────────
def _state_path(job: str, season: int, week: int | None) -> Path:
    wk = "pre" if week is None else f"wk{week}"
    return STATE_DIR / f"{job}_{season}_{wk}.json"


def fingerprint(payload) -> str:
    """A stable hash of what we are about to say.

    Not of the whole digest: timestamps and countdowns change every tick, so
    hashing the rendered text would defeat the dedupe entirely. Callers pass the
    DECISION content.
    """
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def already_sent(job: str, season: int, week: int | None, fp: str) -> bool:
    """§6.4: only the weekly advisory is unconditional. Everything after it
    speaks only when the answer moved."""
    p = _state_path(job, season, week)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("fingerprint") == fp
    except json.JSONDecodeError:
        return False


def record_sent(job: str, season: int, week: int | None, fp: str, **extra) -> None:
    p = _state_path(job, season, week)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "job": job, "season": season, "week": week, "fingerprint": fp,
        "sent_at": datetime.now(tz=timezone.utc).isoformat(), **extra,
    }, indent=2, default=str))


def last_success(job: str, season: int = SEASON) -> datetime | None:
    """When this job last completed. The deadman reads all of these."""
    latest = None
    for p in sorted(STATE_DIR.glob(f"{job}_{season}_*.json")):
        try:
            ts = json.loads(p.read_text()).get("sent_at")
        except json.JSONDecodeError:
            continue
        if ts:
            when = datetime.fromisoformat(ts)
            latest = when if latest is None else max(latest, when)
    return latest
