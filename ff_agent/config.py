"""Project-wide constants and credential loading.

Everything here is settled fact from FANTASY_SPEC.md §1 and CLAUDE.md §0.5.
Scoring rules deliberately live elsewhere — they arrive in Milestone 2.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
OVERRIDES_DIR = ROOT / "overrides"
ARTIFACTS_DIR = ROOT / "artifacts"
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"

for _d in (CACHE_DIR, OVERRIDES_DIR, ARTIFACTS_DIR, LOGS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Seasons ─────────────────────────────────────────────────────────────────
SEASON = 2026
"""The season we are drafting for."""

HISTORY_START = 2016
"""Next Gen Stats boundary — verified 2026-08-19 that ngs_* covers exactly
2016-2025. Earlier seasons exist for pbp/player_stats but lose NGS entirely."""

HISTORY_END = 2025
HISTORY_SEASONS: list[int] = list(range(HISTORY_START, HISTORY_END + 1))
LAST_SEASON = HISTORY_END

CONTAMINATED_SEASONS = frozenset({2020})
"""COVID: no preseason, empty stadiums, mass absences. Keep, but flag."""

# ─── League structure (§1) ───────────────────────────────────────────────────
LEAGUE_NAME = "First Down Syndrome"
PLATFORM = "espn"
N_TEAMS = 9
DRAFT_TYPE = "snake"

REGULAR_SEASON_WEEKS = tuple(range(1, 15))  # 1-14
MY_BYE_WEEKS = frozenset({5, 14})
"""My *fantasy* byes. Nothing on my roster matters these weeks."""

FREE_BYE_WEEKS = MY_BYE_WEEKS
"""§2.1: a player whose NFL bye lands here costs me nothing. First-class
board term, not a tiebreaker."""

MY_GAME_WEEKS = tuple(w for w in REGULAR_SEASON_WEEKS if w not in MY_BYE_WEEKS)
assert len(MY_GAME_WEEKS) == 12

# Marked CONFIRM in §1 — see CLAUDE.md OPEN list. Do not treat as verified.
PLAYOFF_WEEKS_ASSUMED = (15, 16, 17)
PLAYOFF_TEAMS_ASSUMED = 6
FIRST_ROUND_BYES_ASSUMED = 2

STARTER_SLOTS = {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}
ROSTER_TOTAL = 17
BENCH_SLOTS = 7
IR_SLOTS = 1
POSITION_MAXIMA = {"QB": 4, "RB": 8, "WR": 8, "TE": 3, "DST": 3, "K": 3}

FANTASY_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "K", "DST"})
SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})

# ─── Opponents (§2.3) ────────────────────────────────────────────────────────
# weight 2 = faced twice = 8 of my 12 games. Used by the opponent model (M5)
# and in-season scouting; kept here so there is one source of truth.
OPPONENTS: tuple[dict, ...] = (
    {"team": "Hodor's Hodors",        "manager": "Camden Sims",    "weeks": (1, 10), "weight": 2},
    {"team": "Personality Hires",     "manager": "Kylie Leahy",    "weeks": (2, 11), "weight": 2},
    {"team": "Clearing the Fields",   "manager": "R. Sharrett",    "weeks": (3, 12), "weight": 2},
    {"team": "Gibbs Me My Money",     "manager": "Matthew Benca",  "weeks": (4, 13), "weight": 2},
    {"team": "A Chane Reaction",      "manager": "Jeff Boyd",      "weeks": (6,),    "weight": 1},
    {"team": "Unsolicited Dak Pics",  "manager": "Josh Boyd",      "weeks": (7,),    "weight": 1},
    {"team": "Nothing Beats a JJett 2 H...", "manager": "Hanna Rogo", "weeks": (8,), "weight": 1},
    {"team": "leah's team",           "manager": "leah gottlieb",  "weeks": (9,),    "weight": 1},
)
DOUBLE_UP_MANAGERS = tuple(o["manager"] for o in OPPONENTS if o["weight"] == 2)

assert sum(len(o["weeks"]) for o in OPPONENTS) == 12, "schedule must cover 12 games"


# ─── ESPN credentials ────────────────────────────────────────────────────────
class MissingCredentials(RuntimeError):
    """Raised with a human-readable fix, never a bare stack trace (§10)."""


_ENV_LOADED = False


def _ensure_env() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(ROOT / ".env")
        _ENV_LOADED = True


def espn_credentials() -> dict[str, str]:
    """Return ESPN creds, or raise with instructions.

    Cookies expire — §6 warns to re-grab them draft morning. This never
    logs or echoes the values.
    """
    _ensure_env()
    vals = {
        "league_id": (os.getenv("ESPN_LEAGUE_ID") or "").strip(),
        "espn_s2": (os.getenv("ESPN_S2") or "").strip(),
        "swid": (os.getenv("ESPN_SWID") or "").strip(),
    }
    missing = [k for k, v in vals.items() if not v]
    if missing:
        raise MissingCredentials(
            "ESPN credentials are not set: " + ", ".join(missing) + ".\n"
            f"  Fix: open {ROOT / '.env'} and paste your values in.\n"
            "  See SETUP.md §3. Keep the % characters in ESPN_S2 and the\n"
            "  curly braces on ESPN_SWID. Do not commit this file."
        )
    if not vals["league_id"].isdigit():
        raise MissingCredentials(
            f"ESPN_LEAGUE_ID must be digits only, got {vals['league_id']!r}.\n"
            "  It is the leagueId= number in your league URL."
        )
    if not (vals["swid"].startswith("{") and vals["swid"].endswith("}")):
        raise MissingCredentials(
            "ESPN_SWID must keep its curly braces, e.g. {1A2B-...}.\n"
            "  Re-copy it from the cookie exactly as shown."
        )
    return vals


def have_espn_credentials() -> bool:
    """Non-raising probe, for skipping tests cleanly."""
    try:
        espn_credentials()
        return True
    except MissingCredentials:
        return False
