"""ESPN league state.

Two warnings from §6 drive the design here:

  * **Cookies expire.** Re-grab them draft morning and verify *before* you are on
    the clock. So every entry point verifies first and fails with an instruction,
    never a stack trace (§10).
  * **ESPN's live draft feed is flakier than Sleeper's.** Manual entry is the
    primary path; polling is a bonus. Nothing in this module writes to ESPN —
    per CLAUDE.md §0.1 there are no write paths at all, by construction.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from ff_agent.config import (
    ARTIFACTS_DIR, MissingCredentials, SEASON, espn_credentials,
)
from ff_agent.data.cache import cached


class ESPNAuthError(RuntimeError):
    """Cookies rejected or expired. Message must tell the human what to do."""


class ESPNUnavailable(RuntimeError):
    """ESPN reachable but did not return what we asked for."""


def _fmt_auth_error(exc: Exception, year: int) -> ESPNAuthError:
    return ESPNAuthError(
        f"ESPN rejected the request for {year} ({type(exc).__name__}: {exc}).\n"
        f"  Almost always this means espn_s2 / SWID have EXPIRED.\n"
        f"  Fix: re-grab both cookies (SETUP.md §3) and paste them into .env.\n"
        f"  Keep the % characters in ESPN_S2 and the braces on ESPN_SWID.\n"
        f"  §6: re-grab these the MORNING OF THE DRAFT and verify before you are\n"
        f"  on the clock."
    )


def get_league(year: int = SEASON):
    """Construct an espn_api League, translating failures into human fixes."""
    from espn_api.football import League

    creds = espn_credentials()
    try:
        return League(
            league_id=int(creds["league_id"]),
            year=year,
            espn_s2=creds["espn_s2"],
            swid=creds["swid"],
        )
    except Exception as exc:  # espn_api raises bare Exceptions for 401s
        msg = str(exc).lower()
        if "401" in msg or "unauthor" in msg or "private" in msg or "cookie" in msg:
            raise _fmt_auth_error(exc, year) from exc
        if "404" in msg or "not found" in msg:
            raise ESPNUnavailable(
                f"ESPN has no league {creds['league_id']} for {year}.\n"
                f"  If {year} has not been created yet on ESPN, use a prior year.\n"
                f"  Also check ESPN_LEAGUE_ID in .env."
            ) from exc
        raise


def verify_credentials(year: int = SEASON) -> dict[str, Any]:
    """Cheap pre-flight. Run this at the start of every session and draft morning."""
    try:
        lg = get_league(year)
    except MissingCredentials as e:
        return {"ok": False, "stage": "credentials", "detail": str(e)}
    except (ESPNAuthError, ESPNUnavailable) as e:
        return {"ok": False, "stage": "auth", "detail": str(e)}

    teams = getattr(lg, "teams", []) or []
    return {
        "ok": True,
        "stage": "connected",
        "year": year,
        "league_name": getattr(getattr(lg, "settings", None), "name", None),
        "n_teams": len(teams),
        "team_names": [getattr(t, "team_name", "?") for t in teams],
        "detail": "cookies valid",
    }


# ─── Extraction helpers ──────────────────────────────────────────────────────
PLAYER_SCHEMA = {
    "espn_id": pl.Utf8,
    "name": pl.Utf8,
    "position": pl.Utf8,
    "team": pl.Utf8,
    "injury_status": pl.Utf8,
    "pos_rank": pl.Int64,
    "percent_owned": pl.Float64,
    "percent_started": pl.Float64,
    "source": pl.Utf8,
}
"""Explicit schema. ESPN returns [] rather than null for several fields during
the preseason (posRank, acquisitionType), which makes polars' type inference
fail on the first all-empty column. Never rely on inference here."""


def _scalar(v, cast=None):
    """ESPN gives [] for 'no value' on some fields. Flatten to a scalar/None."""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v is None or v == "":
        return None
    if cast is not None:
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None
    return v


def _player_rows(players, source: str) -> list[dict]:
    rows = []
    for p in players:
        rows.append({
            "espn_id": str(getattr(p, "playerId", "") or ""),
            "name": _scalar(getattr(p, "name", None)),
            "position": _scalar(getattr(p, "position", None)),
            "team": _scalar(getattr(p, "proTeam", None)),
            "injury_status": _scalar(getattr(p, "injuryStatus", None)),
            "pos_rank": _scalar(getattr(p, "posRank", None), int),
            "percent_owned": _scalar(getattr(p, "percent_owned", None), float),
            "percent_started": _scalar(getattr(p, "percent_started", None), float),
            "source": source,
        })
    return rows


def draftable_players(year: int = SEASON, size: int = 1200, **kw) -> pl.DataFrame:
    """The full draftable pool: every rostered player plus deep free agency.

    This is half of Milestone 1's test set — the population that actually
    matters on draft day, and where rookies break a crosswalk.
    """
    def fetch() -> pl.DataFrame:
        lg = get_league(year)
        rows: list[dict] = []
        for t in getattr(lg, "teams", []) or []:
            rows += _player_rows(getattr(t, "roster", []) or [], "roster")
        try:
            rows += _player_rows(lg.free_agents(size=size), "free_agent")
        except Exception as exc:
            raise ESPNUnavailable(f"free_agents({size}) failed: {exc}") from exc
        if not rows:
            raise ESPNUnavailable(f"No players returned for {year}.")
        return pl.DataFrame(rows, schema=PLAYER_SCHEMA).unique(subset=["espn_id"], keep="first")

    return cached("espn_players", fetch, season=year, source="espn", **kw)


ROSTER_SCHEMA = {
    "team_id": pl.Int64, "fantasy_team": pl.Utf8, "manager": pl.Utf8,
    "espn_id": pl.Utf8, "name": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8,
}


def rosters(year: int, **kw) -> pl.DataFrame:
    """Final rosters for a season — the historical half of the test set."""
    def fetch() -> pl.DataFrame:
        lg = get_league(year)
        rows = []
        for t in getattr(lg, "teams", []) or []:
            owners = getattr(t, "owners", None) or []
            owner = ", ".join(
                (o.get("firstName", "") + " " + o.get("lastName", "")).strip()
                if isinstance(o, dict) else str(o)
                for o in owners
            )
            for p in getattr(t, "roster", []) or []:
                rows.append({
                    "team_id": getattr(t, "team_id", None),
                    "fantasy_team": getattr(t, "team_name", None),
                    "manager": owner or None,
                    "espn_id": str(getattr(p, "playerId", "") or ""),
                    "name": _scalar(getattr(p, "name", None)),
                    "position": _scalar(getattr(p, "position", None)),
                    "team": _scalar(getattr(p, "proTeam", None)),
                })
        if not rows:
            raise ESPNUnavailable(
                f"No rosters returned for {year}. If {year} is the upcoming "
                f"season, this is expected — the draft has not happened yet."
            )
        return pl.DataFrame(rows, schema=ROSTER_SCHEMA)

    return cached("espn_rosters", fetch, season=year, source="espn", **kw)


def draft_results(year: int, **kw) -> pl.DataFrame:
    """Past draft picks — feeds the opponent model (§7.5, Milestone 5)."""
    def fetch() -> pl.DataFrame:
        lg = get_league(year)
        picks = getattr(lg, "draft", None) or []
        rows = []
        for pk in picks:
            t = getattr(pk, "team", None)
            rows.append({
                "round": getattr(pk, "round_num", None),
                "pick": getattr(pk, "round_pick", None),
                "fantasy_team": getattr(t, "team_name", None) if t else None,
                "team_id": getattr(t, "team_id", None) if t else None,
                "espn_id": str(getattr(pk, "playerId", "") or ""),
                "name": getattr(pk, "playerName", None),
                "keeper": getattr(pk, "keeper_status", None),
                "bid_amount": getattr(pk, "bid_amount", None),
            })
        if not rows:
            raise ESPNUnavailable(f"No draft found for {year}.")
        return pl.DataFrame(rows)

    return cached("espn_draft", fetch, season=year, source="espn", **kw)


def waiver_order(year: int = SEASON, **kw) -> pl.DataFrame:
    """Every team's rolling waiver priority — §9.3 needs all nine, not just mine."""
    def fetch() -> pl.DataFrame:
        lg = get_league(year)
        rows = []
        for t in getattr(lg, "teams", []) or []:
            rows.append({
                "team_id": getattr(t, "team_id", None),
                "fantasy_team": getattr(t, "team_name", None),
                "waiver_rank": getattr(t, "waiver_rank", None),
                "standing": getattr(t, "standing", None),
                "wins": getattr(t, "wins", None),
                "losses": getattr(t, "losses", None),
            })
        if not rows:
            raise ESPNUnavailable(f"No teams returned for {year}.")
        return pl.DataFrame(rows)

    return cached("espn_waivers", fetch, season=year, source="espn", **kw)


# ─── Settings dump — answers the CLAUDE.md OPEN questions ────────────────────
SETTINGS_QUESTIONS = """
This dump is how we close the OPEN items in CLAUDE.md §0.5 from the source of
truth instead of from memory:
  * playoff weeks and playoff team count  (§1 marked CONFIRM)
  * number of first-round byes            (§1 marked CONFIRM)
  * seeding on raw wins vs win percentage (§2.5)
  * the D/ST points-allowed bucket values (§1 lists 18-21 AND 22-27 both at 0)
  * keeper vs full redraft
"""


def settings_dump(year: int = SEASON, write: bool = True) -> dict[str, Any]:
    """Pull raw league settings and persist them for inspection."""
    lg = get_league(year)
    s = getattr(lg, "settings", None)
    if s is None:
        raise ESPNUnavailable(f"League {year} exposed no settings object.")

    out: dict[str, Any] = {"year": year, "_questions": SETTINGS_QUESTIONS.strip()}
    for attr in dir(s):
        if attr.startswith("_"):
            continue
        try:
            v = getattr(s, attr)
        except Exception:
            continue
        if callable(v):
            continue
        try:
            json.dumps(v)
            out[attr] = v
        except (TypeError, ValueError):
            out[attr] = repr(v)

    raw = getattr(lg, "league_json", None) or getattr(lg, "_json", None)
    if isinstance(raw, dict) and "settings" in raw:
        out["_raw_settings"] = raw["settings"]

    if write:
        path = ARTIFACTS_DIR / f"espn_settings_{year}.json"
        path.write_text(json.dumps(out, indent=2, default=str))
        out["_written_to"] = str(path)
    return out
