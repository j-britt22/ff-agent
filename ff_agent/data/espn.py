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
    normalize_team_name,
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


def team_names_by_manager(year: int = SEASON, **kw) -> pl.DataFrame:
    """Manager -> the team name ESPN shows for them RIGHT NOW.

    Team names are mutable and managers are not. §1 records "A Chane Reaction"
    for Jeff Boyd; mid-2026 that team renamed itself "TBD", and every hardcoded
    name became a join that would fail the moment the schedule cache refreshed —
    silently in some places, as a KeyError inside the season simulator in
    others. Managers are the stable key, which is also how §2.3 thinks about the
    league ("weight the four double-up MANAGERS").
    """
    def fetch() -> pl.DataFrame:
        from ff_agent.opponents.history import _owner_names, canonical_manager

        lg = get_league(year)
        rows = []
        for t in lg.teams:
            names = _owner_names(t)
            rows.append({
                "season": year,
                "team_id": getattr(t, "team_id", None),
                "team": normalize_team_name(t.team_name),
                "manager": canonical_manager(names[0]) if names else None,
            })
        if not rows:
            raise RuntimeError(f"no teams for {year}")
        return pl.DataFrame(rows)

    return cached("espn_team_names", fetch, season=year, source="espn", **kw)


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


# ─── In-season reads (M10b) ──────────────────────────────────────────────────
# Everything below is READ ONLY, like everything above it. §0.1 is asserted by
# test_inseason_delivery.py, which scans this file too — the in-season package
# can only be as read-only as what it reads through.

PROJECTION_SCHEMA = {
    "espn_id": pl.Utf8, "name": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8,
    "week": pl.Int64, "projected_points": pl.Float64, "actual_points": pl.Float64,
    "season_projected_points": pl.Float64, "season_actual_points": pl.Float64,
    "season_projected_avg": pl.Float64,
    "injury_status": pl.Utf8, "lineup_slot": pl.Utf8, "on_team_id": pl.Int64,
    "percent_owned": pl.Float64, "source": pl.Utf8,
}
"""Explicit, for the same reason PLAYER_SCHEMA is: ESPN returns [] rather than
null for several fields, and polars' inference dies on an all-empty column."""

SEASON_PROJECTION_COLUMNS = {
    "season_projected_points", "season_actual_points", "season_projected_avg",
}
"""Added after the first live in-season run. Used to spot a pre-migration cache."""

SEASON_TOTAL_PERIOD = 0
"""ESPN keys ``Player.stats`` by scoringPeriodId, and 0 is the SEASON total
rather than a week. Summing the dict naively counts the whole season twice.

It is also the ONLY key that exists before ESPN starts publishing weeks, and it
carries the number the rest-of-season anchor actually wants — see
``ros.from_espn``. Skipping it as "not a week" was right; never READING it was
the bug, because it left one published week to stand in for fourteen."""


def _projection_rows(players, weeks: tuple[int, ...], source: str) -> list[dict]:
    """One row per (player, week) from ESPN's own per-week projections.

    This is F1's resolution in practice: the anchor is POINTS, not RANKS,
    because a rank has a format and a points projection does not. ESPN's numbers
    already arrive in §1 scoring — M2 proved to the decimal that our ruleset
    reproduces ESPN's totals from ESPN's own stat line.
    """
    rows: list[dict] = []
    for p in players:
        stats = getattr(p, "stats", None) or {}
        # Read stats[0] DIRECTLY rather than via espn_api's
        # ``projected_total_points``: that attribute is ``.get(0, {}).get(
        # 'projected_points', 0)``, so a player ESPN has not projected at all
        # and a player projected at zero both arrive as 0.0. Those are opposite
        # facts and the whole anchor turns on telling them apart.
        season = stats.get(SEASON_TOTAL_PERIOD) or {}
        base = {
            "espn_id": str(getattr(p, "playerId", "") or ""),
            "name": _scalar(getattr(p, "name", None)),
            "position": _scalar(getattr(p, "position", None)),
            "team": _scalar(getattr(p, "proTeam", None)),
            "injury_status": _scalar(getattr(p, "injuryStatus", None)),
            "lineup_slot": _scalar(getattr(p, "lineupSlot", None)),
            "on_team_id": _scalar(getattr(p, "onTeamId", None), int),
            "percent_owned": _scalar(getattr(p, "percent_owned", None), float),
            "season_projected_points": _scalar(season.get("projected_points"), float),
            "season_actual_points": _scalar(season.get("points"), float),
            "season_projected_avg": _scalar(season.get("projected_avg_points"), float),
            "source": source,
        }
        for wk in weeks:
            if wk == SEASON_TOTAL_PERIOD:
                continue
            s = stats.get(wk) or {}
            rows.append({
                **base,
                "week": int(wk),
                "projected_points": _scalar(s.get("projected_points"), float),
                "actual_points": _scalar(s.get("points"), float),
            })
    return rows


def player_projections(
    year: int = SEASON,
    weeks: tuple[int, ...] | None = None,
    free_agent_size: int = 400,
    **kw,
) -> pl.DataFrame:
    """Per-week projected AND actual points for every rostered player and the
    top of free agency — the frame ``ros.from_espn`` turns into the anchor.

    ``free_agent_size`` is bounded on purpose. §3.2 says the wire stays rich all
    season (153 roster spots against 500+ relevant players), but the bottom of a
    1200-deep pool is practice-squad noise that costs a slow request and adds
    nothing a claim would ever be made on.
    """
    from ff_agent.config import REGULAR_SEASON_WEEKS

    wks = tuple(weeks) if weeks else tuple(REGULAR_SEASON_WEEKS)

    def fetch() -> pl.DataFrame:
        lg = get_league(year)
        rows: list[dict] = []
        for t in getattr(lg, "teams", []) or []:
            rows += _projection_rows(getattr(t, "roster", []) or [], wks, "roster")
        try:
            rows += _projection_rows(lg.free_agents(size=free_agent_size), wks,
                                     "free_agent")
        except Exception as exc:
            raise ESPNUnavailable(
                f"free_agents({free_agent_size}) failed: {exc}"
            ) from exc
        if not rows:
            raise ESPNUnavailable(
                f"No projections returned for {year}. If the season has not "
                f"started ESPN may not have published weekly projections yet."
            )
        return pl.DataFrame(rows, schema=PROJECTION_SCHEMA)

    df = cached("espn_projections", fetch, season=year, source="espn", **kw)
    # A parquet written before the season columns existed is not stale by the
    # TTL and is not wrong — it is INCOMPLETE, which the staleness check cannot
    # see. Serving it would silently fall back to extrapolating one published
    # week across fourteen, which is the exact failure the season anchor exists
    # to remove. Refetch once; offline it cannot, and says so.
    if SEASON_PROJECTION_COLUMNS - set(df.columns):
        if kw.get("offline"):
            raise ESPNUnavailable(
                "the cached projection table predates the season-projection "
                "columns and cannot be refreshed offline.\n"
                "  Fix: run once online — `uv run python -m ff_agent.cli "
                "monitor --job refresh`."
            )
        df = cached("espn_projections", fetch, season=year, source="espn",
                    **{**kw, "force": True})
    return df


def current_rosters(year: int = SEASON, **kw) -> pl.DataFrame:
    """Live rosters WITH lineup slots — what ESPN currently has starting.

    Distinct from ``rosters()``, which is the historical end-of-season shape and
    carries no slot. §5.3 needs the slot, because a locked player's slot is what
    is already spent and our idea of what the lineup should have been is not.
    """
    def fetch() -> pl.DataFrame:
        lg = get_league(year)
        rows = []
        for t in getattr(lg, "teams", []) or []:
            owners = getattr(t, "owners", None) or []
            owner = ", ".join(
                (o.get("firstName", "") + " " + o.get("lastName", "")).strip()
                if isinstance(o, dict) else str(o) for o in owners
            )
            for p in getattr(t, "roster", []) or []:
                rows.append({
                    "team_id": getattr(t, "team_id", None),
                    "fantasy_team": normalize_team_name(getattr(t, "team_name", None)),
                    "manager": owner or None,
                    "espn_id": str(getattr(p, "playerId", "") or ""),
                    "name": _scalar(getattr(p, "name", None)),
                    "position": _scalar(getattr(p, "position", None)),
                    "team": _scalar(getattr(p, "proTeam", None)),
                    "lineup_slot": _scalar(getattr(p, "lineupSlot", None)),
                    "injury_status": _scalar(getattr(p, "injuryStatus", None)),
                })
        if not rows:
            raise ESPNUnavailable(
                f"No rosters for {year}. If the draft has not happened yet, this "
                f"is expected."
            )
        return pl.DataFrame(rows, schema={
            "team_id": pl.Int64, "fantasy_team": pl.Utf8, "manager": pl.Utf8,
            "espn_id": pl.Utf8, "name": pl.Utf8, "position": pl.Utf8,
            "team": pl.Utf8, "lineup_slot": pl.Utf8, "injury_status": pl.Utf8,
        })

    return cached("espn_current_rosters", fetch, season=year, source="espn", **kw)


def weekly_results(year: int = SEASON, through_week: int | None = None, **kw) -> pl.DataFrame:
    """Actual team scores for weeks already played — the simulator's ``completed``.

    A week-9 title probability that re-simulates weeks 1-8 is not a forecast of
    this season, so this is what stops it being one. Only weeks with a real,
    non-zero score are returned: ESPN reports an unplayed matchup as 0-0, and
    feeding that in as a result would hand every team a shutout.
    """
    def fetch() -> pl.DataFrame:
        lg = get_league(year)
        reg = int(getattr(getattr(lg, "settings", None), "reg_season_count", 14))
        last = min(through_week or reg, reg)
        rows = []
        for wk in range(1, last + 1):
            try:
                boxes = lg.box_scores(wk)
            except Exception:
                continue                       # week not yet available
            for b in boxes:
                for team, score in (
                    (getattr(b, "home_team", None), getattr(b, "home_score", 0.0)),
                    (getattr(b, "away_team", None), getattr(b, "away_score", 0.0)),
                ):
                    name = normalize_team_name(getattr(team, "team_name", None)) if team else None
                    if not name:
                        continue              # a bye is encoded as a null side
                    rows.append({"week": wk, "team": name, "points": float(score or 0.0)})
        if not rows:
            return pl.DataFrame(schema={"week": pl.Int64, "team": pl.Utf8,
                                        "points": pl.Float64})
        df = pl.DataFrame(rows)
        # Drop weeks nobody has scored in — ESPN shows an unplayed matchup as 0-0.
        played = (
            df.group_by("week").agg(pl.col("points").max().alias("hi"))
            .filter(pl.col("hi") > 0)["week"].to_list()
        )
        return df.filter(pl.col("week").is_in(played)).unique(subset=["week", "team"])

    return cached("espn_results", fetch, season=year, source="espn", **kw)


def started_lineup(year: int, week: int) -> pl.DataFrame:
    """Who was actually STARTED in a given week — §11 step 10's ground truth.

    ``box_scores`` carries ``slot_position`` per player, which is the only place
    the distinction between started and rostered survives.
    """
    lg = get_league(year)
    rows = []
    for b in lg.box_scores(week):
        for team, lineup in (
            (getattr(b, "home_team", None), getattr(b, "home_lineup", []) or []),
            (getattr(b, "away_team", None), getattr(b, "away_lineup", []) or []),
        ):
            name = normalize_team_name(getattr(team, "team_name", None)) if team else None
            if not name:
                continue
            for p in lineup:
                rows.append({
                    "week": week, "fantasy_team": name,
                    "espn_id": str(getattr(p, "playerId", "") or ""),
                    "name": _scalar(getattr(p, "name", None)),
                    "slot_position": _scalar(getattr(p, "slot_position", None)),
                    "points": _scalar(getattr(p, "points", None), float),
                    "projected_points": _scalar(getattr(p, "projected_points", None), float),
                })
    if not rows:
        raise ESPNUnavailable(f"No box scores for {year} week {week}.")
    return pl.DataFrame(rows)


def transactions(year: int, scoring_period: int | None = None) -> pl.DataFrame:
    """Adds, drops, waiver claims and FAILED claims.

    ``WAIVER_ERROR`` rows are the league's only observed counterfactual — the
    sole evidence available for calibrating §9.3's P(claim succeeds), and what
    F4's most-added control is rebuilt from now that ``player_owned_espn`` turns
    out to be null in-season.
    """
    lg = get_league(year)
    try:
        txns = lg.transactions(
            scoring_period=scoring_period,
            types={"FREEAGENT", "WAIVER", "WAIVER_ERROR"},
        )
    except Exception as exc:
        raise ESPNUnavailable(f"transactions({scoring_period}) failed: {exc}") from exc
    rows = []
    for t in txns or []:
        for action in getattr(t, "actions", []) or []:
            team, verb, player = (list(action) + [None, None, None])[:3]
            rows.append({
                "scoring_period": scoring_period,
                "fantasy_team": normalize_team_name(getattr(team, "team_name", None))
                if team else None,
                "action": verb,
                "espn_id": str(getattr(player, "playerId", "") or "") or None,
                "name": getattr(player, "name", None) if player else None,
            })
    if not rows:
        return pl.DataFrame(schema={
            "scoring_period": pl.Int64, "fantasy_team": pl.Utf8,
            "action": pl.Utf8, "espn_id": pl.Utf8, "name": pl.Utf8})
    return pl.DataFrame(rows)


def roster_week(year: int, week: int) -> pl.DataFrame:
    """Every team's roster AS OF a past week — F3's reconstruction.

    ``load_roster_week`` re-requests ``mRoster`` with a ``scoringPeriodId`` and
    mutates the League in place, which is why this returns a fresh frame rather
    than a view: calling it twice with different weeks would otherwise silently
    change what an earlier result meant.

    This is what makes the §11 step 10 gate possible at all. ESPN does not
    retain "who was a free agent in week 6", but week W's pool is exactly the
    draftable universe minus the union of week-W rosters.
    """
    lg = get_league(year)
    try:
        lg.load_roster_week(week)
    except Exception as exc:
        raise ESPNUnavailable(
            f"load_roster_week({week}) failed for {year}: {exc}"
        ) from exc
    rows = []
    for t in getattr(lg, "teams", []) or []:
        for p in getattr(t, "roster", []) or []:
            rows.append({
                "week": week,
                "team_id": getattr(t, "team_id", None),
                "fantasy_team": normalize_team_name(getattr(t, "team_name", None)),
                "espn_id": str(getattr(p, "playerId", "") or ""),
                "name": _scalar(getattr(p, "name", None)),
                "position": _scalar(getattr(p, "position", None)),
                "team": _scalar(getattr(p, "proTeam", None)),
            })
    if not rows:
        raise ESPNUnavailable(f"No week-{week} rosters for {year}.")
    return pl.DataFrame(rows)
