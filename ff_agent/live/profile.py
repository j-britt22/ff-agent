"""Detect a league's shape from ESPN, instead of hardcoding one.

CLAUDE.md §1 pins this project to one 9-team, 2-QB league, and that is right for
the tool its owner drafts with. But nothing in the *engine* is specific to it —
the scoring already loads from ESPN, and team count, roster slots and playoff
shape are all in the settings payload. What was specific was `config.py`
asserting one league's answers as constants.

This reads them instead, so the same code can be handed to somebody in a
different league. Two rules carry over unchanged:

  * **§0.1** — read-only, always. Nothing here writes to ESPN.
  * **§10** — fail loudly. A league whose shape we cannot read is an error with
    an instruction, never a silent default that quietly produces a wrong board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ESPN's own slot vocabulary -> ours. §1 calls the flex FLEX; ESPN calls it
# RB/WR/TE. Everything not listed is an IDP or exotic slot this engine does not
# model, and a league that uses one is refused rather than silently mis-scored.
SLOT_ALIASES = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "RB/WR/TE": "FLEX", "D/ST": "DST", "K": "K",
}
UNSUPPORTED_SLOTS = (
    "TQB", "RB/WR", "WR/TE", "OP", "DT", "DE", "LB", "DL", "CB", "S", "DB",
    "DP", "P", "HC",
)

# ESPN's numeric position ids, used by rosterSettings.positionLimits. espn_api
# does not surface that table at all, so the raw endpoint is the only source
# for the per-position roster caps §1 records; -1 there means "no limit".
ESPN_POSITION_IDS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}


class UnsupportedLeague(RuntimeError):
    """The league uses a feature this engine does not model. Say which."""


@dataclass
class LeagueProfile:
    """Everything the draft tools need to know about a league, read from ESPN."""

    season: int
    league_id: str
    league_name: str
    n_teams: int
    my_team_id: int
    my_team_name: str
    my_manager: str
    managers: list[str]                 # by team_id order, ESPN's own ordering
    starter_slots: dict[str, int]
    bench_slots: int
    ir_slots: int
    roster_total: int
    position_maxima: dict[str, int]
    playoff_teams: int
    regular_season_weeks: int
    my_bye_weeks: tuple[int, ...]
    is_keeper: bool
    uses_faab: bool
    my_slot: int | None = None          # 1-based; None until ESPN sets the order
    draft_epoch_ms: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def starters(self) -> int:
        return sum(self.starter_slots.values())

    @property
    def qb_slots(self) -> int:
        return self.starter_slots.get("QB", 1)

    @property
    def is_superflex(self) -> bool:
        """Two starting QBs changes which consensus ranking is even valid (M3)."""
        return self.qb_slots >= 2

    def summary(self) -> str:
        shape = " ".join(f"{k}{v}" for k, v in self.starter_slots.items() if v)
        return (
            f"{self.league_name} — {self.n_teams} teams, {self.roster_total}-man "
            f"rosters\n  starters: {shape}\n  you: {self.my_team_name} "
            f"({self.my_manager})\n  {'2-QB / superflex' if self.is_superflex else '1-QB'}"
            f" · {self.playoff_teams} playoff teams · "
            f"{'keeper' if self.is_keeper else 'redraft'}"
        )


def _swid(creds: dict) -> str:
    return (creds.get("swid") or "").strip().strip("{}").upper()


def _owner_ids(team) -> list[str]:
    """Every owner id on a team, normalised for comparison against SWID."""
    out = []
    for o in getattr(team, "owners", None) or []:
        raw = o.get("id") if isinstance(o, dict) else o
        if raw:
            out.append(str(raw).strip().strip("{}").upper())
    return out


def detect(season: int | None = None, my_team_name: str | None = None) -> LeagueProfile:
    """Read the league's shape. Identifies WHICH team is yours from your SWID.

    The SWID cookie is the account that is logged in, so it identifies the human
    without anybody having to type a team name that they may since have changed
    (§1's "A Chane Reaction" is called "TBD" now — names are not identity).
    ``my_team_name`` is only a fallback for shared or orphaned accounts.
    """
    from ff_agent.config import SEASON, espn_credentials, normalize_team_name
    from ff_agent.data.espn import get_league
    from ff_agent.opponents.history import _owner_names, canonical_manager

    season = season or SEASON
    creds = espn_credentials()
    lg = get_league(season)
    s = getattr(lg, "settings", None)
    if s is None:
        raise UnsupportedLeague(f"ESPN returned no settings for league {season}.")

    teams = list(getattr(lg, "teams", []) or [])
    if not teams:
        raise UnsupportedLeague(f"ESPN returned no teams for {season}.")

    # ── which team is mine ───────────────────────────────────────────────────
    me = None
    swid = _swid(creds)
    if swid:
        me = next((t for t in teams if swid in _owner_ids(t)), None)
    if me is None and my_team_name:
        want = normalize_team_name(my_team_name)
        me = next((t for t in teams
                   if normalize_team_name(t.team_name) == want), None)
    if me is None:
        raise UnsupportedLeague(
            f"Could not tell which of the {len(teams)} teams is yours.\n"
            f"  Your SWID ({swid[:8]}…) does not own any team in this league.\n"
            f"  Either ESPN_SWID belongs to a different account, or you are a\n"
            f"  co-owner ESPN does not list. Fix: pass --team \"Your Team Name\".\n"
            f"  Teams here: " + ", ".join(t.team_name for t in teams)
        )

    # ── roster shape ─────────────────────────────────────────────────────────
    raw_slots: dict[str, Any] = dict(getattr(s, "position_slot_counts", {}) or {})
    if not raw_slots:
        raise UnsupportedLeague("ESPN reported no roster slots for this league.")

    warnings: list[str] = []
    exotic = [k for k in UNSUPPORTED_SLOTS if int(raw_slots.get(k, 0) or 0) > 0]
    if exotic:
        raise UnsupportedLeague(
            f"This league starts {', '.join(exotic)}, which this engine does not\n"
            f"  model — the projections cover QB/RB/WR/TE/K/D-ST only. IDP and\n"
            f"  exotic slots would silently produce a wrong board, so this stops\n"
            f"  here rather than guessing (§10)."
        )

    starters = {}
    for espn_name, ours in SLOT_ALIASES.items():
        n = int(raw_slots.get(espn_name, 0) or 0)
        if n:
            starters[ours] = starters.get(ours, 0) + n
    if not starters:
        raise UnsupportedLeague("No recognised starting slots in this league.")

    bench = int(raw_slots.get("BE", 0) or 0)
    ir = int(raw_slots.get("IR", 0) or 0)
    total = sum(starters.values()) + bench      # IR sits OUTSIDE the roster

    maxima = _position_limits(lg, total, warnings)

    # ── my schedule: how many games, and which weeks are free ────────────────
    reg = int(getattr(s, "reg_season_count", 0) or 0)
    sched = list(getattr(me, "schedule", []) or [])
    my_byes = tuple(
        wk for wk, opp in enumerate(sched[:reg], start=1)
        if getattr(opp, "team_id", None) == getattr(me, "team_id", None)
    )

    names = _owner_names(me)
    managers = []
    for t in teams:
        n = _owner_names(t)
        managers.append(canonical_manager(n[0]) if n else f"team {t.team_id}")

    if not my_byes:
        warnings.append(
            "you play every regular-season week — no free byes, so §2.1's "
            "bye-week arbitrage does not apply in this league")

    prof = LeagueProfile(
        season=season,
        league_id=str(creds.get("league_id", "")),
        league_name=getattr(s, "name", None) or f"league {creds.get('league_id')}",
        n_teams=len(teams),
        my_team_id=int(getattr(me, "team_id", 0) or 0),
        my_team_name=normalize_team_name(me.team_name),
        my_manager=canonical_manager(names[0]) if names else "me",
        managers=managers,
        starter_slots=starters,
        bench_slots=bench,
        ir_slots=ir,
        roster_total=total,
        position_maxima=maxima,
        playoff_teams=int(getattr(s, "playoff_team_count", 0) or 0),
        regular_season_weeks=reg,
        my_bye_weeks=my_byes,
        is_keeper=bool(int(getattr(s, "keeper_count", 0) or 0)),
        uses_faab=bool(getattr(s, "faab", False)),
        warnings=warnings,
    )
    if prof.is_keeper:
        prof.warnings.append(
            "this is a KEEPER league; the board values every player as though "
            "the whole pool is available, which overvalues anyone already kept")

    prof.my_slot, prof.draft_epoch_ms = _detect_slot(lg, prof.my_team_id, prof.n_teams)
    return prof


def _detect_slot(lg, my_team_id: int, n_teams: int) -> tuple[int | None, int | None]:
    """Read the snake position ESPN has already assigned, if it has.

    §5 says the slot is "revealed ~1hr before draft" and treats that as a
    T-60 constraint the tool must survive without. It does not say the slot is
    UNKNOWABLE before then — ``draftSettings.pickOrder`` is the commissioner's
    round-1 order, and once the league sets it, it is just sitting there in the
    settings payload, however far ahead of the draft that happens to be. This
    reads it instead of assuming it can't be read; ``--slot`` remains the
    override for whenever this comes back empty or wrong.
    """
    try:
        req = getattr(lg, "espn_request", None)
        raw = req.league_get(params={"view": "mSettings"}) if req else {}
        ds = raw.get("settings", {}).get("draftSettings", {}) or {}
        order = ds.get("pickOrder") or []
        epoch = ds.get("date")
    except Exception:
        return None, None
    if not order or my_team_id not in order:
        return None, epoch
    if ds.get("type") != "SNAKE" or len(order) != n_teams:
        return None, epoch          # a shape this engine's snake math does not model
    return order.index(my_team_id) + 1, epoch


def _position_limits(lg, roster_total: int, warnings: list[str]) -> dict[str, int]:
    """Per-position roster caps, from the raw settings endpoint.

    These matter to the draft policy, not just to bookkeeping: M7 found that
    without them the simulated opponents drafted eight quarterbacks and no tight
    end, and fielded lineups with empty slots — a ~50 point per week error. So a
    league whose caps cannot be read is WARNED about loudly rather than quietly
    run uncapped.
    """
    out: dict[str, int] = {}
    try:
        req = getattr(lg, "espn_request", None)
        raw = req.league_get(params={"view": "mSettings"}) if req else {}
        limits = (raw.get("settings", {})
                     .get("rosterSettings", {})
                     .get("positionLimits", {})) or {}
    except Exception as exc:                    # offline, or ESPN changed shape
        limits = {}
        warnings.append(f"could not read position limits ({type(exc).__name__}); "
                        f"the draft will run uncapped")
    for pid, name in ESPN_POSITION_IDS.items():
        v = limits.get(str(pid), limits.get(pid))
        if v is None:
            continue
        out[name] = roster_total if int(v) < 0 else int(v)
    missing = [n for n in ESPN_POSITION_IDS.values() if n not in out]
    if missing:
        for n in missing:
            out[n] = roster_total
        if limits:
            warnings.append(f"no roster cap reported for {', '.join(missing)}")
    return out


def managers_for_slot(prof: LeagueProfile, slot: int) -> list[str]:
    """Seat order for a detected league, with me at ``slot``.

    ``draft.build.managers_for_slot`` reads §1's hardcoded OPPONENTS tuple; this
    is the same idea driven by whoever is actually in the league. M7 measured
    that reshuffling the opponents moves my mean by 0.4 weekly points, so the
    order of the OTHER seats is an earned shortcut — only my own slot matters.
    """
    if not 1 <= slot <= prof.n_teams:
        raise ValueError(f"slot {slot} is outside 1..{prof.n_teams}")
    others = [m for m in prof.managers if m != prof.my_manager]
    if len(others) != prof.n_teams - 1:
        # duplicate or missing manager names — fall back to seat labels so the
        # draft can still run, but say so rather than silently mis-seating.
        others = [f"seat {i + 1}" for i in range(prof.n_teams - 1)]
        prof.warnings.append(
            "manager names are not unique; opponents are labelled by seat")
    return others[: slot - 1] + [prof.my_manager] + others[slot - 1:]


def build_board_for(prof: LeagueProfile, season: int | None = None):
    """This league's board, with replacement set by ITS starting lineup.

    VOR is measured against the starter replacement, so a 1-QB or 12-team league
    gets a genuinely different board — not a rescaled one. Nothing about this is
    optional: using §1's replacement levels on another league's shape produces
    confident, wrong rankings.
    """
    from ff_agent.board import build as BB

    return BB.build(
        season or prof.season,
        starter_slots=prof.starter_slots,
        n_teams=prof.n_teams,
    )


def neutral_context(pool, managers, rounds: int, season: int):
    """Opponent model for a league with no usable draft history.

    M5 fits per-manager tilts from past drafts. A league in its first season has
    none, and M5's own finding is that a manager's history can MISLEAD across a
    format change anyway. So the fallback is honest rather than clever: no
    tilts, no reaches — everybody drafts the consensus board with M7's ADP
    kernel. That is exactly the `best_consensus` control M7 measured, and it
    still beat the model-with-tilts by most of the margin.
    """
    from ff_agent.draft import calibrate as CB
    from ff_agent.draft import scenarios as SC

    tilt: dict = {}
    reach: dict = {}
    scen = SC.fit_all(pool, managers, tilt, reach, rounds, season)
    rates = CB.target_rates(target_two_qb=True)
    return tilt, reach, scen, {"qb_cold": rates, "shipped": rates, "qb_hot": rates}
