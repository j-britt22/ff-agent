"""The lock calendar (M10b finding F9).

**ESPN locks each player at HIS OWN kickoff.** So "set the lineup" is not one
decision — it is a sequence of irreversible per-slot commitments made under
increasing information, and the sequence is a different shape every week.

Measured on the real 2026 schedule, that is six to nine distinct lock times a
week, and the irregularities land exactly where §2.4 says the value is:

  * **week 1 opens on a WEDNESDAY** (2026-09-09 20:20), with Thursday at 20:35;
  * **week 16 — my semifinal — has three Christmas Day games** (13:00/16:30/20:15);
  * **week 15 — my quarterfinal — has two Saturday games** (17:00, 20:20);
  * Sunday's "late" window is really TWO windows, 16:05 and 16:25;
  * 2025 week 4 had a Sunday 09:30 London kickoff.

§9.1's four fixed slots (Thu 12:00 · Sat 10:00 · Sun 09:00 · Sun 11:15) would
miss the Wednesday opener entirely, miss all three Christmas games, run its
Saturday check seven hours early, and leave the whole Sunday late slate with no
inactive check at all.

So **decision points are derived from the schedule, never from the clock.** The
crontab is a dumb tick; everything that knows about Wednesday openers, Christmas
kickoffs and London games lives here, where it is testable.

Two things this module refuses to guess:

  * **Timezone.** Every time in §9.1 is Eastern and a container defaults to UTC.
    A job written as 11:15 in a UTC container fires at 06:15 or 07:15 ET
    depending on daylight saving. Asserted at startup, never assumed.
  * **Team vocabulary.** nflverse spells the Rams ``LA``; ESPN and the crosswalk
    spell them ``LAR``. The Rams trap has bitten this project three times, and
    the SECOND game of the 2026 season is ``SF @ LA`` on Thursday night — so an
    un-normalised join would lose a lock time in week 1. Canonicalised at the
    nflverse boundary, then asserted.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import polars as pl

from ff_agent.config import MY_BYE_WEEKS, REGULAR_SEASON_WEEKS, SEASON
from ff_agent.data import byes as BY
from ff_agent.data import crosswalk as CW

ET = ZoneInfo("America/New_York")
"""§9.1's timezone. Not configurable — the NFL schedule is published in it."""

REQUIRED_TZ = "America/New_York"


class ClockError(RuntimeError):
    """The calendar cannot be trusted. Always blocking (§10)."""


# ─── Checkpoint offsets ──────────────────────────────────────────────────────
ADVISORY_LEAD = dt.timedelta(hours=24)
"""One per week, before the week's FIRST lock. The only unconditional email."""

CONFIRM_LEAD = dt.timedelta(hours=3)
"""Before each window. Speaks only if that window's call moved."""

INACTIVES_LEAD = dt.timedelta(minutes=75)
"""Inactives drop at kickoff minus NINETY. Seventy-five leaves time to act, and
is why §9.1's fixed 11:15 is wrong in both directions: fifteen minutes early for
the 1pm slate and hours late for a 09:30 London game."""

MIN_RUNWAY = dt.timedelta(minutes=20)
"""A recommendation that lands after its deadline is worse than none. Below this
the digest escalates instead of advising."""

TICK = dt.timedelta(minutes=15)
"""How often the container wakes. A checkpoint fires when its due time falls in
the tick window, so nothing is missed and nothing fires twice."""


@dataclass(frozen=True)
class Kickoff:
    """One distinct lock window, and whose games are in it."""
    at: dt.datetime
    teams: tuple[str, ...]
    game_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.at.tzinfo is None:
            raise ClockError(
                f"kickoff {self.at} is naive. Every time in this module carries a "
                f"zone — a naive datetime is how a Christmas Day 13:00 becomes 08:00."
            )


CHECKPOINT_KINDS = ("advisory", "confirm", "inactives", "monday_close")


@dataclass(frozen=True)
class Checkpoint:
    kind: str
    window: dt.datetime
    """The kickoff this checkpoint is about."""
    due_at: dt.datetime
    """When the job should run."""
    teams: tuple[str, ...] = ()
    week: int = 0
    unconditional: bool = False
    """True only for the weekly advisory. Everything else speaks on a change."""

    @property
    def runway(self) -> dt.timedelta:
        return self.window - self.due_at

    def label(self) -> str:
        return f"{self.kind}@{self.window.astimezone(ET):%a %m/%d %H:%M} ET"


# ─── Timezone ────────────────────────────────────────────────────────────────
def local_timezone_name() -> str:
    return os.environ.get("TZ") or str(dt.datetime.now().astimezone().tzinfo)


def assert_timezone(strict: bool = True) -> str:
    """F7: a container defaults to UTC and nothing about that looks wrong.

    Returns the observed zone. In ``strict`` mode a mismatch is blocking — the
    Sunday inactives job is the highest-leverage fifteen minutes of the week and
    it is the one that silently moves by four hours.
    """
    tz = os.environ.get("TZ", "")
    if tz == REQUIRED_TZ:
        return tz
    # No TZ env var is fine if the machine itself is already on Eastern.
    offset_matches = (
        dt.datetime.now().astimezone().utcoffset()
        == dt.datetime.now(ET).utcoffset()
    )
    if not tz and offset_matches:
        return local_timezone_name()
    msg = (
        f"TZ is {tz or local_timezone_name()!r}, not {REQUIRED_TZ!r}.\n"
        f"  Every time in §9.1 is Eastern. In a UTC container the Sunday\n"
        f"  inactives job fires at 06:15 or 07:15 ET depending on daylight\n"
        f"  saving — four hours before the inactives it exists to read.\n"
        f"  Fix: set TZ=America/New_York in the image and compose file."
    )
    if strict:
        raise ClockError(msg)
    return f"WRONG: {msg}"


def now_et() -> dt.datetime:
    return dt.datetime.now(tz=ET)


# ─── The kickoff table ───────────────────────────────────────────────────────
def kickoff_table(season: int = SEASON, schedule: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per (team, week) with the exact kickoff, in Eastern.

    Canonicalises the team vocabulary at the nflverse boundary rather than at
    each join downstream — the Rams trap bit this project three times by being
    fixed downstream, and 2026's second game is ``SF @ LA``.
    """
    if schedule is None:
        from ff_agent.data import nflverse as NV
        schedule = NV.schedules()

    games = schedule.filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    if games.is_empty():
        raise ClockError(
            f"no {season} regular-season games in the schedule table. "
            f"Seasons present: {sorted(schedule['season'].unique().to_list())[-5:]}"
        )

    missing = games.filter(pl.col("gametime").is_null() | pl.col("gameday").is_null())
    if missing.height:
        raise ClockError(
            f"{missing.height} {season} game(s) have no kickoff time, e.g. "
            f"{missing.head(3)['game_id'].to_list()}. A lock calendar built from "
            f"a partial schedule silently drops those windows."
        )

    long = pl.concat([
        games.select("game_id", "week", "gameday", "gametime",
                     pl.col("home_team").alias("team"),
                     pl.col("away_team").alias("opponent"),
                     pl.lit(True).alias("home")),
        games.select("game_id", "week", "gameday", "gametime",
                     pl.col("away_team").alias("team"),
                     pl.col("home_team").alias("opponent"),
                     pl.lit(False).alias("home")),
    ])

    long = long.with_columns(
        pl.col("team").map_elements(CW.normalize_team, return_dtype=pl.Utf8).alias("team"),
        pl.col("opponent").map_elements(CW.normalize_team, return_dtype=pl.Utf8).alias("opponent"),
    )
    BY.assert_canonical_teams(long, season, "team")
    BY.assert_canonical_teams(long, season, "opponent")

    out = long.with_columns(
        (pl.col("gameday") + " " + pl.col("gametime"))
        .str.to_datetime("%Y-%m-%d %H:%M", time_zone=None)
        .dt.replace_time_zone(REQUIRED_TZ)
        .alias("kickoff")
    ).drop("gameday", "gametime")

    dupes = out.group_by("team", "week").len().filter(pl.col("len") > 1)
    if dupes.height:
        raise ClockError(
            f"{dupes.height} team-week(s) appear twice in the {season} schedule, "
            f"e.g. {dupes.head(3).to_dicts()}. Two kickoffs for one team-week "
            f"means one of them is a lock time we would apply to the wrong game."
        )
    return out.sort("week", "kickoff", "team")


def lock_windows(
    week: int,
    teams: set[str] | list[str] | None = None,
    season: int = SEASON,
    kickoffs: pl.DataFrame | None = None,
) -> list[Kickoff]:
    """The distinct kickoff windows in ``week``, restricted to ``teams``.

    Restricting matters: my roster covers a handful of NFL teams, so most of a
    week's windows are not my problem, and a checkpoint per window I have no
    player in is pure notification noise.
    """
    kt = kickoff_table(season) if kickoffs is None else kickoffs
    wk = kt.filter(pl.col("week") == week)
    if teams is not None:
        wanted = {CW.normalize_team(t) for t in teams} - {None}
        wk = wk.filter(pl.col("team").is_in(list(wanted)))
    if wk.is_empty():
        return []
    out = []
    for (at,), grp in wk.group_by("kickoff", maintain_order=True):
        out.append(Kickoff(
            at=at,
            teams=tuple(sorted(grp["team"].to_list())),
            game_ids=tuple(sorted(set(grp["game_id"].to_list()))),
        ))
    return sorted(out, key=lambda k: k.at)


# ─── Which league week is it ─────────────────────────────────────────────────
WEEK_TAIL = dt.timedelta(hours=6)
"""A week stays current until its last game is comfortably over."""


def current_week(
    season: int = SEASON,
    now: dt.datetime | None = None,
    kickoffs: pl.DataFrame | None = None,
) -> int | None:
    """The league week in progress, or ``None`` once the regular season is done.

    Defined from the schedule rather than from a date arithmetic guess: the
    current week is the earliest whose last kickoff has not yet finished. That
    is correct through Wednesday openers, Saturday games and flex scheduling
    without knowing any of them exist.
    """
    now = now or now_et()
    kt = kickoff_table(season) if kickoffs is None else kickoffs
    last = (
        kt.group_by("week").agg(pl.col("kickoff").max().alias("last"))
        .sort("week")
    )
    for row in last.iter_rows(named=True):
        if now < row["last"] + WEEK_TAIL:
            return int(row["week"])
    return None


def is_my_bye(week: int | None) -> bool:
    """§2.1. Weeks 5 and 14 — nothing on my roster matters."""
    return week in MY_BYE_WEEKS


def assert_not_my_bye(week: int | None) -> None:
    """§10 lists a lineup set for week 5 or 14 as a sign something is broken."""
    if is_my_bye(week):
        raise ClockError(
            f"week {week} is one of MY fantasy byes {sorted(MY_BYE_WEEKS)}.\n"
            f"  There is no lineup to set and no game to lose. §10 lists this as\n"
            f"  an alarm meaning something upstream is wrong."
        )


# ─── Checkpoints ─────────────────────────────────────────────────────────────
def checkpoints_for_week(
    week: int,
    teams: set[str] | list[str] | None = None,
    season: int = SEASON,
    kickoffs: pl.DataFrame | None = None,
) -> list[Checkpoint]:
    """Every decision point in a week, derived from that week's real kickoffs.

    One unconditional advisory before the week's first lock, then confirm and
    inactives passes per window, then the Monday close. Checkpoints are NOT
    emails: only the advisory always speaks. Nine messages a week is precisely
    the fatigue that stops a digest being read by October.
    """
    windows = lock_windows(week, teams, season, kickoffs)
    if not windows:
        return []

    out: list[Checkpoint] = [
        Checkpoint(
            kind="advisory",
            window=windows[0].at,
            due_at=windows[0].at - ADVISORY_LEAD,
            teams=tuple(sorted({t for w in windows for t in w.teams})),
            week=week,
            unconditional=True,
        )
    ]
    for w in windows:
        out.append(Checkpoint("confirm", w.at, w.at - CONFIRM_LEAD, w.teams, week))
        out.append(Checkpoint("inactives", w.at, w.at - INACTIVES_LEAD, w.teams, week))

    last = windows[-1]
    if last.at.astimezone(ET).weekday() == 0:          # Monday
        out.append(Checkpoint("monday_close", last.at, last.at - CONFIRM_LEAD,
                              last.teams, week))
    return sorted(out, key=lambda c: c.due_at)


def due(
    checkpoints: list[Checkpoint],
    now: dt.datetime | None = None,
    tick: dt.timedelta = TICK,
) -> list[Checkpoint]:
    """Checkpoints whose due time falls in the tick window ending now.

    A half-open window ``(now - tick, now]`` so a checkpoint fires exactly once
    across consecutive ticks — the container wakes every fifteen minutes and a
    double-send is as bad as a miss.
    """
    now = now or now_et()
    return [c for c in checkpoints if now - tick < c.due_at <= now]


def next_lock(
    week: int,
    teams: set[str] | list[str] | None = None,
    season: int = SEASON,
    now: dt.datetime | None = None,
    kickoffs: pl.DataFrame | None = None,
) -> Kickoff | None:
    """The next window that has not kicked off. Every digest leads with this."""
    now = now or now_et()
    for w in lock_windows(week, teams, season, kickoffs):
        if w.at > now:
            return w
    return None


def week_summary(
    week: int,
    teams: set[str] | list[str] | None = None,
    season: int = SEASON,
    kickoffs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Human-readable lock calendar. This is what F9 looks like as a table."""
    rows = []
    for w in lock_windows(week, teams, season, kickoffs):
        local = w.at.astimezone(ET)
        rows.append({
            "week": week,
            "kickoff": local,
            "weekday": local.strftime("%A"),
            "time_et": local.strftime("%H:%M"),
            "n_teams": len(w.teams),
            "teams": ", ".join(w.teams),
        })
    if not rows:
        return pl.DataFrame(schema={
            "week": pl.Int64, "kickoff": pl.Datetime, "weekday": pl.Utf8,
            "time_et": pl.Utf8, "n_teams": pl.Int64, "teams": pl.Utf8,
        })
    return pl.DataFrame(rows)
