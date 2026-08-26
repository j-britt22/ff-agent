"""Weeks 15-17 (§2.4), and §2.2's free week 14.

§2.4 changes the objective: six of nine qualify, so **making the playoffs is the
default outcome rather than the achievement**. Only three teams miss. What
separates seasons is the top-two seed — in a 6-team, 3-week bracket the top two
sit out round one, needing two wins instead of three, which is roughly 12.5%
title odds against 25%. A first-round bye is worth about as much as everything
else combined.

So from around week 10 the digest carries a **playoff view**: roster strength
scored over weeks 15-17 ONLY, with December weather exposure and rest-risk
flagged.

§2.2's week 14 is a different job, not a louder waiver run. My fantasy byes are
weeks 5 and 14, so my record is locked a full week before everyone else's: no
lineup to set, no game to lose. Every drop is free and every add is a pure
playoff bet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ff_agent.config import MY_BYE_WEEKS, PLAYOFF_WEEKS
from ff_agent.inseason import value as V

PLAYOFF_VIEW_FROM = 10
"""Week at which weeks 15-17 start appearing beside the regular-season number."""

OUTDOOR_COLD = frozenset({
    "BUF", "GB", "CHI", "CLE", "PIT", "CIN", "NE", "NYJ", "NYG", "PHI", "WAS",
    "BAL", "DEN", "KC", "TEN", "SEA", "SF", "MIA", "JAX", "CAR", "TB",
})
"""Open-air stadiums. §2.4 asks for dome/weather risk on December outdoor
players; ADD-§F adds that only wind above 15 mph moves much."""

DOMES = frozenset({"ATL", "DAL", "DET", "HOU", "IND", "LV", "LA", "LAR", "LAC",
                   "MIN", "NO", "ARI"})


@dataclass
class PlayoffView:
    weekly_points: float
    bye_cost: float
    weather_exposed: list[str]
    rest_risk: list[str]
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "playoff_weeks": list(PLAYOFF_WEEKS),
            "weekly_points_15_17": round(self.weekly_points, 2),
            "weather_exposed": self.weather_exposed,
            "rest_risk": self.rest_risk,
            "notes": self.notes,
        }


def view(
    roster: pl.DataFrame,
    clinched_teams: set[str] | None = None,
) -> PlayoffView:
    """Roster strength over weeks 15-17 only.

    Byes do not exist this late in the NFL season, so ``play_weeks`` is simply
    the three playoff weeks — which also means the §2.1 free-bye term correctly
    contributes nothing here. A player whose value came mostly from a convenient
    bye is worth exactly his points in the bracket.
    """
    val = V.roster_value(roster, play_weeks=tuple(PLAYOFF_WEEKS))
    outdoor = [
        r["name"] for r in roster.iter_rows(named=True)
        if r.get("team") in OUTDOOR_COLD and r.get("position") in ("K", "QB", "WR")
    ]
    rest = [
        r["name"] for r in roster.iter_rows(named=True)
        if clinched_teams and r.get("team") in clinched_teams
    ]
    notes = []
    if rest:
        notes.append(
            "teams that have clinched may rest starters in weeks 16-17 — §2.4 "
            "lists this beside schedule strength as a real criterion, not a "
            "footnote"
        )
    return PlayoffView(val.mean, val.bye_cost, outdoor, rest, notes)


# ─── §2.2's free week ────────────────────────────────────────────────────────
@dataclass
class Week14Plan:
    drops: pl.DataFrame
    targets: pl.DataFrame
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "week": 14,
            "sets_a_lineup": False,
            "drop": self.drops["name"].to_list() if self.drops.height else [],
            "target": self.targets["name"].to_list() if self.targets.height else [],
            "notes": self.notes,
        }


def week14(
    roster: pl.DataFrame,
    free_agents: pl.DataFrame,
    keep: int = 8,
) -> Week14Plan:
    """§2.2's free-week churn. A different job, not a louder waiver run.

    Week 14 is my fantasy bye: no lineup to set, no game to lose, record already
    locked. So value is measured over weeks 15-17 ONLY — a player who is useful
    in week 14 and useless in the bracket is worth nothing here, and a
    high-variance stash I would never start in a live week is worth everything.
    §2.4's "you want the version of your team with the highest ceiling in weeks
    15-17" applies without qualification, because there is no floor to protect.
    """
    if 14 not in MY_BYE_WEEKS:
        raise ValueError(
            "week 14 is not one of my fantasy byes, so there is no free week to "
            "spend. §1's schedule has changed and this job no longer applies."
        )
    pw = tuple(PLAYOFF_WEEKS)
    scored_roster = roster.with_columns(
        pl.col("weekly_points").alias("_playoff_value"))
    scored_fa = free_agents.with_columns(
        pl.col("weekly_points").alias("_playoff_value"))

    base = V.roster_value(roster, pw).mean
    drops = scored_roster.sort("_playoff_value").head(keep)
    targets = scored_fa.sort("_playoff_value", descending=True).head(keep)

    notes = [
        "week 14 is my bye: no lineup, no game, record already locked. Every "
        "drop is free and every add is a pure weeks 15-17 bet.",
        f"roster is worth {base:.1f} pts/wk in the bracket as it stands.",
        "absorb risk here that would be reckless in a live week — §2.4's ceiling "
        "argument has no floor to trade against this week.",
    ]
    return Week14Plan(drops=drops, targets=targets, notes=notes)


def sets_no_lineup(week: int) -> bool:
    """§10's alarm, as a positive assertion. Weeks 5 and 14 set NO lineup."""
    return week in MY_BYE_WEEKS
