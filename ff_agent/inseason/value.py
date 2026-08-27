"""The common currency: a roster becomes Δ P(championship).

§8 is the objective function and §2.4 explains why: with 67% of the league
qualifying, making the playoffs is the default outcome rather than the
achievement, and a first-round bye is worth roughly a doubling of title odds. So
every recommendation this package makes — a claim, a swap, a trade — is scored
here and nowhere else, or the jobs drift apart.

**Two speeds, deliberately.** M7 had to score ten thousand rosters per slot and
could not afford the real simulator, so it built an interpolated surface (and
paid for it with a silent clamping bug at the grid edge). A weekly job scores
perhaps thirty rosters, so **M10b can afford the real thing** — but a waiver run
screens a couple of hundred (add, drop) pairs before it gets to thirty. Hence:

  * ``weekly_delta`` — a lineup solve. Cheap, exact about lineups, says nothing
    about probability. Used to screen.
  * ``title_delta`` — the real season simulator. Used on the survivors.

The screen is a strict prefilter, not an approximation of the answer: a move that
does not change my starting lineup in any remaining week cannot change my title
odds either, so screening on lineup points loses nothing it should keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ff_agent.config import (
    MY_GAME_WEEKS, MY_TEAM_NAME, SEASON, STARTER_SLOTS,
)
from ff_agent.season import simulate as SIM

POSITIONS = ("QB", "RB", "WR", "TE", "DST", "K")
POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}

STRICT = tuple((POS_INDEX[p], n) for p, n in STARTER_SLOTS.items() if p != "FLEX")
FLEX_IDX = np.array([POS_INDEX[p] for p in ("RB", "WR", "TE")])
N_FLEX = STARTER_SLOTS.get("FLEX", 0)

NO_BYE = 0
"""How "this player has no bye" is encoded. NFL weeks are 1-18, so 0 is safe.

It must NOT be the same value as ``_NO_BYE_PROBE``: the bye cost is measured by
solving the lineup for a week nobody byes in, and if the two collide the probe
masks out every player and the cost comes back as the whole lineup. Caught by
a test that asserts a bye-free roster has a bye cost of exactly zero."""

_NO_BYE_PROBE = -1
"""The week we solve to price the bye. Belongs to nobody and is not ``NO_BYE``."""


@dataclass(frozen=True)
class RosterValue:
    mean: float
    """Expected weekly points from the optimal starting lineup."""
    mean_sd: float
    """Uncertainty in that mean. NOT week-to-week noise — see §2.4 and M7's
    measured variance crossover; being wrong about a player is being wrong in
    every remaining week, which moves standings far more than weekly noise."""
    bye_cost: float
    """Points a week lost to NFL byes falling in weeks I actually play."""


def _week_points(pts, sds, pos, bye, week):
    """One week's optimal lineup: strict slots first, then FLEX from the rest.

    Same rule as ``season/lineup.py``, in numpy so a screen over hundreds of
    candidate rosters stays cheap. Rows must already be sorted by value
    descending — ``cumsum`` is what makes "best available" work.
    """
    ok = bye != week
    starters = np.zeros(pts.shape, dtype=bool)
    for p, need in STRICT:
        sel = ok & (pos == p) & ~starters
        starters |= sel & (np.cumsum(sel, axis=1) <= need)
    if N_FLEX:
        sel = ok & np.isin(pos, FLEX_IDX) & ~starters
        starters |= sel & (np.cumsum(sel, axis=1) <= N_FLEX)
    return (pts * starters).sum(axis=1), (sds ** 2 * starters).sum(axis=1)


def batch_value(
    points: np.ndarray,
    sds: np.ndarray,
    pos_idx: np.ndarray,
    bye_week: np.ndarray,
    play_weeks: tuple[int, ...] = MY_GAME_WEEKS,
) -> RosterValue:
    """Weekly value for a batch of rosters. Arrays are (n_rosters, n_players).

    ``play_weeks`` is the manager's REMAINING game weeks, which is why it is a
    parameter rather than a constant: a decision made in week 9 is worth what it
    is worth over weeks 9-13, and my weeks 5 and 14 are not in anybody's
    denominator but my own (§2.1, §2.2).
    """
    if not play_weeks:
        raise ValueError(
            "play_weeks is empty. A roster with no remaining games has no weekly "
            "value, and returning 0 would rank every candidate identically."
        )
    points = np.atleast_2d(np.asarray(points, dtype=float))
    sds = np.atleast_2d(np.asarray(sds, dtype=float))
    pos_idx = np.atleast_2d(np.asarray(pos_idx, dtype=int))
    bye_week = np.atleast_2d(np.asarray(bye_week, dtype=int))

    order = np.argsort(-points, axis=1)
    points = np.take_along_axis(points, order, axis=1)
    sds = np.take_along_axis(sds, order, axis=1)
    pos_idx = np.take_along_axis(pos_idx, order, axis=1)
    bye_week = np.take_along_axis(bye_week, order, axis=1)

    tot = np.zeros(points.shape[0])
    var = np.zeros(points.shape[0])
    no_bye = _week_points(points, sds, pos_idx, bye_week, _NO_BYE_PROBE)[0]
    for w in play_weeks:
        t, v = _week_points(points, sds, pos_idx, bye_week, w)
        tot += t
        var += v
    n = len(play_weeks)
    return RosterValue(
        mean=tot / n, mean_sd=np.sqrt(var / n), bye_cost=no_bye - tot / n,
    )


# ─── DataFrame front door ────────────────────────────────────────────────────
REQUIRED = ("canonical_id", "position", "weekly_points")


def roster_value(
    roster: pl.DataFrame,
    play_weeks: tuple[int, ...] = MY_GAME_WEEKS,
    sd_col: str = "weekly_sd",
    bye_col: str = "bye_week",
) -> RosterValue:
    """One roster's weekly value. ``roster`` is the joined player frame."""
    missing = [c for c in REQUIRED if c not in roster.columns]
    if missing:
        raise KeyError(f"roster is missing {missing}; needs {list(REQUIRED)}.")
    if roster.is_empty():
        return RosterValue(0.0, 0.0, 0.0)

    unknown = sorted(set(roster["position"].to_list()) - set(POSITIONS))
    if unknown:
        raise ValueError(
            f"roster carries position(s) {unknown} the lineup solver does not "
            f"know. §10 refuses to approximate an unprojected position rather "
            f"than confidently mis-slotting it."
        )

    pts = roster["weekly_points"].fill_null(0.0).to_numpy()[None, :]
    sds = (
        roster[sd_col].fill_null(0.0).to_numpy()[None, :]
        if sd_col in roster.columns else np.zeros_like(pts)
    )
    pos = np.array([[POS_INDEX[p] for p in roster["position"].to_list()]])
    bye = (
        roster[bye_col].fill_null(NO_BYE).cast(pl.Int64).to_numpy()[None, :]
        if bye_col in roster.columns else np.full_like(pos, NO_BYE)
    )
    v = batch_value(pts, sds, pos, bye, play_weeks)
    return RosterValue(float(v.mean[0]), float(v.mean_sd[0]), float(v.bye_cost[0]))


def weekly_delta(
    before: pl.DataFrame,
    after: pl.DataFrame,
    play_weeks: tuple[int, ...] = MY_GAME_WEEKS,
) -> float:
    """The screen. Points per week gained by a roster change.

    Zero means the change never reaches my starting lineup in any remaining week
    — §9.3's "bench upgrades ≈ 0", enforced by arithmetic rather than by a rule
    somebody has to remember, and §3.2's "don't hoard handcuffs" with it.
    """
    return round(
        roster_value(after, play_weeks).mean - roster_value(before, play_weeks).mean, 3
    )


# ─── The real objective ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class TitleOdds:
    p_playoffs: float
    p_top2: float
    p_title: float
    completed_weeks: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "p_playoffs": round(self.p_playoffs, 4),
            "p_top2_seed": round(self.p_top2, 4),
            "p_title": round(self.p_title, 4),
            "weeks_played": len(self.completed_weeks),
        }


DEFAULT_SIMS = 20_000


def title_odds(
    team_means: dict[str, float],
    my_team: str = MY_TEAM_NAME,
    team_mean_sds: dict[str, float] | None = None,
    completed: pl.DataFrame | None = None,
    season: int = SEASON,
    n_sims: int = DEFAULT_SIMS,
    seed: int = 7,
) -> TitleOdds:
    """Run the real season simulator and pull out my row."""
    if my_team not in team_means:
        raise KeyError(
            f"{my_team!r} is not among the simulated teams {sorted(team_means)}.\n"
            f"  Team names are mutable in ESPN — resolve through manager or "
            f"team_id, never a hardcoded name (§1's names are a fallback)."
        )
    res = SIM.simulate(
        team_means, season=season, n_sims=n_sims, seed=seed,
        team_mean_sds=team_mean_sds, completed=completed,
    )
    i = res.teams.index(my_team)
    return TitleOdds(
        p_playoffs=float(res.p_playoffs[i]),
        p_top2=float(res.p_top2[i]),
        p_title=float(res.p_title[i]),
        completed_weeks=res.completed_weeks,
    )


def title_delta(
    team_means: dict[str, float],
    my_new_mean: float,
    my_team: str = MY_TEAM_NAME,
    also: dict[str, float] | None = None,
    **kw,
) -> dict[str, float]:
    """Δ P(playoffs / top-2 / title) from changing my weekly mean.

    ``also`` moves other teams at the same time, which is what a TRADE does —
    and it is how §2.3's double-up penalty prices itself without a special case:
    a partner I face twice gets stronger in two of my twelve games, so the
    simulator charges me twice, automatically.
    """
    base = title_odds(team_means, my_team, **kw)
    moved = dict(team_means)
    moved[my_team] = my_new_mean
    for t, m in (also or {}).items():
        if t not in moved:
            raise KeyError(f"{t!r} is not in the league: {sorted(moved)}")
        moved[t] = m
    new = title_odds(moved, my_team, **kw)
    return {
        "d_playoffs": round(new.p_playoffs - base.p_playoffs, 4),
        "d_top2_seed": round(new.p_top2 - base.p_top2, 4),
        "d_title": round(new.p_title - base.p_title, 4),
        "p_title_before": round(base.p_title, 4),
        "p_title_after": round(new.p_title, 4),
    }


# ─── §2.4's variance crossover, as a live posture ───────────────────────────
TITLE_CROSSOVER = 15.9
"""M7 measured it: below this roster delta variance PAYS, above it variance
spends the first-round bye. §2.4 asserts "roster variance is good" without
qualification and is right only on one side of this line."""

TOP2_CROSSOVER = 12.1
"""And the top-2 crossover arrives FIRST, which is why it is worth knowing which
of the two I am actually chasing."""


def posture(my_mean: float, league_mean: float) -> dict:
    """Should I be chasing ceiling or protecting the mean this week?

    §9.2 frames this per matchup; §2.4 frames it per season. Both matter, and
    they can disagree — a heavy favourite this week can still be a team that
    needs variance over the season. Reported, not resolved: the digest says
    which side of each crossover I am on and lets the numbers argue.
    """
    delta = my_mean - league_mean
    return {
        "delta_vs_league": round(delta, 2),
        "chase_variance_for_title": delta < TITLE_CROSSOVER,
        "chase_variance_for_top2": delta < TOP2_CROSSOVER,
        "note": (
            "clearly ahead — variance now spends the first-round bye"
            if delta >= TITLE_CROSSOVER else
            "ahead on top-2 but not on the title — the two crossovers disagree"
            if delta >= TOP2_CROSSOVER else
            "average or behind — ceiling is worth more than floor (§2.4)"
        ),
    }
