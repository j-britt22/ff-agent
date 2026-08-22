"""Drafted roster -> P(championship), fast enough to score 10,000 of them.

§8 is the objective function: "every recommendation reports its delta in
championship probability". M7 needs that for every simulated roster, and a
20,000-season simulation per roster is not available at 10k sims x 9 slots x
several policies.

So two steps.

**1. Roster -> (weekly mean, mean uncertainty).** The lineup rule is M6's, run in
numpy over the 17 drafted players instead of polars: fill the strict slots best
first, then take the FLEX from what is left — provably optimal for this slot
structure, per ``season/lineup.py``. Byes are applied week by week over the
twelve weeks I actually play, so §2.1's free bye is worth exactly what it is
worth and not a hand-tuned nudge. ``test_draft.py`` checks this against
``optimal_lineup`` on random rosters.

**2. A 2-D surface.** Run the real season simulator over a grid of
(my mean *relative to the league*, my mean uncertainty) and interpolate.
Relative, because drafting is zero-sum: a great roster for me is worse rosters
for them, and an absolute grid would count that twice.

The second axis is not decoration. §2.4 asserts three times that roster variance
is good — you qualify at 67% anyway, so you want ceiling. That has never been
measured here. The surface's slope in the uncertainty direction measures it, and
``validate()`` says how far the whole approximation can be trusted before any of
it is believed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ff_agent.config import MY_GAME_WEEKS, MY_TEAM_NAME, SEASON, STARTER_SLOTS
from ff_agent.draft import pool as PL
from ff_agent.season import simulate as SIM

GAMES = 17

STRICT = tuple(
    (PL.POS_INDEX[p], n) for p, n in STARTER_SLOTS.items() if p != "FLEX"
)
FLEX_IDX = np.array([PL.POS_INDEX[p] for p in ("RB", "WR", "TE")])
N_FLEX = STARTER_SLOTS.get("FLEX", 0)


def weekly_scale(season_value: np.ndarray, bye_week: np.ndarray = None) -> np.ndarray:
    """Season total -> per-game value. Always 17.

    An earlier version divided by 16 for anyone with a known bye, reasoning that
    a bye costs a game. **That was wrong, and it inflated every player by 6.25%.**
    The NFL plays 18 weeks and 17 GAMES: the bye is the eighteenth week, not one
    of the seventeen. Measured on 2024 actuals, 215 players logged `games == 17`,
    which is only possible if a full season is seventeen games.

    Double-charging is prevented where it should be — at the week level in
    ``_week_points``, which drops a player in his own bye week — and never
    depended on the divisor. 17 is also what ``board/build.py``,
    ``season/strength.py`` and ``projections/consensus.py`` all use, so this now
    agrees with the rest of the pipeline instead of quietly disagreeing with it.

    ``bye_week`` is accepted and ignored, so callers read symmetrically.
    """
    return season_value / GAMES


@dataclass
class RosterValue:
    mean: np.ndarray        # (n_sims,) expected weekly starting points
    mean_sd: np.ndarray     # (n_sims,) uncertainty in that mean
    bye_cost: np.ndarray    # (n_sims,) points lost to byes, vs a bye-free season


def lineup_stats(
    pool: PL.DraftPool,
    roster: np.ndarray,
    play_weeks: tuple[int, ...] = MY_GAME_WEEKS,
) -> RosterValue:
    """Weekly starting-lineup mean and uncertainty for each simulated roster."""
    pts = weekly_scale(pool.points[roster], pool.bye_week[roster]).astype(np.float64)
    sds = weekly_scale(pool.points_sd[roster], pool.bye_week[roster]).astype(np.float64)
    pos = pool.pos_idx[roster]
    bye = pool.bye_week[roster]

    order = np.argsort(-pts, axis=1)
    pts = np.take_along_axis(pts, order, axis=1)
    sds = np.take_along_axis(sds, order, axis=1)
    pos = np.take_along_axis(pos, order, axis=1)
    bye = np.take_along_axis(bye, order, axis=1)

    n_sims = roster.shape[0]
    tot = np.zeros(n_sims)
    var = np.zeros(n_sims)
    # week -1 belongs to nobody, so this is the same lineup with no bye at all
    no_bye = _week_points(pts, sds, pos, bye, -1)[0]
    for w in play_weeks:
        t, v = _week_points(pts, sds, pos, bye, w)
        tot += t
        var += v
    n = len(play_weeks)
    return RosterValue(
        mean=tot / n,
        mean_sd=np.sqrt(var / n),
        bye_cost=no_bye - tot / n,
    )


def _week_points(pts, sds, pos, bye, week):
    """One week's optimal lineup: strict slots first, then FLEX from the rest."""
    ok = bye != week
    starters = np.zeros(pts.shape, dtype=bool)
    for p, need in STRICT:
        sel = ok & (pos == p) & ~starters
        starters |= sel & (np.cumsum(sel, axis=1) <= need)
    if N_FLEX:
        sel = ok & np.isin(pos, FLEX_IDX) & ~starters
        starters |= sel & (np.cumsum(sel, axis=1) <= N_FLEX)
    return (pts * starters).sum(axis=1), (sds**2 * starters).sum(axis=1)


@dataclass
class Surrogate:
    """P(outcome) on a (relative mean, mean uncertainty) grid, plus interpolation."""

    deltas: np.ndarray
    sds: np.ndarray
    p_title: np.ndarray       # (n_deltas, n_sds)
    p_top2: np.ndarray
    p_playoffs: np.ndarray
    base_mean: float
    opponent_offsets: dict[str, float]
    opponent_mean_sd: float
    my_team: str
    n_sims: int

    def out_of_range(self, delta: np.ndarray, sd: np.ndarray) -> float:
        """Fraction of queries the grid has to clamp.

        Bilinear interpolation clips silently, and a clipped query does not look
        wrong — it looks like the grid edge. The first validation run scored
        0.21 absolute error on p_title purely because deltas reached +49 against
        a grid ending at +30; every sample INSIDE the grid was accurate to 0.003.
        So the clamping is now reported rather than absorbed.
        """
        d = np.asarray(delta, dtype=float)
        s = np.asarray(sd, dtype=float)
        outside = (
            (d < self.deltas[0]) | (d > self.deltas[-1])
            | (s < self.sds[0]) | (s > self.sds[-1])
        )
        return float(np.mean(outside))

    def evaluate(self, delta: np.ndarray, sd: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "p_title": _bilinear(self.deltas, self.sds, self.p_title, delta, sd),
            "p_top2": _bilinear(self.deltas, self.sds, self.p_top2, delta, sd),
            "p_playoffs": _bilinear(self.deltas, self.sds, self.p_playoffs, delta, sd),
            "clamped_fraction": np.array([self.out_of_range(delta, sd)]),
        }

    def variance_crossover(self) -> dict:
        """Where variance stops helping — §2.4's claim, with a boundary on it.

        §2.4 states without qualification that roster variance is good: you
        qualify at 67% anyway, so chase ceiling. Measured on this schedule that
        is true *up to a point* and false past it. Once you are clearly the best
        team, extra variance costs the first-round bye — the thing §2.4 itself
        calls worth about as much as everything else combined.
        """
        d_title = self.p_title[:, -1] - self.p_title[:, 0]
        d_top2 = self.p_top2[:, -1] - self.p_top2[:, 0]
        return {
            "title_crossover_delta": _crossing(self.deltas, d_title),
            "top2_crossover_delta": _crossing(self.deltas, d_top2),
            "note": (
                "Weekly points relative to the league average at which extra "
                "season-mean uncertainty stops paying. Below it, variance is "
                "worth chasing; above it, variance costs seeding. The top-2 "
                "crossover comes FIRST, which is the part §2.4 misses."
            ),
        }

    def variance_slope(self) -> pl.DataFrame:
        """§2.4's claim, as a table: what does uncertainty buy at each strength?"""
        rows = []
        for i, d in enumerate(self.deltas):
            rows.append({
                "delta_mean": round(float(d), 2),
                "p_title_low_sd": round(float(self.p_title[i, 0]), 4),
                "p_title_high_sd": round(float(self.p_title[i, -1]), 4),
                "d_title": round(float(self.p_title[i, -1] - self.p_title[i, 0]), 4),
                "d_top2": round(float(self.p_top2[i, -1] - self.p_top2[i, 0]), 4),
                "d_playoffs": round(
                    float(self.p_playoffs[i, -1] - self.p_playoffs[i, 0]), 4
                ),
            })
        return pl.DataFrame(rows)


def _crossing(xs: np.ndarray, ys: np.ndarray) -> float | None:
    """First x where ``ys`` crosses from positive to negative, interpolated."""
    for i in range(len(ys) - 1):
        if ys[i] > 0 >= ys[i + 1]:
            span = ys[i] - ys[i + 1]
            t = ys[i] / span if span else 0.0
            return round(float(xs[i] + t * (xs[i + 1] - xs[i])), 2)
    return None


def _bilinear(xs, ys, z, x, y):
    x = np.clip(np.asarray(x, dtype=float), xs[0], xs[-1])
    y = np.clip(np.asarray(y, dtype=float), ys[0], ys[-1])
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
    tx = (x - xs[i]) / (xs[i + 1] - xs[i])
    ty = (y - ys[j]) / (ys[j + 1] - ys[j])
    return (
        z[i, j] * (1 - tx) * (1 - ty)
        + z[i + 1, j] * tx * (1 - ty)
        + z[i, j + 1] * (1 - tx) * ty
        + z[i + 1, j + 1] * tx * ty
    )


def _team_inputs(
    my_team: str,
    teams: list[str],
    delta: float,
    my_sd: float,
    base_mean: float,
    opponent_offsets: dict[str, float],
    opponent_mean_sd: float,
) -> tuple[dict, dict]:
    means, msds = {}, {}
    for t in teams:
        if t == my_team:
            means[t] = base_mean + delta
            msds[t] = my_sd
        else:
            means[t] = base_mean + opponent_offsets.get(t, 0.0)
            msds[t] = opponent_mean_sd
    return means, msds


def fit(
    base_mean: float,
    opponent_offsets: dict[str, float],
    opponent_mean_sd: float,
    deltas: np.ndarray,
    sds: np.ndarray,
    my_team: str = MY_TEAM_NAME,
    season: int = SEASON,
    n_sims: int = 8_000,
    seed: int = 5,
) -> Surrogate:
    """Run the real season simulator at every grid point."""
    from ff_agent.season import schedule as SCH

    teams = SCH.games_played(season)["team"].to_list()
    if my_team not in teams:
        raise ValueError(f"{my_team!r} not in the {season} league schedule: {teams}")

    shape = (len(deltas), len(sds))
    title = np.zeros(shape)
    top2 = np.zeros(shape)
    play = np.zeros(shape)
    for i, d in enumerate(deltas):
        for j, s in enumerate(sds):
            means, msds = _team_inputs(
                my_team, teams, float(d), float(s), base_mean,
                opponent_offsets, opponent_mean_sd,
            )
            r = SIM.simulate(
                means, season=season, n_sims=n_sims, seed=seed + i * 100 + j,
                team_mean_sds=msds,
            )
            k = r.teams.index(my_team)
            title[i, j] = r.p_title[k]
            top2[i, j] = r.p_top2[k]
            play[i, j] = r.p_playoffs[k]
    return Surrogate(
        deltas=np.asarray(deltas, dtype=float), sds=np.asarray(sds, dtype=float),
        p_title=title, p_top2=top2, p_playoffs=play, base_mean=base_mean,
        opponent_offsets=opponent_offsets, opponent_mean_sd=opponent_mean_sd,
        my_team=my_team, n_sims=n_sims,
    )


def validate(
    sur: Surrogate,
    samples: list[dict],
    season: int = SEASON,
    n_sims: int = 8_000,
) -> pl.DataFrame:
    """Surrogate vs a full season simulation, on held-out draft outcomes.

    ``samples`` carry the REAL opponent means from a simulated draft, not the
    averaged profile the grid was fitted on. That is the whole point: if the
    error here is comparable to the gaps between drafting policies, then ranking
    those policies is ranking noise, and this table is what says so.
    """
    rows = []
    for n, s in enumerate(samples):
        means = dict(s["team_means"])
        msds = dict(s["team_mean_sds"])
        r = SIM.simulate(
            means, season=season, n_sims=n_sims, seed=1_000 + n, team_mean_sds=msds
        )
        k = r.teams.index(sur.my_team)
        others = [v for t, v in means.items() if t != sur.my_team]
        delta = means[sur.my_team] - float(np.mean(others))
        est = sur.evaluate(np.array([delta]), np.array([msds[sur.my_team]]))
        rows.append({
            "delta_mean": round(delta, 2),
            "my_mean_sd": round(msds[sur.my_team], 2),
            "clamped": bool(
                sur.out_of_range(np.array([delta]), np.array([msds[sur.my_team]]))
            ),
            "true_p_title": round(float(r.p_title[k]), 4),
            "surrogate_p_title": round(float(est["p_title"][0]), 4),
            "abs_err_title": round(abs(float(r.p_title[k] - est["p_title"][0])), 4),
            "true_p_top2": round(float(r.p_top2[k]), 4),
            "abs_err_top2": round(abs(float(r.p_top2[k] - est["p_top2"][0])), 4),
        })
    return pl.DataFrame(rows).sort("abs_err_title", descending=True)
