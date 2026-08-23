"""Finish an unfinished draft, so a mid-draft report is not read off phantom holes.

Analysing a draft in progress without this is actively misleading, and the
failure is loud once you see it. Team strength is the optimal starting lineup, so
a roster that has not reached the kicker yet has an EMPTY kicker slot — and every
counterfactual then discovers that the best thing any manager could have done at
any pick was take a kicker. Run against a draft 127 picks deep, the report
recommended "Texans D/ST + Brandon Aubrey" instead of nine consecutive picks.

So the remaining picks are projected. Two rules, both of them chosen to avoid
mistakes this project has already made:

* **Greedy on MARGINAL LINEUP VALUE, never on VOR.** M7: uncapped VOR takes four
  quarterbacks in the first five rounds, because VOR is measured against the
  starter replacement and prices QB3 as though he would start. Padding a roster
  by VOR would hand every team a quarterback room and no tight end.
* **Legality applies** (``policy.allowed_positions``, ESPN maxima, starter
  feasibility and §3.5's late K/D-ST), and picks are consumed in real snake
  order, so two managers cannot be projected the same player.

This is a model of what the managers will do, not a prediction of what they will
do, and it is only ever applied to picks that have not happened. On a completed
draft this module is a no-op.
"""

from __future__ import annotations

import numpy as np

from ff_agent.config import MY_GAME_WEEKS
from ff_agent.draft import engine as E
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.draft import surrogate as SUR
from ff_agent.postdraft.results import DraftRecord


def project_remaining(
    record: DraftRecord, play_weeks: dict[str, tuple[int, ...]] | None = None
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Every manager's roster extended to ``rounds``, plus how many were projected.

    Returns rosters keyed by manager. On a complete draft the rosters are the
    real ones and every count is zero.
    """
    real = {m: list(record.roster_indices(m)) for m in record.managers}
    projected = {m: 0 for m in record.managers}
    if record.complete:
        return real, projected

    pool = record.pool
    weeks = play_weeks or {}
    avail = np.ones(pool.n, dtype=bool)
    for idx in real.values():
        for i in idx:
            avail[i] = False

    counts = {
        m: np.zeros((1, len(PL.POSITIONS)), dtype=np.int16) for m in record.managers
    }
    for m, idx in real.items():
        for i in idx:
            counts[m][0, pool.pos_idx[i]] += 1

    order = E.snake_order(record.n_teams, record.rounds)
    for k in range(record.n_picks, record.total_picks):
        m = record.managers[int(order[k])]
        round_no = k // record.n_teams + 1
        choice = _best_addition(
            pool, np.array(real[m], dtype=np.int64), avail, counts[m],
            round_no, record.rounds, tuple(weeks.get(m, MY_GAME_WEEKS)),
        )
        if choice is None:
            continue
        real[m].append(choice)
        projected[m] += 1
        avail[choice] = False
        counts[m][0, pool.pos_idx[choice]] += 1
    return real, projected


def _best_addition(
    pool, roster: np.ndarray, avail: np.ndarray, counts: np.ndarray,
    round_no: int, rounds: int, weeks: tuple[int, ...],
) -> int | None:
    """The available player who most improves this roster's starting lineup.

    Only the best legal candidate at each POSITION is tried. Within a position
    the ordering is VOR, which is safe there — VOR misranks a quarterback
    against a running back, not one quarterback against another — and it keeps
    this to six lineup solves per projected pick instead of four hundred.
    """
    allowed = POL.allowed_positions(counts, round_no, rounds)[0]
    legal = avail & allowed[pool.pos_idx]
    if not legal.any():
        return None

    cands = []
    for p in range(len(PL.POSITIONS)):
        at_p = np.flatnonzero(legal & (pool.pos_idx == p))
        if at_p.size:
            cands.append(int(at_p[np.argmax(pool.vor[at_p])]))
    if not cands:
        return None

    rosters = np.array(
        [np.concatenate([roster, [c]]) for c in cands], dtype=np.int64
    )
    vals = SUR.lineup_stats(pool, rosters, play_weeks=weeks).mean
    return cands[int(np.argmax(vals))]
