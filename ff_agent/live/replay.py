"""MILESTONE 9 GATE (§11 step 9): "full mock, sub-second, manual entry works
with WiFi off, --slot N loads in under 5 seconds."

Replays the real 2025 draft through the live loop, pick by pick, exactly as it
would run on draft day: every pick goes in through the manual-entry resolver by
NAME, and every one of my turns pays for a real recommendation.

What this gate is and is not. It measures the **mechanics under a clock** —
latency, name resolution, roster legality, logging — not predictive accuracy;
M7's backtest already grades the model. So it deliberately uses a pool that
contains every player who was actually drafted. On draft day the pool is the
2026 board and that question does not arise; here it keeps the test about the
loop rather than about coverage.

The measurement that matters is **p95 latency at my own turns**, because that is
the number that either fits inside ESPN's clock or does not.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from ff_agent.draft import backtest as BT
from ff_agent.draft import pool as PL
from ff_agent.live import advise as AD
from ff_agent.live import log as LG
from ff_agent.live.entry import Resolver
from ff_agent.live.loop import handle
from ff_agent.live.state import DraftState

SEASON = 2025
MY_MANAGER = "Jordan Britt"


def replay_pool(season: int = SEASON, n_teams: int = 8) -> PL.DraftPool:
    """The backtest pool, widened to contain everyone who was actually taken.

    See the module docstring: this gate is about the loop, not about whether the
    board would have found them.
    """
    from ff_agent.board import build as BB

    board = BB.build(season)
    actual = BT.actual_picks(season)
    keep = set(actual["canonical_id"].to_list())
    top = (
        board.sort("vor", descending=True, nulls_last=True)
        .head(BT.POOL_TOP_N)["canonical_id"].to_list()
    )
    wanted = keep | set(top)
    sub = board.filter(pl.col("canonical_id").is_in(list(wanted)))

    # All 32 defenses and a deep kicker list: the drafted ones are picked by
    # ownership/production rank, and the top 20 does not cover everyone this
    # league actually took. Missing them left my replayed roster with no D/ST
    # at all, which is how the gap announced itself.
    k_before, d_before = PL.N_KICKERS, PL.N_DEFENSES
    PL.N_KICKERS, PL.N_DEFENSES = 40, 32
    try:
        return PL.build_pool(
            season=season, n_teams=n_teams, board=sub, kdst_source="prior_season"
        )
    finally:
        PL.N_KICKERS, PL.N_DEFENSES = k_before, d_before


def run(
    season: int = SEASON,
    budget: float = AD.LATENCY_BUDGET,
    write_log: bool = True,
) -> dict:
    managers = BT.draft_order(season)
    pool = replay_pool(season, len(managers))
    my_index = managers.index(MY_MANAGER)
    rounds = int(BT.actual_picks(season)["round"].max())

    state = DraftState(pool=pool, managers=managers, my_index=my_index, rounds=rounds)
    resolver = Resolver(pool)
    tilt, reach, scen, targets = AD.load_context(season, pool, managers, rounds)

    log = LG.SessionLog(LG.session_path(season, my_index + 1)) if write_log else None

    ids = pool.df["canonical_id"].to_list()
    by_id = {c: i for i, c in enumerate(ids)}
    actual = BT.actual_picks(season).sort("overall_pick").to_dicts()

    latencies, unresolved, advised = [], [], 0
    for row in actual:
        if state.my_turn:
            t0 = time.perf_counter()
            a = AD.advise(state, tilt, reach, scen, targets, budget=budget)
            latencies.append(time.perf_counter() - t0)
            advised += 1
            if log:
                log.recommendation(a, state)

        # the human types the name — the same path draft day uses
        reply = handle(state, resolver, row["name"])
        if reply.recorded is None:
            target = by_id.get(row["canonical_id"])
            if target is None:
                unresolved.append({"pick": row["overall_pick"], "name": row["name"],
                                   "why": "not in pool"})
                continue
            unresolved.append({"pick": row["overall_pick"], "name": row["name"],
                               "why": reply.lines[0][:80]})
            state.record(target)
        if log:
            state_team = managers[int(state.order[len(state.history) - 1])]
            log.pick(state, state.history[-1], state_team, "manual")

    counts = state.position_counts()
    lat = np.array(latencies) if latencies else np.zeros(1)
    return {
        "season": season,
        "n_teams": len(managers),
        "rounds": rounds,
        "picks_replayed": len(state.history),
        "actual_picks": len(actual),
        "pool_size": pool.n,
        "my_slot": my_index + 1,
        "turns_advised": advised,
        "latency": {
            "budget": budget,
            "p50": round(float(np.percentile(lat, 50)), 3),
            "p95": round(float(np.percentile(lat, 95)), 3),
            "max": round(float(lat.max()), 3),
            "over_budget": int((lat > budget * 1.5).sum()),
        },
        "entry": {
            "resolved_by_name": len(actual) - len(unresolved),
            "needed_fallback": len(unresolved),
            "examples": unresolved[:5],
        },
        "my_roster": {
            "size": len(state.my_roster()),
            "positions": counts,
            "legal": _legal(counts),
        },
        "alarms": state.alarms(),
        "log": str(log.path) if log else None,
    }


def _legal(counts: dict[str, int]) -> bool:
    need = {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "DST": 1, "K": 1}
    if any(counts.get(p, 0) < n for p, n in need.items()):
        return False
    return sum(counts.get(p, 0) for p in ("RB", "WR", "TE")) >= 6
