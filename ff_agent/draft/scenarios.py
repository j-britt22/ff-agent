"""QB scenarios — the axis the whole draft turns on.

M7's gate found the quarterback timing is the least trustworthy part of the
opponent model, and the 2026 plans show exactly why that matters. Under the
shipped positional offsets the simulator says **Lamar Jackson is 93% available
at pick 14**. If that is right it decides the draft. If the model under-drafts
quarterbacks, it is a trap that costs the position entirely.

The evidence says it probably does under-draft them:

    simulated 2026        first QB pick 6 [1, 14],  3 QBs in rounds 1-3
    2025 actual           first QB pick 7,          5 QBs in 3 rounds
    2025 scaled to 9 teams                        ~5.6

Timing is right; volume is understated. The shipped offsets carry
``FORMAT_MATCH_WEIGHT = 3`` on 2025, which is 60% of the evidence — the other
40% is one-QB seasons, and it drags the quarterback rate down.

There is no way to fix that with more data; the league has run exactly one 2-QB
draft. So instead of picking one number and hoping, every plan is built under
three scenarios and the plan reports where the recommendation CHANGES. A
recommendation that survives all three is a plan. One that does not is a bet,
and the sheet should say which one you are holding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ff_agent.draft import calibrate as CB
from ff_agent.draft import pool as PL
from ff_agent.opponents import history as H

SCENARIOS = ("qb_cold", "shipped", "qb_hot")

DESCRIPTIONS = {
    "qb_cold": (
        "Positional mix fitted on 2023-24 only — both ONE-QB seasons. "
        "Quarterbacks go late and stay available. The pessimistic case for "
        "waiting being punished, and the optimistic case for QB value."
    ),
    "shipped": (
        "All three seasons, format-weighted (2025 at 3x). What draft_sim.json "
        "ships. Reproduces 2025's first-QB timing but understates QB VOLUME."
    ),
    "qb_hot": (
        "Fitted on 2025 alone — the only 2-QB draft this league has run. "
        "Thin evidence (one draft, 8 teams) but the only format-matched "
        "evidence there is. Quarterbacks go early and do not come back."
    ),
}


@dataclass
class Scenario:
    name: str
    offsets: np.ndarray
    description: str
    trace: pl.DataFrame


def fit_all(
    pool: PL.DraftPool,
    managers: list[str],
    tilt: dict,
    reach: dict,
    rounds: int,
    season: int = 2026,
    n_sims: int = 1_500,
) -> dict[str, Scenario]:
    """Fit the three offset sets. ~17s each, done once per plan build."""
    prior = tuple(s for s in H.HISTORY_SEASONS if s < 2025)
    targets = {
        "qb_cold": CB.target_rates(seasons=prior, target_two_qb=True),
        "shipped": CB.target_rates(target_two_qb=True),
        "qb_hot": CB.target_rates(seasons=(2025,), target_two_qb=True),
    }
    out = {}
    for name, target in targets.items():
        off, trace = CB.fit_offsets(
            pool, managers, tilt, reach, target, rounds, n_sims=n_sims
        )
        out[name] = Scenario(name, off, DESCRIPTIONS[name], trace)
    return out


OFFSETS_FILE = "scenario_offsets.npz"


def save(scen: dict[str, Scenario], artifacts_dir) -> str:
    """Persist the fitted offsets. Fitting takes ~45s; draft day cannot pay it."""
    path = artifacts_dir / OFFSETS_FILE
    np.savez(path, **{k: v.offsets for k, v in scen.items()})
    return str(path)


def load(artifacts_dir) -> dict[str, Scenario] | None:
    """Read the cached offsets, or None if they were never built."""
    path = artifacts_dir / OFFSETS_FILE
    if not path.exists():
        return None
    z = np.load(path)
    return {
        k: Scenario(k, z[k], DESCRIPTIONS[k], pl.DataFrame())
        for k in SCENARIOS if k in z
    }


def qb_pressure(scen: dict[str, Scenario]) -> pl.DataFrame:
    """The early-QB multiplier each scenario applies, side by side."""
    qi = PL.POS_INDEX["QB"]
    rows = []
    for name in SCENARIOS:
        s = scen[name]
        rows.append({
            "scenario": name,
            "early_qb_multiplier": round(float(np.exp(s.offsets[0, qi])), 3),
            "middle_qb_multiplier": round(float(np.exp(s.offsets[1, qi])), 3),
            "late_qb_multiplier": round(float(np.exp(s.offsets[2, qi])), 3),
        })
    return pl.DataFrame(rows)


def divergent_turns(per_scenario: dict[str, list[dict]]) -> pl.DataFrame:
    """Turns where the scenarios disagree about what to take.

    This is the actionable output. Where all three agree, the plan is robust and
    the sheet can state it flatly. Where they disagree, the sheet has to say so
    — and say which observable at the table tells you which world you are in.
    """
    rows = []
    n = len(per_scenario["shipped"])
    for i in range(n):
        calls = {
            name: turns[i]["recommended"]["modal_position"]
            for name, turns in per_scenario.items()
        }
        agree = len(set(calls.values())) == 1
        rows.append({
            "pick": per_scenario["shipped"][i]["pick"],
            "round": per_scenario["shipped"][i]["round"],
            **{f"take_if_{k}": v for k, v in calls.items()},
            "robust": agree,
        })
    return pl.DataFrame(rows)
