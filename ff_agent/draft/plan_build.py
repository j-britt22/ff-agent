"""plan_1..9.json (§5, §11 step 8) — the nine precomputed slot plans.

§5's whole design constraint: I learn my slot one hour before the draft, and
that hour must need **zero computation**. So everything expensive happens here,
and `draft --slot N` is a file read.

Each plan is built under the shipped positional offsets and then re-checked
under the two QB scenarios in ``scenarios.py``, because M7's gate showed
quarterback timing is the least reliable thing in the opponent model and the
2026 plans hinge on it. Turns where all three scenarios agree are marked robust;
turns where they disagree carry the disagreement onto the sheet rather than
picking one and hoping.

The §11 test — "plans differ meaningfully; slot 5 shouldn't read like slot 1" —
is measured against re-seed noise, not against zero. Nine Monte Carlo plans
always differ a little.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from ff_agent.config import ARTIFACTS_DIR, N_TEAMS, ROSTER_TOTAL, SEASON
from ff_agent.draft import build as DB
from ff_agent.draft import plans as PN
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.draft import qbcount as QC
from ff_agent.draft import scenarios as SC
from ff_agent.opponents import pickmodel as PM
from ff_agent.opponents import tendencies as TD

SHIPPED_SIMS = 10_000
"""§5: "Run 10,000+ simulated drafts per slot"."""

SCENARIO_SIMS = 5_000
"""The alternates only have to detect where the recommendation flips."""

RESEED = 20_260_821
"""Seed for the within-slot noise measurement. Must differ from the shipped
runs' seed, which ``run_slot`` derives from the slot number."""


def _lookups():
    return PM.build_lookups(
        TD.picks_with_consensus().filter(pl.col("canonical_id").is_not_null())
    )


def build_all(
    season: int = SEASON,
    n_sims: int = SHIPPED_SIMS,
    scenario_sims: int = SCENARIO_SIMS,
    qb_cap: int | None = None,
    grid_sims: int = 8_000,
) -> dict:
    DB.assert_teams_match_schedule(season)
    pool = PL.build_pool(season, N_TEAMS)
    PL.save_pool(pool, ARTIFACTS_DIR)     # M9's live loop reads this (§5, offline)
    tilt, reach = _lookups()
    managers = DB.managers_for_slot(1)
    scen = SC.fit_all(pool, managers, tilt, reach, ROSTER_TOTAL, season)
    SC.save(scen, ARTIFACTS_DIR)   # M9 loads these instead of refitting (~45s)

    # the surrogate and the noise-exploitation constant both come from M7
    edge = DB.decompose_edge(pool, tilt, reach, scen["shipped"].offsets, season=season)
    noise = float(edge["noise_exploitation"][0])

    def policy_for(cap):
        p = POL.NeedWeighted(caps={"QB": cap} if cap else None)
        p.name = f"need_weighted_qb{cap}" if cap else "need_weighted"
        return p

    # ── pass 1: the shipped runs, which also size the surrogate grid ─────────
    shipped_runs = {
        slot: DB.run_slot(pool, slot, policy_for(qb_cap), tilt, reach,
                          scen["shipped"].offsets, n_sims=n_sims, season=season)
        for slot in range(1, N_TEAMS + 1)
    }
    sur = DB.fit_surrogate(list(shipped_runs.values()), season=season, n_sims=grid_sims)
    crossover = sur.variance_crossover()["title_crossover_delta"]

    # ── the QB-count question M4 and M7 disagreed about ─────────────────────
    qb_sweep = QC.sweep(pool, tilt, reach, scen["shipped"].offsets, sur, noise,
                        season=season)
    qb_decision = QC.decide(qb_sweep, crossover)
    chosen_cap = qb_cap or qb_decision["decision"]

    if qb_cap is None and chosen_cap:
        shipped_runs = {
            slot: DB.run_slot(pool, slot, policy_for(chosen_cap), tilt, reach,
                              scen["shipped"].offsets, n_sims=n_sims, season=season)
            for slot in range(1, N_TEAMS + 1)
        }

    slot_plans = {
        slot: PN.build_slot_plan(r.result, slot, r.value.mean)
        for slot, r in shipped_runs.items()
    }

    # ── pass 2: the same slots under the two QB scenarios ───────────────────
    divergence_by_slot = {}
    for slot in range(1, N_TEAMS + 1):
        per_scenario = {"shipped": slot_plans[slot].per_turn}
        for name in ("qb_cold", "qb_hot"):
            r = DB.run_slot(pool, slot, policy_for(chosen_cap), tilt, reach,
                            scen[name].offsets, n_sims=scenario_sims, season=season)
            per_scenario[name] = PN.build_slot_plan(r.result, slot, r.value.mean).per_turn
        divergence_by_slot[slot] = SC.divergent_turns(per_scenario)

    # ── §11 step 8: between-slot signal against within-slot noise ───────────
    noise_pairs = []
    for slot in (1, 5, 9):
        a = slot_plans[slot]
        # A DIFFERENT SEED, or this measures nothing: run_slot derives its
        # default seed from the slot, so re-running it reproduces the same
        # draft exactly and the gate passes with a ratio of infinity.
        b = PN.build_slot_plan(
            DB.run_slot(pool, slot, policy_for(chosen_cap), tilt, reach,
                        scen["shipped"].offsets, n_sims=n_sims, season=season,
                        seed=RESEED).result,
            slot, shipped_runs[slot].value.mean,
        )
        noise_pairs.append((a, b))
    gate = PN.plans_differ(slot_plans, noise_pairs)

    shared = {
        "season": season,
        "n_sims_per_slot": n_sims,
        "policy": policy_for(chosen_cap).name,
        "qb_count_decision": qb_decision,
        "qb_cap_sweep": qb_sweep.to_dicts(),
        "qb_scenarios": {
            "pressure": SC.qb_pressure(scen).to_dicts(),
            "descriptions": SC.DESCRIPTIONS,
        },
        "variance_crossover": sur.variance_crossover(),
        "edge_decomposition": edge.to_dicts(),
        "plans_differ": gate,
        "caveats": {
            "expressed_in": (
                "Plans are stated in availability probability and positional "
                "shape, which M7's gate calibrated. Absolute title probabilities "
                "are inflated by the noise-exploitation term and are only used "
                "here for RANKING options against each other."
            ),
            "qb_timing": (
                "The single biggest uncertainty. The shipped model reproduces "
                "2025's first-QB timing but understates QB VOLUME (3 in rounds "
                "1-3 against ~5.6 scaled from 2025). Where the three scenarios "
                "disagree, the sheet says so."
            ),
        },
    }
    return {"shared": shared, "plans": slot_plans, "divergence": divergence_by_slot}


def write(season: int = SEASON, **kw) -> list[str]:
    """One self-contained file per slot, plus the shared index."""
    out = build_all(season, **kw)
    shared, plans, div = out["shared"], out["plans"], out["divergence"]
    paths = []
    for slot, plan in plans.items():
        d = plan.to_dict()
        d["shared"] = shared
        d["scenario_divergence"] = div[slot].to_dicts()
        p = ARTIFACTS_DIR / f"plan_{slot}.json"
        p.write_text(json.dumps(d, indent=2, default=str))
        paths.append(str(p))
    idx = ARTIFACTS_DIR / "plans_index.json"
    idx.write_text(json.dumps({
        **shared,
        "slots": {
            str(s): {
                "pick_numbers": p.turns,
                "snake_gaps": sorted(set(np.diff(p.turns).tolist())),
                "roster_strength": p.roster_strength,
                "robust_turns": int(
                    sum(1 for r in div[s].to_dicts() if r["robust"])
                ),
                "total_turns": len(p.turns),
            }
            for s, p in plans.items()
        },
    }, indent=2, default=str))
    paths.append(str(idx))
    return paths


def load(slot: int) -> dict:
    """The T-60 path: read one file, compute nothing (§5)."""
    p = ARTIFACTS_DIR / f"plan_{slot}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Run `uv run python -m ff_agent.cli plans` "
            "BEFORE draft day — this command is a file read by design and will "
            "not compute a plan while you are on the clock."
        )
    return json.loads(p.read_text())
