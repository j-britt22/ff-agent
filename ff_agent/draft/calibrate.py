"""League-level positional calibration — the term M5's model does not have.

M5 builds per-manager positional tilt as ``log(manager rate / league rate)``.
That is the right shape for saying *who* differs from the league, and it is
structurally incapable of saying how the league differs from ADP: the tilts are
measured against the league's own rate, so across eight managers they very
nearly cancel. Run forward, the simulated positional mix is therefore whatever
superflex ECR implies — nothing else in the model can move it.

The M7 gate found this immediately. Simulating the 2025 draft off the shipped
model took the first QB at pick **1** (median) and **10** quarterbacks inside
three rounds. The league actually took the first at pick 7, and five. ECR is not
wrong; it is national, and this league is not the nation.

So: one (phase x position) log-offset, fitted by simulation until the simulated
mix matches the measured mix. Iterative proportional fitting, six passes, damped.

**Fitting these offsets makes gate measure (a) — positional composition — true
by construction.** That is stated loudly rather than quietly banked: once
offsets are fitted, (a) stops being an independent test and the load falls on
(b) QB timing detail and (c) per-player calibration, which the offsets do not
directly target.

The seasons the offsets are fitted on matter more than usual here, because the
league changed format underneath itself. Fitting on 2023-24 to predict 2025
means fitting a ONE-QB positional mix to predict a TWO-QB draft, and it will be
wrong in a knowable direction. That is not a defect of the method — it is M5's
central finding arriving from a second direction.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ff_agent.draft import engine as E
from ff_agent.draft import pool as PL
from ff_agent.opponents import history as H
from ff_agent.opponents import tendencies as TD

N_PHASES = len(TD.PHASES)
DAMPING = 0.7
"""Step size on each fitting pass. Full steps oscillate; this converges."""

ITERATIONS = 6
MIN_RATE = 1e-3
"""Floor on a target rate, so a position absent from three drafts becomes very
unlikely rather than impossible."""


def target_rates(
    seasons: tuple[int, ...] | None = None,
    target_two_qb: bool = True,
    hist: pl.DataFrame | None = None,
) -> np.ndarray:
    """(n_phases, n_positions) measured P(position | phase) for this league.

    Seasons are weighted by ``tendencies.season_weight``, so a format-matched
    season counts triple — the same rule M5 uses, for the same reason.
    """
    h = H.draft_history() if hist is None else hist
    if seasons is not None:
        h = h.filter(pl.col("season").is_in(list(seasons)))
    if h.is_empty():
        raise ValueError("no draft history to calibrate against")

    h = h.with_columns(
        pl.col("season")
        .map_elements(lambda s: TD.season_weight(s, target_two_qb), return_dtype=pl.Float64)
        .alias("w")
    )
    out = np.zeros((N_PHASES, len(PL.POSITIONS)), dtype=np.float64)
    for pi, (_, lo, hi) in enumerate(TD.PHASES):
        seg = h.filter(
            (pl.col("pick_fraction") >= lo) & (pl.col("pick_fraction") < hi)
        )
        tot = float(seg["w"].sum()) or 1.0
        for qi, pos in enumerate(PL.POSITIONS):
            out[pi, qi] = float(seg.filter(pl.col("position") == pos)["w"].sum()) / tot
    out = np.maximum(out, MIN_RATE)
    return out / out.sum(axis=1, keepdims=True)


def simulated_rates(res: E.DraftResult) -> np.ndarray:
    """(n_phases, n_positions) positional mix the simulator actually produced."""
    total = res.picks.shape[1]
    pf = (np.arange(total) + 1) / total
    phase = np.array([E._phase_index(float(f)) for f in pf])
    pos = res.pool.pos_idx[res.picks]

    out = np.zeros((N_PHASES, len(PL.POSITIONS)), dtype=np.float64)
    for pi in range(N_PHASES):
        cols = np.where(phase == pi)[0]
        if cols.size == 0:
            out[pi] = 1.0 / len(PL.POSITIONS)
            continue
        seg = pos[:, cols].ravel()
        counts = np.bincount(seg, minlength=len(PL.POSITIONS)).astype(float)
        out[pi] = counts / max(counts.sum(), 1.0)
    return np.maximum(out, 1e-9)


def fit_offsets(
    pool: PL.DraftPool,
    managers: list[str],
    tilt_lookup: dict,
    reach: dict[str, float],
    target: np.ndarray,
    rounds: int,
    n_sims: int = 1_500,
    iterations: int = ITERATIONS,
    damping: float = DAMPING,
    seed: int = 909,
    adp_sigma: float | None = None,
) -> tuple[np.ndarray, pl.DataFrame]:
    """Iteratively nudge the offsets until the simulated mix matches ``target``.

    Returns the offsets and a convergence trace, because a fitted correction that
    nobody can see converge is a fudge factor.
    """
    from ff_agent.opponents import pickmodel as PM

    sigma = PM.ADP_SIGMA if adp_sigma is None else adp_sigma
    offsets = np.zeros_like(target)
    trace = []
    for it in range(iterations):
        res = E.simulate_drafts(
            pool, managers, my_index=-1, policy=None, tilt_lookup=tilt_lookup,
            reach=reach, n_sims=n_sims, rounds=rounds, seed=seed + it,
            adp_sigma=sigma, pos_offsets=offsets,
        )
        sim = simulated_rates(res)
        err = float(np.abs(sim - target).sum(axis=1).max())
        trace.append({"iteration": it, "max_total_variation": round(err, 4)})
        offsets = offsets + damping * np.log(target / sim)
    return offsets, pl.DataFrame(trace)


def offsets_table(offsets: np.ndarray) -> pl.DataFrame:
    """The fitted correction, readable. Positive = this league takes MORE."""
    rows = []
    for pi, (ph, _, _) in enumerate(TD.PHASES):
        for qi, pos in enumerate(PL.POSITIONS):
            rows.append({
                "phase": ph, "position": pos,
                "log_offset": round(float(offsets[pi, qi]), 3),
                "multiplier": round(float(np.exp(offsets[pi, qi])), 3),
            })
    return pl.DataFrame(rows).sort(["phase", "position"])
