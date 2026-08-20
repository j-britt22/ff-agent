"""P(player taken | pick, manager, board state) — what M7's simulator consumes.

Structure follows §7.5's instruction directly: superflex ADP is the backbone and
manager tendencies are adjustments on top, because the history is thin. Three
terms:

1. **ADP proximity** — players go near their consensus rank; the drift term is
   how far ahead of consensus this manager habitually reaches.
2. **Positional tilt** — this manager's shrunk P(position | phase) against the
   league's.
3. **Run pressure** — §3.1's named residual risk: "once two or three managers
   grab their second QB, the rest stampede." Picks are not independent.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ff_agent.opponents import tendencies as TD

ADP_SIGMA = 0.055
"""Width of the ADP proximity kernel, as a fraction of the draft. Roughly one
round in a 9-team, 17-round draft."""

RUN_BOOST = 0.35
"""Extra log-weight on a position when a run is under way. §3.1 names the QB
stampede as the specific risk in an 18-starter league."""

RUN_WINDOW = 8
"""Picks looked back over to detect a run."""


def _tilt_lookup(tilt: pl.DataFrame) -> dict:
    return {
        (r["manager"], r["phase"], r["position"]): (r["rate"], r["league_rate"])
        for r in tilt.to_dicts()
    }


def _phase_for(frac: float) -> str:
    for name, lo, hi in TD.PHASES:
        if lo <= frac < hi:
            return name
    return "late"


def pick_probabilities(
    available: pl.DataFrame,
    pick_fraction: float,
    manager: str,
    tilt_lookup: dict,
    reach: dict[str, float],
    recent_positions: list[str] | None = None,
    adp_sigma: float = ADP_SIGMA,
) -> np.ndarray:
    """Softmax over available players for one pick.

    ``available`` needs ``consensus_fraction`` and ``position``.
    """
    cf = available["consensus_fraction"].fill_null(1.5).to_numpy().astype(float)
    pos = available["position"].fill_null("NA").to_list()

    target = pick_fraction + reach.get(manager, 0.0)
    util = -np.abs(cf - target) / adp_sigma

    phase = _phase_for(pick_fraction)
    tilt = np.zeros(len(pos))
    for i, p in enumerate(pos):
        rate, lg = tilt_lookup.get((manager, phase, p), (None, None))
        if rate and lg and lg > 0:
            tilt[i] = np.log(max(rate, 1e-6) / lg)
    util += tilt

    if recent_positions:
        recent = recent_positions[-RUN_WINDOW:]
        for p in set(recent):
            share = recent.count(p) / len(recent)
            if share >= 0.4:                       # a genuine run, not noise
                util += RUN_BOOST * np.array([1.0 if q == p else 0.0 for q in pos])

    util -= util.max()
    w = np.exp(util)
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1.0 / max(len(w), 1))


def build_lookups(train: pl.DataFrame) -> tuple[dict, dict[str, float]]:
    """Fit the manager terms on a training slice of history."""
    tilt = TD.positional_tendency(train)
    reach_df = TD.reach_profile(train)
    reach = dict(zip(reach_df["manager"].to_list(), reach_df["mean_reach"].to_list()))
    return _tilt_lookup(tilt), reach
