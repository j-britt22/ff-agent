"""How many quarterbacks? M4 said two, M7 said three. M8 has to commit.

The two answers were never comparable, which is why they disagreed:

* **M4** (`board/review.py`) assumed I get the top two QBs on the board (Allen
  bye wk 7, Jackson wk 13, neither covered by my weeks 5/14), valued QB3 at the
  weeks a starter is actually out — **17.9 points** — and compared that against
  the best non-QB stash's **26.4 points of upside**, which is `ceiling - mean`.
  So it compared an EXPECTED value against a VARIANCE measure and concluded two.
* **M7** simulated the quarterbacks I actually end up with and scored expected
  weekly points, where a cap of three beat two by +1.9 and four by +1.4.

Both are half the answer, because §8 says the objective is championship
probability and that combines mean AND variance. M7's own surrogate measures
exactly that surface, so the resolution is to evaluate both caps on it.

One correction is essential first. M7's edge decomposition showed that ~31 of
the ~32 weekly points a policy shows over the league is **noise exploitation** —
bought from M5's opponents drafting with an 8-pick spread, not from anything
that happens on draft day. That matters here more than anywhere else, because
the variance term changes sign at delta +15.9: read at the inflated delta the
model says variance is bad, and read at the honest delta it says variance is
good. The QB3-versus-stash question turns on precisely that. So both the raw and
the corrected reading are reported, and the corrected one is the answer.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ff_agent.draft import build as DB
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.draft import surrogate as SUR

CAPS = (2, 3, 4)


def sweep(
    pool: PL.DraftPool,
    tilt: dict,
    reach: dict,
    offsets: np.ndarray,
    sur: SUR.Surrogate,
    noise_exploitation: float,
    slots: tuple[int, ...] = (1, 5, 9),
    caps: tuple[int, ...] = CAPS,
    base_policy=POL.NeedWeighted,
    label: str = "need_weighted",
    n_sims: int = 6_000,
    season: int = 2026,
) -> pl.DataFrame:
    """P(title) by QB cap, at the raw delta and at the honest one."""
    rows = []
    for slot in slots:
        for cap in caps:
            pol = base_policy(caps={"QB": cap})
            pol.name = f"{label}_qb{cap}"
            r = DB.run_slot(pool, slot, pol, tilt, reach, offsets,
                            n_sims=n_sims, season=season)
            raw = r.delta
            honest = raw - noise_exploitation
            sd = r.value.mean_sd
            p_raw = sur.evaluate(raw, sd)
            p_hon = sur.evaluate(honest, sd)
            pos = np.array(pool.df["position"].to_list())
            mine = pos[r.result.my_roster()]
            rows.append({
                "slot": slot,
                "qb_cap": cap,
                "mean_weekly_points": round(float(r.value.mean.mean()), 2),
                "delta_raw": round(float(raw.mean()), 2),
                "delta_honest": round(float(honest.mean()), 2),
                "mean_sd": round(float(sd.mean()), 3),
                "p_title_raw_delta": round(float(p_raw["p_title"].mean()), 4),
                "p_title_honest_delta": round(float(p_hon["p_title"].mean()), 4),
                "p_top2_honest_delta": round(float(p_hon["p_top2"].mean()), 4),
                "n_RB": round(float((mine == "RB").sum(axis=1).mean()), 2),
                "n_WR": round(float((mine == "WR").sum(axis=1).mean()), 2),
                "n_TE": round(float((mine == "TE").sum(axis=1).mean()), 2),
            })
    return pl.DataFrame(rows)


def decide(sweep_df: pl.DataFrame, crossover: float | None) -> dict:
    """Commit to a number, and say what would change it."""
    by_cap = (
        sweep_df.group_by("qb_cap")
        .agg(
            pl.col("p_title_honest_delta").mean().alias("p_title"),
            pl.col("p_title_raw_delta").mean().alias("p_title_inflated"),
            pl.col("mean_weekly_points").mean().alias("weekly"),
            pl.col("mean_sd").mean().alias("mean_sd"),
            pl.col("delta_honest").mean().alias("delta_honest"),
        )
        .sort("p_title", descending=True)
    )
    best = by_cap.row(0, named=True)
    second = by_cap.row(1, named=True)
    margin = float(best["p_title"] - second["p_title"])

    inflated_order = by_cap.sort("p_title_inflated", descending=True)["qb_cap"].to_list()
    honest_order = by_cap["qb_cap"].to_list()

    return {
        "decision": int(best["qb_cap"]),
        "margin_over_runner_up": round(margin, 4),
        "runner_up": int(second["qb_cap"]),
        "ranking_at_honest_delta": honest_order,
        "ranking_at_inflated_delta": inflated_order,
        "correction_changes_the_answer": honest_order[0] != inflated_order[0],
        "variance_crossover_delta": crossover,
        "my_honest_delta": round(float(best["delta_honest"]), 2),
        "above_crossover": (
            None if crossover is None else bool(best["delta_honest"] > crossover)
        ),
        "by_cap": by_cap.to_dicts(),
        "note": (
            "M4 said TWO by comparing QB3's expected bye coverage (17.9 pts) "
            "against a stash's ceiling-minus-mean (26.4) — an expected value "
            "against a variance measure. M7 said THREE on expected points only. "
            "This scores both on P(title), which is what §8 actually asks for, "
            "at the delta corrected for M7's measured noise-exploitation term. "
            "Whether variance helps or hurts flips at the crossover, and that is "
            "exactly the quantity the two earlier answers each had half of."
        ),
    }
