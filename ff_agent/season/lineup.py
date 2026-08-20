"""Optimal starting lineup (§9.2).

§9.2 warns that "greedy slot-filling gets the FLEX wrong". The failure it names
is filling the FLEX before the strict slots, or filling slots in a fixed order.

Filling every STRICT slot first with the best available at that position, then
taking the FLEX from whatever remains, is optimal for this slot structure — the
FLEX accepts a superset (RB/WR/TE) of what the strict slots accept, so moving a
player out of his strict slot into the FLEX only forces a worse player into the
slot he vacated, and the FLEX could have taken that worse player anyway.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import STARTER_SLOTS

FLEX_ELIGIBLE = ("RB", "WR", "TE")
STRICT_SLOTS = {k: v for k, v in STARTER_SLOTS.items() if k != "FLEX"}


def optimal_lineup(players: pl.DataFrame, value: str = "weekly_points") -> pl.DataFrame:
    """Pick the starting lineup from a roster. Returns the starters with slots."""
    pool = players.sort(value, descending=True, nulls_last=True)
    chosen, used = [], set()

    for pos, n in STRICT_SLOTS.items():
        take = (
            pool.filter(pl.col("position") == pos)
            .filter(~pl.col("canonical_id").is_in(list(used)) if used else pl.lit(True))
            .head(n)
        )
        for row in take.to_dicts():
            used.add(row["canonical_id"])
            chosen.append({**row, "slot": pos})

    for _ in range(STARTER_SLOTS.get("FLEX", 0)):
        take = (
            pool.filter(pl.col("position").is_in(list(FLEX_ELIGIBLE)))
            .filter(~pl.col("canonical_id").is_in(list(used)) if used else pl.lit(True))
            .head(1)
        )
        for row in take.to_dicts():
            used.add(row["canonical_id"])
            chosen.append({**row, "slot": "FLEX"})

    if not chosen:
        return pl.DataFrame(schema={**players.schema, "slot": pl.Utf8})
    return pl.DataFrame(chosen)


def lineup_points(players: pl.DataFrame, value: str = "weekly_points") -> float:
    lu = optimal_lineup(players, value)
    return float(lu[value].sum()) if lu.height else 0.0
