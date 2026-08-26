"""Optimal starting lineup (§9.2).

§9.2 warns that "greedy slot-filling gets the FLEX wrong". The failure it names
is filling the FLEX before the strict slots, or filling slots in a fixed order.

Filling every STRICT slot first with the best available at that position, then
taking the FLEX from whatever remains, is optimal for this slot structure — the
FLEX accepts a superset (RB/WR/TE) of what the strict slots accept, so moving a
player out of his strict slot into the FLEX only forces a worse player into the
slot he vacated, and the FLEX could have taken that worse player anyway.

**Pinning (M10b).** ESPN locks each player at HIS OWN kickoff, so from the moment
Thursday night kicks off the lineup is no longer a free assignment problem: some
slots are already spent and the rest must be solved *around* them. ``pinned``
expresses that. The optimality argument above survives unchanged, because pinning
only *removes* players and slots from the free problem — the FLEX still accepts a
superset of what the remaining strict slots accept.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import STARTER_SLOTS

FLEX_ELIGIBLE = ("RB", "WR", "TE")
STRICT_SLOTS = {k: v for k, v in STARTER_SLOTS.items() if k != "FLEX"}


class PinError(ValueError):
    """A pin that cannot be honoured. Loud, never silently dropped (§10)."""


def slot_accepts(slot: str, position: str | None) -> bool:
    """Is ``position`` legal in ``slot``? The only eligibility rule there is."""
    if slot == "FLEX":
        return position in FLEX_ELIGIBLE
    return position == slot


def _validate_pins(players: pl.DataFrame, pinned: dict[str, str]) -> None:
    """Every pin must name a real player, a real slot, and fit inside it.

    A pin that cannot be honoured means the caller's model of the lineup and
    ESPN's disagree — which during a live week means we are about to advise on a
    slot that is already spent. That is a blocking error, not a warning.
    """
    known = set(players["canonical_id"].to_list())
    pos_of = dict(zip(players["canonical_id"].to_list(), players["position"].to_list()))

    counts: dict[str, int] = {}
    for cid, slot in pinned.items():
        if cid not in known:
            raise PinError(
                f"pinned player {cid!r} is not on this roster.\n"
                f"  Pins come from what ESPN has already locked, so this means the\n"
                f"  roster we are solving is not the roster that locked."
            )
        if slot not in STARTER_SLOTS:
            raise PinError(
                f"pinned slot {slot!r} is not a starting slot. "
                f"Known slots: {sorted(STARTER_SLOTS)}"
            )
        if not slot_accepts(slot, pos_of.get(cid)):
            raise PinError(
                f"{cid!r} is a {pos_of.get(cid)!r} and cannot start at {slot!r}."
            )
        counts[slot] = counts.get(slot, 0) + 1

    for slot, n in counts.items():
        if n > STARTER_SLOTS[slot]:
            raise PinError(
                f"{n} players pinned to {slot!r}, which has only "
                f"{STARTER_SLOTS[slot]} starting spot(s)."
            )


def optimal_lineup(
    players: pl.DataFrame,
    value: str = "weekly_points",
    *,
    pinned: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Pick the starting lineup from a roster. Returns the starters with slots.

    ``pinned`` maps ``canonical_id -> slot`` for players whose games have already
    kicked off. Those players occupy their slots regardless of value, and the
    remainder of the lineup is solved around them.
    """
    pinned = dict(pinned or {})
    if pinned:
        _validate_pins(players, pinned)

    pool = players.sort(value, descending=True, nulls_last=True)
    chosen, used = [], set()

    # Pinned players are placed first and consume slot capacity.
    if pinned:
        by_id = {r["canonical_id"]: r for r in pool.to_dicts()}
        for cid, slot in pinned.items():
            used.add(cid)
            chosen.append({**by_id[cid], "slot": slot})

    spent: dict[str, int] = {}
    for slot in pinned.values():
        spent[slot] = spent.get(slot, 0) + 1

    def fill(slot: str, eligible: pl.Expr, n: int) -> None:
        if n <= 0:
            return
        take = (
            pool.filter(eligible)
            .filter(~pl.col("canonical_id").is_in(list(used)) if used else pl.lit(True))
            .head(n)
        )
        for row in take.to_dicts():
            used.add(row["canonical_id"])
            chosen.append({**row, "slot": slot})

    for pos, n in STRICT_SLOTS.items():
        fill(pos, pl.col("position") == pos, n - spent.get(pos, 0))

    for _ in range(STARTER_SLOTS.get("FLEX", 0) - spent.get("FLEX", 0)):
        fill("FLEX", pl.col("position").is_in(list(FLEX_ELIGIBLE)), 1)

    if not chosen:
        return pl.DataFrame(schema={**players.schema, "slot": pl.Utf8})
    return pl.DataFrame(chosen)


def lineup_points(
    players: pl.DataFrame,
    value: str = "weekly_points",
    *,
    pinned: dict[str, str] | None = None,
) -> float:
    lu = optimal_lineup(players, value, pinned=pinned)
    return float(lu[value].sum()) if lu.height else 0.0
