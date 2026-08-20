"""Replacement levels (§7.3), from MEASURED flex usage.

§7.3 says to allocate the nine flex slots "by historical flex usage (~4 RB / 4 WR
/ 1 TE in half PPR), not evenly". That parenthetical is an estimate, and this
league's own history disagrees with it: across 364 flex starts in 2023-2025 the
flex was **56% RB, 44% WR and never once a tight end**.

Zero. Not one TE in three seasons.

It is a coherent result rather than a fluke — §3.4 shows half-PPR plus 0.05 per
carry pushes flex value toward backs — but it moves the replacement ranks §7.3
prescribes, and TE most of all.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import N_TEAMS, STARTER_SLOTS
from ff_agent.data.cache import cached

FLEX_SLOT_NAMES = frozenset({"RB/WR/TE", "FLEX", "OP", "RB/WR", "WR/TE"})
FLEX_ELIGIBLE = ("RB", "WR", "TE")

SPEC_FLEX_SHARE = {"RB": 4 / 9, "WR": 4 / 9, "TE": 1 / 9}
"""§7.3's estimate, kept for comparison rather than use."""


def measure_flex_usage(seasons: tuple[int, ...] = (2023, 2024, 2025), **kw) -> pl.DataFrame:
    """Who actually gets STARTED in the flex, in this league."""

    def fetch() -> pl.DataFrame:
        from ff_agent.data.espn import get_league

        rows: list[dict] = []
        for yr in seasons:
            lg = get_league(yr)
            for wk in range(1, 15):
                try:
                    boxes = lg.box_scores(wk)
                except Exception:
                    continue
                for m in boxes:
                    for p in list(m.home_lineup) + list(m.away_lineup):
                        slot = getattr(p, "slot_position", None)
                        if slot in FLEX_SLOT_NAMES:
                            rows.append({"season": yr, "week": wk,
                                         "position": getattr(p, "position", None)})
        if not rows:
            raise RuntimeError("No flex starts found in league history")
        return pl.DataFrame(rows)

    return cached("flex_usage", fetch, source="espn", **kw)


def flex_share(seasons: tuple[int, ...] = (2023, 2024, 2025)) -> dict[str, float]:
    """Measured share of flex starts by position. TE is expected to be ~0 here."""
    u = measure_flex_usage(seasons)
    counts = (
        u.filter(pl.col("position").is_in(list(FLEX_ELIGIBLE)))
        .group_by("position").len()
    )
    total = int(counts["len"].sum())
    share = {p: 0.0 for p in FLEX_ELIGIBLE}
    for r in counts.to_dicts():
        share[r["position"]] = r["len"] / total
    return share


def replacement_ranks(
    seasons: tuple[int, ...] = (2023, 2024, 2025),
    n_teams: int = N_TEAMS,
    use_measured: bool = True,
) -> dict[str, float]:
    """§7.3's replacement ranks, with the flex allocated by measured usage.

        replacement_rank[QB] = teams x starting QB slots
        replacement_rank[RB] = teams x RB slots + flex_share[RB] x flex slots
        ...

    A fractional rank is honest: RB23.1 means replacement sits between RB23 and
    RB24, and rounding it away pretends to a precision the estimate lacks.
    """
    share = flex_share(seasons) if use_measured else dict(SPEC_FLEX_SHARE)
    flex_slots = STARTER_SLOTS.get("FLEX", 0) * n_teams
    out: dict[str, float] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        base = STARTER_SLOTS.get(pos, 0) * n_teams
        out[pos] = base + share.get(pos, 0.0) * flex_slots
    out["DST"] = STARTER_SLOTS.get("DST", 0) * n_teams
    out["K"] = STARTER_SLOTS.get("K", 0) * n_teams
    return out


def replacement_table(
    curve: pl.DataFrame, ranks: dict[str, float] | None = None
) -> pl.DataFrame:
    """Replacement-level §1 points per position, read off the M3 rank curve.

    Fractional ranks are interpolated between the two neighbouring integer ranks.
    """
    ranks = replacement_ranks() if ranks is None else ranks
    rows = []
    col = ("expected_points_smooth" if "expected_points_smooth" in curve.columns
           else "expected_points")
    for pos, r in ranks.items():
        sub = curve.filter(pl.col("position") == pos).sort("pos_rank")
        if sub.is_empty():
            continue
        lo, hi = int(r), int(r) + 1
        vlo = sub.filter(pl.col("pos_rank") == lo)[col]
        vhi = sub.filter(pl.col("pos_rank") == hi)[col]
        if vlo.is_empty():
            continue
        a = vlo[0]
        b = vhi[0] if not vhi.is_empty() else a
        frac = r - lo
        rows.append({
            "position": pos,
            "replacement_rank": round(r, 2),
            "replacement_points": round(a + (b - a) * frac, 2),
        })
    return pl.DataFrame(rows).sort("position")
