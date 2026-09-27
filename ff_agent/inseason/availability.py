"""Will he actually play? (M10b finding F10.)

The option value of waiting is mostly the probability that a player's status
*changes*, so the Thursday decision turns on ``P(does not play | designation)``.
nflverse ``injuries`` carries ``report_status`` and ``practice_status`` weekly and
keys on **``gsis_id``** — it joins straight to canonical with no crosswalk hop.

Measured on 2025 regular season, skill positions:

===========================  ====  ===============
Friday designation              n  P(did not play)
===========================  ====  ===============
*(on report, no designation)*  942  0.149
Questionable                   352  0.418
Doubtful                        26  1.000
Out                            430  1.000
===========================  ====  ===============

Two sub-findings are worth more than the headline.

**A Questionable QB is a different animal** — 0.735 against a Questionable RB's
0.360, roughly double. Plausible mechanism: a quarterback who cannot go is
replaced outright and takes no snap, while a receiver who dresses nearly always
records something. In a 2-QB league that is the position where it matters most,
and it is also where §9.3 says waiver priority gets spent.

**Full practice participation does NOT separate from limited** — 0.421 against
0.404, with did-not-practice at 0.512. The folk rule ("full practice Friday means
he plays") does not hold on this data. Small cell (n=76), so suggestive rather
than settled — but it is the opposite of what a hand-written heuristic would have
encoded, which is the whole reason to measure instead of assume.

**The levels are biased upward and the bias is estimated, not waved at.** The
proxy is absence from the weekly stats table, which conflates a true inactive
with a player who dressed and recorded nothing. Players on the report with *no*
designation come back at 0.149, which is a direct estimate of that bias — so the
corrected read is roughly 0.27 for Questionable and 0.59 for a Questionable QB.
``corrected=True`` (the default) subtracts it. One season, and Doubtful has 26
rows; M10b-4 redoes this against snap counts across 2016-2025. It is used now
because the positional asymmetry is far larger than any plausible correction.
"""

from __future__ import annotations

import polars as pl

MEASURED_SEASON = 2025
BASELINE_ABSENCE = 0.149
"""P(no stats row | on the injury report, no game-status designation).

This is the measurement bias, not a real absence rate. Subtracting it is what
turns the raw numbers into something usable."""

CERTAIN_OUT = ("Out", "Doubtful")
"""Both measured at 1.000 in 2025. Doubtful on n=26 — small, but a Doubtful
player who plays is rare enough that treating him as out costs little and
treating him as available costs a start."""

QUESTIONABLE_RAW = 0.418
QUESTIONABLE_BY_POSITION = {
    "QB": 0.735, "TE": 0.418, "WR": 0.382, "RB": 0.360,
}
"""The asymmetry that makes the Thursday decision positional."""

PRACTICE_RAW = {
    "Did Not Participate In Practice": 0.512,
    "Limited Participation in Practice": 0.404,
    "Full Participation in Practice": 0.421,
}
"""Kept for the record and deliberately NOT used to adjust the estimate: full
and limited are inseparable (0.421 vs 0.404), so a practice-based adjustment
would be noise dressed as precision. Only did-not-practice separates, and it is
folded in below as a small bump rather than a third scale."""

DNP_BUMP = 0.10
"""Roughly 0.512 - 0.404. The one practice signal that survives the measurement."""

HEALTHY = 0.02
"""A player with no injury designation at all. Not zero: people get hurt in
warmups, and a floor of exactly zero would let the optimizer treat a start as
risk-free."""


def p_out(
    report_status: str | None,
    position: str | None = None,
    practice_status: str | None = None,
    corrected: bool = True,
) -> float:
    """Probability this player does not play, from his Friday designation."""
    status = (report_status or "").strip()
    if status in CERTAIN_OUT:
        return 1.0
    if status != "Questionable":
        return HEALTHY

    raw = QUESTIONABLE_BY_POSITION.get((position or "").upper(), QUESTIONABLE_RAW)
    if (practice_status or "").startswith("Did Not"):
        raw += DNP_BUMP
    if corrected:
        raw -= BASELINE_ABSENCE
    return float(min(1.0, max(HEALTHY, raw)))


def attach(
    roster: pl.DataFrame,
    injuries: pl.DataFrame | None = None,
    corrected: bool = True,
) -> pl.DataFrame:
    """Add ``p_out`` to a roster from an nflverse injuries frame.

    Joins on ``gsis_id == canonical_id``. If ``injuries`` is None every player is
    treated as healthy and the caller is expected to say so — a silent default of
    "everyone plays" during a week when the report was unavailable is exactly the
    stale-data failure §10 is about.
    """
    if injuries is None or injuries.is_empty():
        return roster.with_columns(
            pl.lit(HEALTHY).alias("p_out"),
            pl.lit(None, dtype=pl.Utf8).alias("report_status"),
        )

    inj = injuries.select(
        pl.col("gsis_id").alias("canonical_id"),
        pl.col("report_status"),
        pl.col("practice_status"),
    ).unique(subset=["canonical_id"], keep="first")

    out = roster.join(inj, on="canonical_id", how="left")
    if out.height != roster.height:
        raise ValueError(
            f"the injury join changed the row count ({roster.height} -> "
            f"{out.height}). §0.2: a fan-out breaks one-row-per-player as "
            f"thoroughly as an unresolved id."
        )
    return out.with_columns(
        pl.struct("report_status", "position", "practice_status")
        .map_elements(
            lambda s: p_out(s["report_status"], s["position"],
                            s["practice_status"], corrected),
            return_dtype=pl.Float64,
        )
        .alias("p_out")
    )


def flagged(roster: pl.DataFrame, threshold: float = 0.15) -> pl.DataFrame:
    """Players worth naming in the digest. §9.1's "flag every Q/D"."""
    if "p_out" not in roster.columns:
        return roster.head(0)
    return roster.filter(pl.col("p_out") >= threshold).sort("p_out", descending=True)
