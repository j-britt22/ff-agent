"""Tiers (§7.4) — cluster by projected-point gaps within position.

"Tiers beat ranks — #6 vs #7 RB is noise, a tier cliff is real." So a tier break
is a GAP that stands out against the gaps around it, not a fixed count of players.

Tier boundaries are unstable to small projection changes by nature, so
``tier_stability`` re-runs the clustering under perturbation and reports how
often each break survives. A cliff that only appears at one exact projection is
not a cliff.
"""

from __future__ import annotations

import numpy as np
import polars as pl

GAP_MULTIPLIER = 2.0
"""A break needs a gap this many times the position's median gap."""

MIN_TIER_SIZE = 1
MAX_TIERS = 12


def assign_tiers(
    df: pl.DataFrame,
    value: str = "blended_points",
    position_col: str = "position",
    gap_multiplier: float = GAP_MULTIPLIER,
) -> pl.DataFrame:
    """Add ``tier`` and ``players_remaining_in_tier`` within each position."""
    out = []
    for pos in df[position_col].unique().to_list():
        sub = (df.filter(pl.col(position_col) == pos)
               .sort(value, descending=True, nulls_last=True))
        vals = sub[value].fill_null(0.0).to_numpy().astype(float)
        if len(vals) < 2:
            out.append(sub.with_columns(pl.lit(1).alias("tier")))
            continue
        gaps = -np.diff(vals)                     # descending, so positive
        med = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 0.0
        threshold = max(med * gap_multiplier, 1e-9)

        tier, tiers = 1, [1]
        for g in gaps:
            if g >= threshold and tier < MAX_TIERS:
                tier += 1
            tiers.append(tier)
        out.append(sub.with_columns(pl.Series("tier", tiers, dtype=pl.Int32)))

    res = pl.concat(out)
    return res.with_columns(
        pl.len().over([position_col, "tier"]).alias("tier_size"),
        (pl.len().over([position_col, "tier"])
         - pl.col(value).rank("ordinal", descending=True)
         .over([position_col, "tier"]) + 1)
        .cast(pl.Int32).alias("players_remaining_in_tier"),
    )


def tier_stability(
    df: pl.DataFrame,
    value: str = "blended_points",
    noise_pct: float = 0.05,
    trials: int = 40,
    seed: int = 17,
) -> pl.DataFrame:
    """How often does each player's tier BREAK survive perturbation?

    Risk control, not decoration: presenting a cliff that only exists at one
    exact projection would be false precision.

    Requires one row per ``canonical_id`` (§0.2), and refuses anything else.
    Duplicated input breaks this function twice over. The count is incremented
    once per ROW per trial, so a player present twice reports
    ``2 x trials / trials = 2.0`` — impossible for a field documented as a
    frequency, and the 2026 board shipped 28 such rows. And the returned frame
    inherits the duplicate, so joining it back onto the caller's frame SQUARES
    the duplication rather than merely passing it through.
    """
    dupes = df.group_by("canonical_id").len().filter(pl.col("len") > 1)
    if dupes.height:
        cols = [c for c in ("canonical_id", "name", "position", "team", value)
                if c in df.columns]
        detail = (df.filter(pl.col("canonical_id").is_in(dupes["canonical_id"].implode()))
                  .select(cols).sort("canonical_id"))
        raise ValueError(
            f"tier_stability needs one row per canonical_id; {dupes.height} "
            f"id(s) appear more than once ({int(dupes['len'].sum())} rows). "
            f"Deduplicating here would hide a join that fanned out upstream — "
            f"find that join instead.\n{detail}"
        )
    rng = np.random.default_rng(seed)
    base = assign_tiers(df, value=value).select(
        "canonical_id", "position", pl.col("tier").alias("base_tier")
    )
    counts: dict[str, int] = {}
    for _ in range(trials):
        noisy = df.with_columns(
            (pl.col(value) * (1.0 + pl.Series(
                rng.normal(0.0, noise_pct, df.height)))).alias(value)
        )
        t = assign_tiers(noisy, value=value)
        for r in t.select("canonical_id", "tier").to_dicts():
            key = f"{r['canonical_id']}|{r['tier']}"
            counts[key] = counts.get(key, 0) + 1
    rows = []
    for r in base.to_dicts():
        k = f"{r['canonical_id']}|{r['base_tier']}"
        rows.append({
            "canonical_id": r["canonical_id"],
            "tier_stability": round(counts.get(k, 0) / trials, 3),
        })
    return pl.DataFrame(rows)
