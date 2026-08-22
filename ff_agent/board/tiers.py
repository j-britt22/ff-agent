"""Tiers (§7.4) — cluster by projected-point gaps within position.

"Tiers beat ranks — #6 vs #7 RB is noise, a tier cliff is real." So a tier break
is a GAP that stands out against the gaps around it, not a fixed count of players.

Tier boundaries are unstable to small projection changes by nature, so
``tier_stability`` re-runs the clustering under perturbation and reports how
often each break survives. A cliff that only appears at one exact projection is
not a cliff.

That guarantee only holds if the stability number is itself reproducible, which
until 2026-08-21 it was not: see ``_noise_matrix``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

GAP_MULTIPLIER = 2.0
"""A break needs a gap this many times the position's median gap."""

MIN_TIER_SIZE = 1
MAX_TIERS = 12

TRIALS = 400
"""Perturbation trials, chosen against the precision we report, not by feel.

The estimate is ``k / trials``, a binomial proportion whose standard error peaks
at ``0.5 / sqrt(trials)`` — 0.025 here. The previous 40 trials gave **0.079**,
so the third decimal in ``artifacts/board.json`` described noise, not a cliff.
Costs ~0.5s on a 530-row board, against a ~7s board build."""

STABILITY_DECIMALS = 2
"""Reported precision, tied to TRIALS' standard error (0.025) rather than taste.

Reporting digits the estimator cannot resolve is the exact false precision
``tier_stability`` exists to prevent, so the two constants move together."""


def assign_tiers(
    df: pl.DataFrame,
    value: str = "blended_points",
    position_col: str = "position",
    gap_multiplier: float = GAP_MULTIPLIER,
) -> pl.DataFrame:
    """Add ``tier`` and ``players_remaining_in_tier`` within each position."""
    out = []
    # maintain_order on BOTH: polars' default unique() is a multithreaded hash
    # and its default sort is unstable, so without these the position blocks and
    # the tied rows inside them come back in a different order on every call —
    # measured at 7 distinct row orders in 8 calls on the 2026 board.
    for pos in df[position_col].unique(maintain_order=True).to_list():
        sub = (df.filter(pl.col(position_col) == pos)
               .sort(value, descending=True, nulls_last=True, maintain_order=True))
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


def _noise_matrix(
    ids: list[str], trials: int, noise_pct: float, seed: int
) -> np.ndarray:
    """One reproducible noise vector per PLAYER, keyed on the canonical id.

    Returns ``(len(ids), trials)``, aligned to ``ids`` positionally but derived
    from the id, so shuffling the frame permutes the rows without changing any
    player's draw.

    This is the fix for a live reproducibility bug. The draws used to be taken as
    one ``rng.normal(..., df.height)`` vector applied POSITIONALLY, which quietly
    made the result a function of row order — and row order is not stable here:
    ``assign_tiers`` concatenates position blocks in ``unique()`` order and sorts
    on a float column full of ties. Two consecutive board builds on byte-identical
    code disagreed on 85 of 515 rows, by up to 0.225, with Gibbs and Chase among
    them. A number whose job is to expose false precision cannot itself move 0.2
    between identical runs.

    Keying on the id also buys insensitivity to the POOL: adding a player, or
    reprojecting one, leaves every other player's draws untouched, so a genuine
    projection change shows up only where it actually landed.
    """
    per_id: dict[str, np.ndarray] = {}
    for pid in dict.fromkeys(ids):          # dedupe, insertion-ordered
        # blake2b, not hash(): PYTHONHASHSEED randomises str hashing per process,
        # which would reintroduce run-to-run drift through the back door.
        key = int.from_bytes(
            hashlib.blake2b(str(pid).encode(), digest_size=8).digest(), "big")
        per_id[pid] = np.random.default_rng([seed, key]).normal(
            0.0, noise_pct, trials)
    return np.array([per_id[p] for p in ids])


def tier_stability(
    df: pl.DataFrame,
    value: str = "blended_points",
    noise_pct: float = 0.05,
    trials: int = TRIALS,
    seed: int = 17,
) -> pl.DataFrame:
    """How often does each player's tier BREAK survive perturbation?

    Risk control, not decoration: presenting a cliff that only exists at one
    exact projection would be false precision. Reproducible run-to-run and
    invariant to the frame's row order — both are pinned in
    ``tests/test_tier_stability.py``.
    """
    if df["canonical_id"].null_count():
        raise ValueError(
            "tier_stability: canonical_id has nulls, and the perturbation is "
            "keyed on it — a null key would silently share one draw."
        )
    
    # §0.2: refuse duplicated input. The count is incremented once per ROW per
    # trial, so a player present twice reports ``2 x trials / trials = 2.0`` —
    # impossible for a field documented as a frequency. And the returned frame
    # inherits the duplicate, so joining it back fanned 15 → 60 rows in 2026.
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
    
    ids = df["canonical_id"].to_list()
    noise = _noise_matrix(ids, trials, noise_pct, seed)

    base = assign_tiers(df, value=value).select(
        "canonical_id", "position", pl.col("tier").alias("base_tier")
    )
    counts: dict[tuple[str, int], int] = {}
    for t in range(trials):
        noisy = df.with_columns(
            (pl.col(value) * (1.0 + pl.Series(noise[:, t]))).alias(value)
        )
        for key in assign_tiers(noisy, value=value).select(
            "canonical_id", "tier"
        ).iter_rows():
            counts[key] = counts.get(key, 0) + 1

    rows = [
        {
            "canonical_id": r["canonical_id"],
            "tier_stability": round(
                counts.get((r["canonical_id"], r["base_tier"]), 0) / trials,
                STABILITY_DECIMALS,
            ),
        }
        for r in base.to_dicts()
    ]
    return pl.DataFrame(rows)
