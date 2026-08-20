"""Human review tools — the Milestone 4 test is a judgement call, so make it fast.

§11: "top 50 sanity-checked; every divergence from public rankings has an
explanation you agree with, and week 5/14 bye players visibly rise."

That is a review, not an assertion, and a reviewer needs the REASON attached to
each divergence rather than a ranked list to squint at.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import FREE_BYE_WEEKS, POSITION_MAXIMA, STARTER_SLOTS


BLEND_WEIGHT = 0.12
"""M3's blend weight — needed to size the model's contribution in points."""


def divergences(board: pl.DataFrame, top: int = 50, min_gap: int = 5) -> pl.DataFrame:
    """Where this board disagrees with consensus, and WHY.

    ``rank_vs_ecr`` is positive when we rank a player HIGHER than consensus.

    Attribution compares the actual point contributions and picks the largest one
    whose SIGN matches the direction of the disagreement. Reporting whichever
    flag happens to be set instead would mislabel players: Tucker Kraft carries a
    TD-regression flag and still ranks 39 spots ABOVE consensus, so the flag is
    not what moved him — positional scarcity is.
    """
    sub = board.head(top).filter(pl.col("rank_vs_ecr").abs() >= min_gap)

    contrib = sub.with_columns(
        pl.col("sack_adjustment").fill_null(0.0).alias("c_sack"),
        pl.col("bye_adjustment").fill_null(0.0).alias("c_bye"),
        (BLEND_WEIGHT * (pl.col("model_points") - pl.col("consensus_points"))
         ).fill_null(0.0).alias("c_model"),
    )
    # whatever the named terms do not explain is positional: ECR ranks across
    # positions, VOR prices each against its own replacement level
    contrib = contrib.with_columns(
        (pl.col("vor") - pl.col("consensus_points")
         - pl.col("c_sack") - pl.col("c_bye") - pl.col("c_model")).alias("c_position")
    )

    up = pl.col("rank_vs_ecr") > 0
    def aligned(col: str) -> pl.Expr:
        """Contribution magnitude, but only when it pushes the right way."""
        c = pl.col(col)
        return pl.when((up & (c > 0)) | (~up & (c < 0))).then(c.abs()).otherwise(0.0)

    reason = (
        pl.when((aligned("c_sack") >= aligned("c_bye"))
                & (aligned("c_sack") >= aligned("c_model"))
                & (aligned("c_sack") >= 2.0))
        .then(pl.lit("sack rate vs expected (§3.3) — a persistent trait consensus does not price"))
        .when((aligned("c_model") >= aligned("c_bye"))
              & (aligned("c_model") >= 2.0))
        .then(pl.lit("opportunity model disagrees with consensus (§7.2 step 3)"))
        .when(aligned("c_bye") >= 1.0)
        .then(pl.lit("free bye in week 5 or 14 (§2.1)"))
        .otherwise(pl.lit("positional replacement level (§7.3) — scarcity, not the player"))
        .alias("reason")
    )
    return contrib.with_columns(
        reason,
        (pl.col("model_points") - pl.col("consensus_points")).round(1)
        .alias("model_minus_consensus"),
        pl.when(up).then(pl.lit("HIGHER")).otherwise(pl.lit("lower")).alias("vs_consensus"),
    ).select(
        "overall_rank", "name", "position", "team", "ecr", "rank_vs_ecr",
        "vs_consensus", "reason", "vor", "bye_adjustment", "sack_adjustment",
        "model_minus_consensus", "flag_sack_risk", "flag_td_regression",
    ).sort("rank_vs_ecr", descending=True)


def bye_lift(board: pl.DataFrame, top: int = 100) -> pl.DataFrame:
    """§11 asks that week 5/14 bye players 'visibly rise'. Measure it."""
    sub = board.head(top)
    return (
        sub.group_by("free_bye_week_5_or_14")
        .agg(
            pl.len().alias("n"),
            pl.col("bye_adjustment").mean().round(2).alias("mean_bye_adj"),
            pl.col("rank_vs_ecr").mean().round(2).alias("mean_rank_gain_vs_ecr"),
        )
        .sort("free_bye_week_5_or_14", descending=True)
    )


def qb_count_decision(board: pl.DataFrame) -> dict:
    """§2.1's explicit question: do I need a third QB?

        "if both your starters bye in weeks 5 or 14, you may only need two,
         freeing a bench spot for an upside swing."

    Compares the marginal value of QB3 (insurance for the weeks your top two are
    out) against the best available non-QB stash.
    """
    qbs = board.filter(pl.col("position") == "QB").sort("vor", descending=True)
    if qbs.height < 3:
        return {"decision": "insufficient QB data"}

    top2 = qbs.head(2)
    byes = [b for b in top2["bye_week"].to_list() if b is not None]
    covered = [b for b in byes if b in FREE_BYE_WEEKS]
    uncovered = [b for b in byes if b not in FREE_BYE_WEEKS]

    qb3 = qbs.row(2, named=True)
    repl_qb = float(qbs["replacement_points"][0])
    weeks_needed = len(uncovered)
    # QB3 only starts in the weeks a starter is actually out
    qb3_value = max(0.0, (float(qb3["blended_points"]) - repl_qb) / 17.0 * weeks_needed)

    best_stash = (
        board.filter(~pl.col("position").is_in(["QB", "K", "DST"]))
        .sort("ceiling_points", descending=True, nulls_last=True)
        .head(1)
    )
    stash_upside = 0.0
    if best_stash.height:
        stash_upside = float(
            (best_stash["ceiling_points"][0] or 0) - (best_stash["blended_points"][0] or 0)
        )

    return {
        "top2_qbs": top2["name"].to_list(),
        "top2_bye_weeks": byes,
        "byes_covered_by_my_byes": covered,
        "byes_needing_cover": uncovered,
        "weeks_qb3_would_start": weeks_needed,
        "qb3_marginal_points": round(qb3_value, 2),
        "best_non_qb_stash_upside": round(stash_upside, 2),
        "decision": (
            "TWO QBs — both starter byes fall in my bye weeks, so QB3 never "
            "starts; use the bench spot on an upside swing (§2.1, §3.2)"
            if weeks_needed == 0 else
            f"THREE QBs — {weeks_needed} starter bye week(s) need covering"
            if qb3_value >= stash_upside else
            f"TWO QBs — QB3 is worth {qb3_value:.1f} pts but the best stash "
            f"offers {stash_upside:.1f} pts of upside"
        ),
    }


def sanity_alarms(board: pl.DataFrame) -> list[str]:
    """§10's alarms, each of which means something is broken."""
    alarms = []
    top = board.head(9 * 12)          # roughly rounds 1-12 in a 9-team league
    k_early = top.filter(pl.col("position") == "K")
    if k_early.height:
        alarms.append(
            f"KICKER inside the first 12 rounds: {k_early['name'].to_list()[:3]} "
            f"— §3.5 says never before the last two rounds"
        )
    for pos, cap in POSITION_MAXIMA.items():
        n = board.filter(pl.col("position") == pos).height
        if 0 < n < STARTER_SLOTS.get(pos, 0):
            alarms.append(f"only {n} {pos} on the board, fewer than the {STARTER_SLOTS[pos]} starting slots")
    if board.filter(pl.col("bye_adjustment") < 0).height:
        alarms.append("negative bye adjustment — a free bye must never be a penalty")
    if board.filter(pl.col("vor").is_null()).height:
        alarms.append("null VOR present")
    return alarms
