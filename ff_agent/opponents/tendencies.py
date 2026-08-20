"""Per-manager tendencies (§7.5), shrunk hard toward the league prior.

With 17 to 49 picks per manager, shrinkage is not conservatism — it is the only
way to avoid publishing confident nonsense. A manager who happened to take two
tight ends early in his single draft is not "a TE guy".

Every tendency carries the evidence behind it, because §2.3 weights the four
double-up managers most heavily and those are not the managers with the most
data. The `opponent-scout` agent is explicit: flag anything resting on fewer than
two drafts.
"""

from __future__ import annotations

import polars as pl

from ff_agent.opponents import history as H
from ff_agent.projections import consensus as C

SHRINKAGE_PICKS = 30.0
"""Pseudo-count pulling each manager toward the league prior.

A manager with 17 picks lands ~36% of the way to his own rate; one with 49 picks
lands ~62%. Deliberately heavy — see the module docstring."""

PHASES = (("early", 0.0, 0.25), ("middle", 0.25, 0.6), ("late", 0.6, 1.01))
"""Draft phases as FRACTIONS, so 8-, 9- and 10-team drafts are comparable."""

SKILL = ("QB", "RB", "WR", "TE")

FORMAT_MATCH_WEIGHT = 3.0
"""Weight on picks from a season whose FORMAT matches the target season.

The strict backtest showed why this is necessary rather than tidy. Fitting on
2023-24 (both 1-QB) and predicting 2025 (2-QB) made two managers WORSE than pure
ADP, the worst being Jordan Britt at -0.27 — he took the first quarterback at
pick 7 in the 2-QB season, behaviour his 1-QB drafts give no hint of. A manager's
own history can actively mislead across a format change, so the format-matched
season carries triple weight in the shipped model."""


def season_weight(season: int, target_two_qb: bool = True) -> float:
    """Evidence weight for a season, given the format we are predicting."""
    matches = (season in H.TWO_QB_SEASONS) == target_two_qb
    return FORMAT_MATCH_WEIGHT if matches else 1.0


def _ecr_for(season: int) -> pl.DataFrame:
    """Consensus ranks matched to that season's FORMAT.

    Superflex for 2-QB seasons, standard for the 1-QB ones. Comparing a 1-QB
    draft against superflex ranks would manufacture enormous fake "reaches" at
    quarterback — the format difference showing up as a personality trait.
    """
    kind = C.SUPERFLEX if season in H.TWO_QB_SEASONS else "ro"
    e = C.ecr(season, ecr_type=kind).filter(pl.col("canonical_id").is_not_null())
    return e.select(
        "canonical_id",
        pl.col("ecr").alias("consensus_rank"),
        pl.lit(kind).alias("ecr_type"),
    )


def picks_with_consensus(hist: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every pick joined to the consensus rank that applied at the time."""
    h = H.draft_history() if hist is None else hist
    frames = []
    for season in sorted(h["season"].unique().to_list()):
        s = h.filter(pl.col("season") == season)
        e = _ecr_for(season)
        canon = __import__("ff_agent.data.crosswalk", fromlist=["x"]).canonical_players()
        s = s.join(
            canon.select(pl.col("espn_id"), pl.col("canonical_id")),
            on="espn_id", how="left",
        ).join(e, on="canonical_id", how="left")
        # consensus rank expressed as a fraction of the same draft
        s = s.with_columns(
            (pl.col("consensus_rank") / pl.col("total_picks")).alias("consensus_fraction")
        ).with_columns(
            # positive = took him EARLIER than consensus (a reach)
            (pl.col("consensus_fraction") - pl.col("pick_fraction")).alias("reach")
        )
        frames.append(s)
    return pl.concat(frames, how="diagonal")


def _phase_expr() -> pl.Expr:
    e = None
    for name, lo, hi in PHASES:
        cond = (pl.col("pick_fraction") >= lo) & (pl.col("pick_fraction") < hi)
        e = pl.when(cond).then(pl.lit(name)) if e is None else e.when(cond).then(pl.lit(name))
    return e.otherwise(pl.lit("late")).alias("phase")


def positional_tendency(
    picks: pl.DataFrame | None = None, two_qb_only: bool = False
) -> pl.DataFrame:
    """P(position | manager, phase), shrunk toward the league rate for that phase."""
    p = picks_with_consensus() if picks is None else picks
    if two_qb_only:
        p = p.filter(pl.col("two_qb_format"))
    p = (p.filter(pl.col("position").is_in(list(SKILL)))
         .with_columns(_phase_expr())
         .with_columns(
             pl.col("season")
             .map_elements(lambda s: season_weight(s), return_dtype=pl.Float64)
             .alias("w")
         ))

    league = (
        p.group_by("phase", "position").agg(pl.col("w").sum().alias("lg_n"))
        .with_columns((pl.col("lg_n") / pl.col("lg_n").sum().over("phase")).alias("league_rate"))
        .select("phase", "position", "league_rate")
    )
    mgr = (
        p.group_by("manager", "phase", "position").agg(pl.col("w").sum().alias("n"))
        .with_columns(pl.col("n").sum().over(["manager", "phase"]).alias("phase_picks"))
        .with_columns((pl.col("n") / pl.col("phase_picks")).alias("raw_rate"))
    )
    grid = (
        mgr.select("manager").unique()
        .join(league, how="cross")
    )
    joined = (
        grid.join(mgr, on=["manager", "phase", "position"], how="left")
        .with_columns(
            pl.col("n").fill_null(0),
            pl.col("phase_picks").fill_null(0),
            pl.col("raw_rate").fill_null(0.0),
        )
    )
    k = SHRINKAGE_PICKS
    return joined.with_columns(
        ((pl.col("n") + k * pl.col("league_rate"))
         / (pl.col("phase_picks") + k)).alias("rate"),
        (pl.col("phase_picks") / (pl.col("phase_picks") + k)).alias("evidence_weight"),
    ).with_columns(
        (pl.col("rate") - pl.col("league_rate")).alias("tilt_vs_league")
    ).sort(["manager", "phase", "position"])


def reach_profile(picks: pl.DataFrame | None = None) -> pl.DataFrame:
    """How far ahead of consensus each manager drafts, and how consistently.

    Positive ``mean_reach`` means taking players earlier than consensus. Measured
    as a fraction of the draft so it survives the changing team count.
    """
    p = picks_with_consensus() if picks is None else picks
    p = p.filter(pl.col("reach").is_not_null())
    lg_mean = float(p["reach"].mean())
    k = SHRINKAGE_PICKS
    return (
        p.group_by("manager")
        .agg(
            pl.len().alias("n_picks_ranked"),
            pl.col("reach").mean().alias("raw_mean_reach"),
            pl.col("reach").std().alias("reach_sd"),
            pl.col("season").n_unique().alias("n_drafts"),
        )
        .with_columns(
            ((pl.col("n_picks_ranked") * pl.col("raw_mean_reach") + k * lg_mean)
             / (pl.col("n_picks_ranked") + k)).round(4).alias("mean_reach"),
            pl.lit(round(lg_mean, 4)).alias("league_mean_reach"),
        )
        .sort("mean_reach", descending=True)
    )


def qb_timing(picks: pl.DataFrame | None = None) -> pl.DataFrame:
    """When each manager takes quarterbacks — **2-QB seasons only**.

    §7.5 calls this out as the thing national ADP gets most wrong here. It is
    also the thing this league has the least evidence about: one 2-QB draft.
    """
    p = picks_with_consensus() if picks is None else picks
    p = p.filter(pl.col("two_qb_format") & (pl.col("position") == "QB"))
    return (
        p.group_by("manager")
        .agg(
            pl.len().alias("qbs_taken"),
            pl.col("pick_fraction").min().round(3).alias("first_qb_fraction"),
            pl.col("pick_fraction").median().round(3).alias("median_qb_fraction"),
            pl.col("season").n_unique().alias("n_drafts"),
        )
        .with_columns(pl.lit("one 2-QB draft — treat as weak").alias("confidence"))
        .sort("first_qb_fraction")
    )


def team_bias(picks: pl.DataFrame | None = None, min_picks: int = 3) -> pl.DataFrame:
    """NFL teams a manager returns to more than the league does."""
    p = picks_with_consensus() if picks is None else picks
    from ff_agent.data.crosswalk import canonical_players

    canon = canonical_players().select(
        pl.col("canonical_id"), pl.col("team").alias("nfl_team")
    )
    p = p.join(canon, on="canonical_id", how="left").filter(pl.col("nfl_team").is_not_null())
    league = (
        p.group_by("nfl_team").len().rename({"len": "lg_n"})
        .with_columns((pl.col("lg_n") / pl.col("lg_n").sum()).alias("league_rate"))
    )
    mgr = (
        p.group_by("manager", "nfl_team").len().rename({"len": "n"})
        .with_columns(pl.col("n").sum().over("manager").alias("mgr_picks"))
    )
    return (
        mgr.join(league.select("nfl_team", "league_rate"), on="nfl_team", how="left")
        .with_columns((pl.col("n") / pl.col("mgr_picks")).alias("rate"))
        .with_columns((pl.col("rate") - pl.col("league_rate")).alias("bias"))
        .filter(pl.col("n") >= min_picks)
        .sort("bias", descending=True)
    )
