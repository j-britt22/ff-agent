"""Rest-of-season projections — F1's resolution.

**There is no format-matched rest-of-season consensus.** Measured 2026-08-23 on
the DynastyProcess ECR table: ``rsf``, the superflex list M3 shipped on, is
scraped in the PRESEASON ONLY — one snapshot at 2025-09-05, then nothing until
the following August. The lists that do update weekly in-season are
``do dp drk dsf ro rp wo wp wsf``. Of those exactly one is superflex-and-redraft
shaped — ``wsf`` — and it covers THIS WEEK ONLY. ``ro``/``rp`` are
rest-of-season but 1-QB: on 2025-10-31 ``ro``'s top 24 held **zero** quarterbacks
against ``wsf``'s **13**, and its overall #1 was a linebacker, because the page
ships IDP-polluted.

M5 already recorded the consequence of ignoring format — "format difference
posing as personality". Here the same mistake manufactures fake waiver value at
quarterback, precisely where §9.3 says to spend priority.

**So the anchor is POINTS, not RANKS.** A rank has a format; a points projection
does not. ESPN publishes per-week projected points already expressed in §1
scoring, and M2 proved to the decimal that our ruleset reproduces ESPN's totals
from ESPN's own stat line, on both rulesets.

On top of the anchor go exactly two corrections, and only because each one is
something ESPN's projection is *structurally unable* to contain:

  1. **The sack term** (§3.3, M3b). This league charges -1 per sack taken and
     effectively no other league does, so no public projection prices it.
     Sacks-over-expected persists at **0.434** — four times TD-over-expected's
     0.104 — which is what makes it an adjustment rather than a warning label.
  2. **The kicker distance buckets** (§1, ADD-§E). M3 found ESPN merges 50-59
     and 60+ into one ``50Plus`` bucket while §1 pays 5 and 6 respectively. Its
     projection therefore *cannot* price the exact distinction this league pays
     for.

A general usage model is built here too, but its blend weight ships at **zero**
until M10b-2's walk-forward gate measures it. M3's fitted 0.12 was fitted for
PRESEASON season-long projections scored on rank correlation; carrying it over to
an in-season points task would be exactly the unjustified transfer this project
keeps refusing to make. If the gate says zero, that is the finding, and it goes
beside xFP in the list of things measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ff_agent.board.build import SOE_PERSISTENCE
from ff_agent.config import REGULAR_SEASON_WEEKS, SEASON

MODEL_WEIGHT = 0.0
"""Weight on our usage model against the ESPN points anchor.

Set by M10b-2's gate, never by intuition. Zero means "the gate has not run, so
ship the anchor" — which is a shippable state, not a placeholder: M3 measured
our model ALONE at 0.687 mean Spearman against consensus's 0.761, so the prior
that a fitted weight is small is strong and the prior that it is zero is not
unreasonable."""

GAMES_IN_NFL_SEASON = 17
"""The NFL plays 18 WEEKS and 17 GAMES — the bye is the eighteenth week, not one
of the seventeen. M7's ``weekly_scale`` divided by 16 on the opposite reasoning
and inflated every player by 6.25%; a test had pinned the wrong answer, which is
why it survived. ESPN's season projection spans all 17."""

MIN_GAMES_FOR_USAGE = 4
"""Below this, current-season usage is noise wearing a decimal point.

ADD-§B measured volume stability at 0.584 and efficiency at 0.292 over FULL
seasons. Three games of either is worth less than the preseason projection it
would be replacing, which is why the shrinkage below is by games played."""

USAGE_PRIOR_GAMES = 6.0
"""Pseudo-games pulling current-season usage toward the preseason projection.

The in-season analogue of M5's 30-pick pseudo-count. Without it the week-3
digest is a list of people who had one good game."""


class ROSError(RuntimeError):
    """Rest-of-season inputs are unusable. Blocking, never a warning (§10)."""


@dataclass
class ROSProjection:
    """Per-player rest-of-season value, plus how it was arrived at."""
    frame: pl.DataFrame
    season: int
    from_week: int
    weeks_remaining: tuple[int, ...]
    model_weight: float
    notes: list[str] = field(default_factory=list)

    @property
    def n_players(self) -> int:
        return self.frame.height


REQUIRED_ANCHOR = ("canonical_id", "position", "team", "ros_points", "games_remaining")


def remaining_weeks(from_week: int, season: int = SEASON) -> tuple[int, ...]:
    """NFL regular-season weeks from ``from_week`` onward, inclusive.

    Deliberately the NFL calendar, not my fantasy one: a player's bye and his
    remaining games are properties of the NFL season. Which of those weeks
    matter to ME is §2.1's question and is answered in ``value.py`` by
    ``play_weeks`` — keeping the two apart is what stops my weeks 5 and 14
    leaking into everybody else's denominator.
    """
    if from_week < 1:
        raise ROSError(f"from_week must be >= 1, got {from_week}")
    return tuple(w for w in REGULAR_SEASON_WEEKS if w >= from_week)


def weekly_rate(frame: pl.DataFrame) -> pl.DataFrame:
    """Season-remaining points -> a per-GAME rate the lineup solver can use.

    Divided by games remaining, not weeks remaining. The distinction is the one
    M7 got wrong in the other direction and paid 6.25% for: **the NFL plays 18
    weeks and 17 games**, so a bye is a week without a game, not a shorter
    season. ``value.py`` re-applies the bye week by week, and double-charging is
    prevented by dividing by games here.
    """
    if "games_remaining" not in frame.columns:
        raise ROSError("frame has no games_remaining; cannot form a weekly rate.")
    bad = frame.filter(pl.col("games_remaining") <= 0)
    if bad.height:
        raise ROSError(
            f"{bad.height} player(s) have games_remaining <= 0, e.g. "
            f"{bad.head(3)['canonical_id'].to_list()}. A player with no games "
            f"left has no weekly rate; drop him rather than dividing by zero."
        )
    return frame.with_columns(
        (pl.col("ros_points") / pl.col("games_remaining")).alias("weekly_points")
    )


def sack_correction(frame: pl.DataFrame, games_remaining_col: str = "games_remaining") -> pl.DataFrame:
    """§3.3, priced into the rest-of-season total for quarterbacks.

    ``sacks_over_expected_per_game`` is this season's observed rate. Each sack is
    -1 under §1 and the rate carries forward at 0.434, so the remaining cost is
    ``-0.434 x rate x games_remaining``. A quarterback taking three sacks a game
    more than expected gives back roughly 1.3 points a game against a neutral
    one, every remaining week, and no public projection has charged him for it.
    """
    if "sacks_over_expected_per_game" not in frame.columns:
        return frame.with_columns(pl.lit(0.0).alias("sack_correction"))
    return frame.with_columns(
        pl.when(
            (pl.col("position") == "QB")
            & pl.col("sacks_over_expected_per_game").is_not_null()
        )
        .then(
            -pl.col("sacks_over_expected_per_game")
            * SOE_PERSISTENCE
            * pl.col(games_remaining_col)
        )
        .otherwise(0.0)
        .round(3)
        .alias("sack_correction")
    )


KICKER_LONG_BONUS = 1.0
"""§1 pays 6 for a 60+ field goal and 5 for 50-59; ESPN's projection merges both
into one ``50Plus`` bucket (M3). So every 60-yarder ESPN projects is worth one
point more here than ESPN's own number says. One point per expected 60+ attempt
— small, real, and invisible to every other source."""


def kicker_correction(frame: pl.DataFrame) -> pl.DataFrame:
    """The bucket ESPN cannot express, in points.

    Needs ``expected_fg_60_plus`` — expected makes from 60+ over the remaining
    season. Absent, the correction is zero and says so rather than guessing a
    league-average leg.
    """
    if "expected_fg_60_plus" not in frame.columns:
        return frame.with_columns(pl.lit(0.0).alias("kicker_correction"))
    return frame.with_columns(
        pl.when((pl.col("position") == "K") & pl.col("expected_fg_60_plus").is_not_null())
        .then(pl.col("expected_fg_60_plus") * KICKER_LONG_BONUS)
        .otherwise(0.0)
        .round(3)
        .alias("kicker_correction")
    )


def blend(
    anchor: pl.DataFrame,
    model: pl.DataFrame | None = None,
    weight: float = MODEL_WEIGHT,
) -> pl.DataFrame:
    """Blend our usage model into the points anchor at ``weight``.

    §7.2 step 4's "blend, don't replace" is not stylistic: M3 measured the model
    ALONE at 0.687 mean Spearman against consensus's 0.761, because consensus
    encodes offseason and in-week news no historical model can see. In-season
    that gap should NARROW — observed usage is real information the preseason
    consensus did not have — which is precisely what M10b-2's gate measures.
    """
    if not 0.0 <= weight <= 1.0:
        raise ROSError(f"weight must be in [0, 1], got {weight}")
    if model is None or weight == 0.0:
        return anchor.with_columns(
            pl.col("ros_points").alias("anchor_points"),
            pl.lit(None, dtype=pl.Float64).alias("model_points"),
            pl.lit(weight).alias("model_weight"),
        )

    joined = anchor.join(
        model.select("canonical_id", pl.col("ros_points").alias("model_points")),
        on="canonical_id", how="left",
    )
    if joined.height != anchor.height:
        raise ROSError(
            f"the model join changed the row count ({anchor.height} -> "
            f"{joined.height}). §0.2: a join that fans out breaks the one-row-"
            f"per-player rule just as thoroughly as an unresolved id."
        )
    return joined.with_columns(
        pl.col("ros_points").alias("anchor_points"),
        pl.lit(weight).alias("model_weight"),
    ).with_columns(
        pl.when(pl.col("model_points").is_null())
        .then(pl.col("anchor_points"))
        .otherwise(
            (1 - weight) * pl.col("anchor_points") + weight * pl.col("model_points")
        )
        .alias("ros_points")
    )


def assert_one_row_per_player(frame: pl.DataFrame, stage: str) -> pl.DataFrame:
    """§0.2 at every boundary. A fan-out is as fatal as an unresolved id.

    M7's data-validator pass found ``model.project`` emitting one row per
    (player, prior NFL team), which fanned out through two joins and put fifteen
    players on the board four times each. The engine marks pool INDICES taken,
    not people, so two teams could draft the same person. Same class of bug is
    available here on every weekly join.
    """
    dupes = frame.group_by("canonical_id").len().filter(pl.col("len") > 1)
    if dupes.height:
        raise ROSError(
            f"{stage}: {dupes.height} player(s) appear more than once, e.g. "
            f"{dupes.head(5).to_dicts()}.\n"
            f"  §0.2 is not only about resolution — a join that fans out breaks "
            f"it just as thoroughly."
        )
    return frame


NUMERIC_SCHEMA = {
    "ros_points": pl.Float64, "weekly_points": pl.Float64,
    "anchor_points": pl.Float64, "model_points": pl.Float64,
    "sack_correction": pl.Float64, "kicker_correction": pl.Float64,
    "games_remaining": pl.Int64, "bye_week": pl.Int64,
    "espn_projection": pl.Float64,
}
"""Pinned dtypes, because the engines CONCATENATE constantly.

A waiver evaluation splices a free-agent row into a roster frame; a trade
splices two rosters together. polars refuses to vstack Int32 onto Int64, so a
column whose width came out of an arithmetic expression rather than a literal
schema is a crash waiting for the first real run — which is exactly how this
was found. Normalising once here beats casting at every call site.
"""


def normalize_schema(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns([
        pl.col(c).cast(t) for c, t in NUMERIC_SCHEMA.items() if c in frame.columns
    ])


def build(
    anchor: pl.DataFrame,
    from_week: int,
    model: pl.DataFrame | None = None,
    weight: float = MODEL_WEIGHT,
    season: int = SEASON,
) -> ROSProjection:
    """Assemble the rest-of-season projection from a points anchor.

    ``anchor`` is ESPN's per-week projected points summed over the remaining
    weeks, already resolved to canonical ids — see ``state.py`` for the read.
    Kept as an argument rather than fetched here so the arithmetic is testable
    without credentials (§0.3).
    """
    missing = [c for c in REQUIRED_ANCHOR if c not in anchor.columns]
    if missing:
        raise ROSError(f"anchor is missing {missing}; needs {list(REQUIRED_ANCHOR)}.")
    assert_one_row_per_player(anchor, "the rest-of-season anchor")

    weeks = remaining_weeks(from_week, season)
    notes: list[str] = []

    out = blend(anchor, model, weight)
    out = sack_correction(out)
    out = kicker_correction(out)
    out = out.with_columns(
        (pl.col("ros_points") + pl.col("sack_correction") + pl.col("kicker_correction"))
        .round(2).alias("ros_points")
    )
    out = weekly_rate(out)
    out = normalize_schema(out)
    assert_one_row_per_player(out, "the finished rest-of-season projection")

    if weight == 0.0:
        notes.append(
            "model weight is 0.0 — shipping the ESPN points anchor plus the two "
            "league-specific corrections. M10b-2's gate sets this, not intuition."
        )
    sacked = out.filter(pl.col("sack_correction").abs() > 0.5)
    if sacked.height:
        notes.append(
            f"{sacked.height} quarterback(s) carry a sack correction worth more "
            f"than half a point (§3.3 — nobody else prices this)."
        )
    return ROSProjection(
        frame=out.sort("ros_points", descending=True, nulls_last=True),
        season=season, from_week=from_week, weeks_remaining=weeks,
        model_weight=weight, notes=notes,
    )


# ─── Building the anchor from live ESPN projections ──────────────────────────
def from_espn(
    projections: pl.DataFrame,
    from_week: int,
    byes: pl.DataFrame | None = None,
    soe: pl.DataFrame | None = None,
    season: int = SEASON,
    weight: float = MODEL_WEIGHT,
) -> ROSProjection:
    """Per-week ESPN projections -> the rest-of-season anchor.

    ``projections`` is the long frame from ``data/espn.py::player_projections``,
    already resolved to canonical ids: one row per (player, week) carrying
    ``projected_points``.

    Summing ESPN's own per-week numbers is F1's resolution in practice. It also
    sidesteps M3's trap on the SEASON projection, which reports yardage per game
    while every other field is a season total — read literally that made every
    projection roughly 17x wrong. Per-week points carry no such asymmetry, and
    they are already in §1 scoring.
    """
    need = {"canonical_id", "position", "team", "week", "projected_points"}
    missing = sorted(need - set(projections.columns))
    if missing:
        raise ROSError(f"projections missing {missing}; needs {sorted(need)}.")

    weeks = remaining_weeks(from_week, season)
    if not weeks:
        raise ROSError(
            f"no regular-season weeks remain from week {from_week}. "
            f"Rest-of-season value is undefined once the season is over."
        )

    ahead = projections.filter(pl.col("week").is_in(list(weeks)))
    if ahead.is_empty():
        raise ROSError(
            f"no projection rows for weeks {weeks[0]}-{weeks[-1]}. ESPN may not "
            f"have published them yet, which is not the same as projecting zero."
        )

    # ─────────────────────────────────────────────────────────────────────
    # WHY THE ANCHOR IS THE SEASON PROJECTION AND NOT THE PUBLISHED WEEKS
    #
    # ESPN publishes per-week projections only for weeks it has actually
    # projected — in practice the upcoming one. Measured on this league at week
    # 1 of 2026: the MEDIAN player had 1 of 14 remaining weeks published.
    #
    # Two earlier versions both failed on that, in opposite directions:
    #
    #   1. Summing the window and treating unpublished weeks as ZERO made every
    #      player roughly 1/14th of himself, and unevenly so.
    #   2. Taking the MEAN of the published weeks and extending it fixed the
    #      scale but not the substance, because a one-week mean is a WEEKLY
    #      projection wearing a season's clothes. It carries that week's
    #      opponent, that week's snap projection and that week's injury
    #      designation — and then multiplies all three by fourteen. A starting
    #      receiver ESPN had projected at 0.0 for week 1 because he was out
    #      became a player worth zero for the SEASON, and therefore the cheapest
    #      thing on the roster to cut. It recommended dropping Brian Thomas Jr.
    #      for a kicker.
    #
    # ESPN answers the rest-of-season question directly, at scoring period 0:
    # a full-season projected total in this league's scoring. That is the
    # anchor. The per-week numbers are kept — as ``espn_projection``, this
    # week's number — and belong to the WEEKLY jobs, where they are exactly
    # right. This is F1's own resolution applied one level down: match the
    # source to the horizon of the question instead of stretching one to cover
    # the other.
    #
    # NULL and 0.0 remain different facts at both horizons. A null is a week or
    # a season ESPN has not published; a published 0.0 is what ESPN says about a
    # player who is out. The first must never be filled, the second must never
    # be discarded.
    #
    # The bye week is excluded from the RATE rather than counted as a zero,
    # because games_remaining already excludes it — counting it in both places
    # would charge for the same bye twice.
    # ─────────────────────────────────────────────────────────────────────
    n_weeks = len(weeks)
    if byes is not None and "bye_week" in byes.columns:
        ahead = ahead.join(byes.select("team", "bye_week"), on="team", how="left")
    else:
        ahead = ahead.with_columns(pl.lit(None, dtype=pl.Int64).alias("bye_week"))

    on_bye = pl.col("bye_week").is_not_null() & (pl.col("week") == pl.col("bye_week"))
    scored = pl.col("projected_points").is_not_null() & ~on_bye
    this_week = pl.col("week") == from_week

    have_season = "season_projected_points" in ahead.columns
    season_aggs = [
        pl.col("season_projected_points").first().alias("season_projected"),
        pl.col("season_actual_points").first().alias("season_actual"),
        pl.col("season_projected_avg").first().alias("season_avg"),
    ] if have_season else []

    agg = ahead.group_by("canonical_id").agg(
        pl.col("position").first(),
        pl.col("team").first(),
        pl.col("bye_week").first(),
        pl.col("name").first() if "name" in ahead.columns else pl.lit(None).alias("name"),
        pl.col("projected_points").filter(scored).mean().alias("_week_rate"),
        pl.col("projected_points").filter(scored).len().alias("weeks_projected"),
        pl.col("projected_points").filter(this_week).first().alias("espn_projection"),
        *season_aggs,
    )
    if not have_season:
        agg = agg.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("season_projected"),
            pl.lit(None, dtype=pl.Float64).alias("season_actual"),
            pl.lit(None, dtype=pl.Float64).alias("season_avg"),
        )

    notes: list[str] = []

    # ESPN's season projection covers the NFL's 17 GAMES over 18 weeks, so the
    # rate it implies is divided by games left in the NFL season — not by
    # `games_remaining`, which is my fantasy window (weeks `from_week`..14) and
    # is what the rate is multiplied BY on the next line. Conflating the two
    # would inflate every rate by the ratio of the windows.
    elapsed = max(from_week - 1, 0)
    agg = agg.with_columns(
        pl.when(pl.col("bye_week").is_not_null() & (pl.col("bye_week") < from_week))
        .then(pl.lit(elapsed - 1)).otherwise(pl.lit(elapsed))
        .clip(lower_bound=0).alias("_nfl_games_played")
    ).with_columns(
        (GAMES_IN_NFL_SEASON - pl.col("_nfl_games_played"))
        .clip(lower_bound=1).alias("_nfl_games_left"),
        (
            n_weeks
            - pl.when(pl.col("bye_week").is_in(list(weeks))).then(1).otherwise(0)
        ).cast(pl.Int64).alias("games_remaining"),
    ).with_columns(
        # What ESPN says is LEFT: the full-season projection minus what has
        # already been scored. Preseason that is the whole projection; in week 9
        # it is weeks 9-18. Correct in both regimes without a special case.
        (
            (pl.col("season_projected") - pl.col("season_actual").fill_null(0.0))
            .clip(lower_bound=0.0) / pl.col("_nfl_games_left")
        ).alias("_season_rate")
    ).with_columns(
        pl.coalesce(pl.col("_season_rate"), pl.col("_week_rate")).alias("_rate"),
        pl.when(pl.col("_season_rate").is_not_null()).then(pl.lit("season"))
        .when(pl.col("_week_rate").is_not_null()).then(pl.lit("week"))
        .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("anchor_source"),
    ).with_columns(
        (pl.col("_rate") * pl.col("games_remaining")).round(2).alias("ros_points")
    )

    if soe is not None and "sacks_over_expected_per_game" in soe.columns:
        agg = agg.join(
            soe.select("canonical_id", "sacks_over_expected_per_game"),
            on="canonical_id", how="left",
        )
    else:
        agg = agg.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("sacks_over_expected_per_game"))
        # Deliberately silent: the CALLER knows WHY there is no input — "it is
        # week 2" reads very differently from "the cache is missing" — and says
        # so. Adding a second, vaguer sentence here printed the same warning
        # twice in every digest.

    dead = agg.filter(pl.col("games_remaining") <= 0)
    if dead.height:
        agg = agg.filter(pl.col("games_remaining") > 0)
        notes.append(
            f"{dead.height} player(s) have no games left in the window and were "
            f"dropped rather than divided by zero."
        )

    if not have_season:
        notes.append(
            "the projection table predates the season-projection columns, so "
            "rest-of-season value is one published week extended over the "
            "season. That is a WEEKLY number doing a season's job — refresh "
            "before trusting any drop it suggests."
        )

    # M3's "read literally, every projection is 17x wrong" trap, checked in the
    # new place rather than assumed absent. ESPN ships appliedAverage beside
    # appliedTotal, so the two readings can be compared directly: if the season
    # total were secretly a per-game number the ratio would sit near 17, not 1.
    check = agg.filter(
        pl.col("season_projected").is_not_null()
        & pl.col("season_avg").is_not_null()
        & (pl.col("season_projected") > 20.0)
        & (pl.col("season_avg") > 0.0)
    )
    if check.height:
        ratio = float(
            (check["season_projected"] / GAMES_IN_NFL_SEASON / check["season_avg"])
            .median() or 0.0
        )
        if not 0.7 <= ratio <= 1.4:
            notes.append(
                f"ESPN's season total / {GAMES_IN_NFL_SEASON} is {ratio:.2f}x its "
                f"own per-game average, which should be about 1.0. One of the two "
                f"is not what it claims to be (M3 found exactly this on the "
                f"yardage fields). Treat every rest-of-season number below as "
                f"suspect until it is resolved."
            )

    # Coverage is stated, because "ESPN has projected one of fourteen weeks" and
    # "this player is projected to score very little" look identical in a total.
    covered = agg.filter(pl.col("weeks_projected") > 0)
    median_cov = int(covered["weeks_projected"].median() or 0) if covered.height else 0
    if median_cov < n_weeks:
        notes.append(
            f"ESPN has published per-week projections for a median of "
            f"{median_cov} of the {n_weeks} remaining weeks, so those serve the "
            f"WEEKLY lineup call only. Rest-of-season value comes from ESPN's "
            f"season projection instead — a one-week number extended over "
            f"fourteen carries that week's matchup and injury with it."
        )

    # The players the change actually rescues, named. A season anchor above
    # replacement paired with a weekly rate at nothing is the Brian Thomas Jr.
    # shape: out this week, fine for the season, and previously worth zero.
    rescued = agg.filter(
        (pl.col("anchor_source") == "season")
        & pl.col("_week_rate").is_not_null()
        & (pl.col("_week_rate") <= 1.0)
        & (pl.col("_rate") >= 5.0)
    ).sort("_rate", descending=True)
    if rescued.height:
        who = ", ".join(
            f"{r['name'] or r['canonical_id']} ({r['_rate']:.1f}/wk)"
            for r in rescued.head(5).iter_rows(named=True)
        )
        notes.append(
            f"{rescued.height} player(s) are projected at ~0 for week {from_week} "
            f"but well above that for the season: {who}"
            + ("..." if rescued.height > 5 else "")
            + ". They are priced on the season, not on this week — sit them, do "
            f"not drop them."
        )

    blank = agg.filter(pl.col("anchor_source").is_null())
    if blank.height:
        agg = agg.filter(pl.col("anchor_source").is_not_null())
        notes.append(
            f"{blank.height} player(s) have no projection at all, weekly or "
            f"season, and were dropped rather than priced at zero."
        )
    if agg.is_empty():
        raise ROSError(
            f"ESPN has published no projections at all for weeks "
            f"{weeks[0]}-{weeks[-1]}. Refusing to price a roster at zero — that "
            f"is not the same as projecting zero."
        )
    agg = agg.drop("_week_rate", "_season_rate", "_rate",
                   "_nfl_games_played", "_nfl_games_left")

    out = build(agg, from_week=from_week, weight=weight, season=season)
    out.notes = notes + out.notes
    return out


# ─── §3.3's sack term, from live current-season play-by-play ─────────────────
MIN_QB_GAMES_FOR_SOE = 3
"""Below this, a quarterback's sacks-over-expected is one bad afternoon.

M3 measured SOE's year-over-year persistence at 0.394 and M3b its carry-forward
at 0.434 — both on FULL seasons. Two games of it is noise, and noise multiplied
by nine remaining games is a confidently wrong number pointed at exactly the
position §9.3 says to spend waiver priority on.
"""


def live_sacks_over_expected(
    season: int, min_games: int = MIN_QB_GAMES_FOR_SOE
) -> tuple[pl.DataFrame | None, str | None]:
    """Current-season SOE per game, or ``(None, why_not)``.

    Returns a REASON rather than an empty frame when it cannot be computed, so
    the digest can say "the sack term is dark because it is week 2" instead of
    silently pricing every quarterback as league-average. §3.3 calls this the
    cheap, durable edge nobody else prices; a silent zero is the one way to
    lose it without noticing.
    """
    try:
        from ff_agent.projections import xfp as X

        per_week = X.expected_sacks(season)
    except Exception as exc:
        return None, (
            f"sacks-over-expected unavailable ({type(exc).__name__}), so §3.3's "
            f"term is zero for every quarterback. That is the one edge this "
            f"league has that no public ranking prices."
        )
    if per_week is None or per_week.is_empty():
        return None, (
            "no current-season play-by-play yet, so §3.3's sack term is zero. "
            "It switches on by itself once a few games exist."
        )

    agg = (
        per_week.group_by("player_id")
        .agg(
            pl.len().alias("games"),
            pl.col("sacks_over_expected").sum().alias("soe_total"),
        )
        .filter(pl.col("games") >= min_games)
        .with_columns(
            (pl.col("soe_total") / pl.col("games")).round(3)
            .alias("sacks_over_expected_per_game"),
            pl.col("player_id").alias("canonical_id"),
        )
        .select("canonical_id", "games", "sacks_over_expected_per_game")
    )
    if agg.is_empty():
        return None, (
            f"no quarterback has {min_games} games yet, so §3.3's sack term is "
            f"still zero. Two games of sacks-over-expected is one bad afternoon, "
            f"not a trait — it switches on by itself shortly."
        )
    return agg, None
