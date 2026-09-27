"""League state as of a given week — the read half of the monitor.

**§0.1 lives here.** Everything in this module reads. There are no write paths
and there never will be: ``test_inseason_package_cannot_write_to_espn`` scans
this package for the same verbs ``ff_agent/live/`` is already scanned for. M9
recorded that the rule needed a SECOND guard once a browser was involved; an
unattended container on a schedule is the third place it could break and the
first where nobody would notice.

Three things this module refuses to do quietly:

  * **Serve stale data.** §10: an optimizer silently running week-3 data in
    week 8 is worse than no optimizer. In-season that becomes a specific
    assertion — the data must cover the most recently COMPLETED week.
  * **Drop a player it cannot resolve.** §0.2's gate is HARDER in-season than it
    was in the preseason: M1 closed it against five finite, known populations,
    while the in-season free-agent pool grows every week with practice-squad
    callups nflverse has never issued a ``gsis_id`` for. They are REPORTED, in
    their own section, and go nowhere near a recommendation.
  * **Trust a team name.** Names are mutable and this league has renamed teams
    three times mid-project. Identity is manager -> team_id -> the table's own
    spelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ff_agent.config import MY_TEAM_NAME, SEASON
from ff_agent.data import crosswalk as CW


class StateError(RuntimeError):
    """League state cannot be trusted. Always blocking (§10)."""


@dataclass
class LeagueState:
    """Everything a job needs about the league, as of ``week``."""

    season: int
    week: int
    rosters: pl.DataFrame
    """One row per rostered player: fantasy_team, manager, team_id,
    canonical_id, espn_id, name, position, team, lineup_slot, injury_status."""
    free_agents: pl.DataFrame
    """Same shape, for everyone unrostered. Resolved, never guessed."""
    unresolved: pl.DataFrame
    """§0.2. Players we could not resolve — reported, never dropped silently."""
    waiver_order: pl.DataFrame
    """team_id, fantasy_team, waiver_rank, wins, losses. §9.3 needs all nine."""
    completed: pl.DataFrame
    """[week, team, points] for weeks already played. Feeds the simulator."""
    my_team: str = MY_TEAM_NAME
    notes: list[str] = field(default_factory=list)

    # ── convenience ────────────────────────────────────────────────────────
    @property
    def my_roster(self) -> pl.DataFrame:
        return self.rosters.filter(pl.col("fantasy_team") == self.my_team)

    @property
    def managers(self) -> list[str]:
        return sorted(
            m for m in self.rosters["fantasy_team"].unique().to_list() if m
        )

    def roster_of(self, team: str) -> pl.DataFrame:
        return self.rosters.filter(pl.col("fantasy_team") == team)

    def rostered_ids(self) -> set[str]:
        return set(self.rosters["canonical_id"].drop_nulls().to_list())

    def position_counts(self, team: str) -> dict[str, int]:
        r = self.roster_of(team)
        return {
            str(k[0]): int(v)
            for k, v in zip(*[
                r.group_by("position").len()["position"].to_list(),
                r.group_by("position").len()["len"].to_list(),
            ])
        } if r.height else {}


# ─── §0.2 resolution, with the in-season twist ───────────────────────────────
RESOLVE_COLUMNS = ("espn_id", "name", "position", "team")


def resolve(
    players: pl.DataFrame,
    label: str,
    canonical: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Resolve ESPN rows to canonical ids. Returns ``(resolved, unresolved)``.

    Deliberately does NOT call ``assert_all_resolved``. In the preseason the
    population was finite and known, so an unmatched player was a bug to fix
    before the draft. In-season the free-agent pool grows every week with people
    nflverse has never issued an id for, and refusing to produce a Tuesday digest
    because a practice-squad tight end was promoted would be the tool failing at
    its job.

    So the rule bends in exactly one direction and no further: unresolved players
    are **carried out separately and reported by name**, and they never reach a
    recommendation. They are never name-matched, never dropped, never defaulted.
    The fix stays what §0.2 says it is — an explicit override row a human wrote.
    """
    missing = [c for c in RESOLVE_COLUMNS if c not in players.columns]
    if missing:
        raise StateError(f"{label}: missing {missing}; needs {list(RESOLVE_COLUMNS)}.")
    if players.is_empty():
        empty = players.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("canonical_id"),
            pl.lit(None, dtype=pl.Utf8).alias("match_method"),
        )
        return empty, empty

    skill = players.filter(~pl.col("position").map_elements(
        lambda p: CW.is_dst(p), return_dtype=pl.Boolean))
    dst = players.filter(pl.col("position").map_elements(
        lambda p: CW.is_dst(p), return_dtype=pl.Boolean))

    parts = []
    if skill.height:
        parts.append(CW.resolve_players(skill, canonical))
    if dst.height:
        parts.append(CW.resolve_dst(dst))
    out = pl.concat(parts, how="diagonal_relaxed") if parts else players

    ok = out.filter(
        pl.col("canonical_id").is_not_null() & (pl.col("match_method") != "unresolved")
    )
    bad = out.filter(
        pl.col("canonical_id").is_null() | (pl.col("match_method") == "unresolved")
    )

    # A collision — two DIFFERENT ESPN players on one canonical id — is NOT a
    # soft failure. It silently merges two people, which is worse than not
    # resolving one.
    #
    # Counted over DISTINCT espn_ids, not rows. Counting rows makes any LONG
    # frame look like a mass collision: the per-week projection table carries one
    # row per (player, week), so a 14-week season reported all 553 players as
    # colliding with themselves. Philip Rivers appearing fourteen times is
    # fourteen weeks of Philip Rivers, not fourteen Philip Riverses.
    collisions = (
        ok.group_by("canonical_id")
        .agg(pl.col("espn_id").n_unique().alias("n_players"))
        .filter(pl.col("n_players") > 1)
    )
    if collisions.height:
        rows = (
            ok.filter(pl.col("canonical_id").is_in(collisions["canonical_id"].to_list()))
            .select("espn_id", "name", "position", "canonical_id")
            .unique()
            .sort("canonical_id", "espn_id")
        )
        raise StateError(
            f"{label}: {collisions.height} canonical id(s) claimed by more than "
            f"one ESPN player — two people silently merged into one.\n"
            f"{rows}\n"
            f"  Fix with an explicit override row (§0.2); never by picking one."
        )
    return ok, bad


def resolve_long(
    long_frame: pl.DataFrame,
    label: str,
    key: str = "espn_id",
    canonical: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Resolve a frame carrying MANY rows per player — e.g. one row per week.

    Resolution is a question about PEOPLE, so it is asked once per person and the
    answer joined back. Passing a long frame to ``resolve`` directly also does
    553 lookups fourteen times over, which is slow as well as wrong.
    """
    people = long_frame.select(
        key, "name", "position", "team"
    ).unique(subset=[key], keep="first")
    ok, bad = resolve(people, label, canonical)
    if ok.is_empty():
        return long_frame.head(0).with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("canonical_id")), bad
    joined = long_frame.join(
        ok.select(key, "canonical_id"), on=key, how="inner"
    )
    return joined, bad


def unresolved_note(unresolved: pl.DataFrame, label: str) -> str | None:
    """The line the digest prints. Names, not a count — a count is ignorable."""
    if unresolved.is_empty():
        return None
    names = unresolved.head(8).select("name", "position", "espn_id").to_dicts()
    listed = ", ".join(f"{r['name']} ({r['position']}, espn {r['espn_id']})" for r in names)
    more = f" and {unresolved.height - 8} more" if unresolved.height > 8 else ""
    return (
        f"{unresolved.height} {label} could not be resolved to a canonical id and "
        f"were NOT evaluated: {listed}{more}. "
        f"Add an override row in overrides/player_id_overrides.csv (§0.2)."
    )


# ─── Staleness, in the in-season sense ───────────────────────────────────────
def assert_covers_completed_week(
    completed: pl.DataFrame, through_week: int, label: str = "results"
) -> None:
    """§10's stale-data rule, made specific.

    A digest built without last week's results is the "week-3 data in week 8"
    failure wearing a fresh timestamp: every cache file is young, every fetch
    succeeded, and the numbers are still answering last month's question.
    """
    if through_week < 1:
        return
    have = set(int(w) for w in completed["week"].unique().to_list()) if completed.height else set()
    missing = sorted(set(range(1, through_week + 1)) - have)
    if missing:
        raise StateError(
            f"{label}: weeks {missing} have been played but carry no results.\n"
            f"  Refusing to build a digest on them. §10: an optimizer silently\n"
            f"  running week-3 data in week 8 is worse than no optimizer.\n"
            f"  Fix: run `cli monitor --job refresh` to ingest completed weeks."
        )


# ─── Roster shape, shared by the waiver and trade engines ────────────────────
from ff_agent.config import POSITION_MAXIMA, ROSTER_TOTAL, STARTER_SLOTS

FLEX_ELIGIBLE = ("RB", "WR", "TE")


def starter_holes(counts: dict[str, int]) -> list[str]:
    """Starting slots this roster cannot fill (§1's ten starters).

    Used twice and it matters that both uses agree: §9.3 estimates a rival's
    interest in a waiver claim from their holes, and §9.4 finds trades from the
    same shape. Two implementations would let the waiver engine and the trade
    engine believe different things about the same roster.
    """
    have = dict(counts)
    holes: list[str] = []
    for pos, need in STARTER_SLOTS.items():
        if pos == "FLEX":
            continue
        short = need - have.get(pos, 0)
        if short > 0:
            holes += [pos] * short
        have[pos] = max(0, have.get(pos, 0) - need)
    spare = sum(have.get(p, 0) for p in FLEX_ELIGIBLE)
    if STARTER_SLOTS.get("FLEX", 0) > spare:
        holes += ["FLEX"] * (STARTER_SLOTS["FLEX"] - spare)
    return holes


def can_add(counts: dict[str, int], position: str) -> bool:
    """§1's position maxima. Without them M7's failure returns — eight
    quarterbacks, no tight end, about fifty points a week."""
    cap = POSITION_MAXIMA.get(position)
    return cap is None or counts.get(position, 0) < cap


def droppable(
    roster: pl.DataFrame, position_of_add: str | None = None
) -> pl.DataFrame:
    """Players who can be dropped without making the lineup unfillable.

    §10 lists an unfillable starting lineup as a sign something is broken, so the
    waiver engine is not allowed to propose one. A drop is refused when it would
    leave a strict slot with nobody in it.
    """
    if roster.is_empty():
        return roster
    counts: dict[str, int] = {}
    for p in roster["position"].to_list():
        counts[p] = counts.get(p, 0) + 1

    keep = []
    for row in roster.iter_rows(named=True):
        after = dict(counts)
        after[row["position"]] -= 1
        if position_of_add:
            after[position_of_add] = after.get(position_of_add, 0) + 1
        keep.append(not starter_holes(after))
    return roster.filter(pl.Series(keep))


# ─── Building a real LeagueState from live ESPN ──────────────────────────────
def _normalize(df: pl.DataFrame) -> pl.DataFrame:
    """ESPN's spellings -> ours, at the boundary rather than at each join.

    The Rams trap bit this project three times by being fixed downstream, so
    team AND position vocabulary are canonicalised once, here.
    """
    from ff_agent.inseason import freeagents as FA

    return df.with_columns(
        pl.col("position").map_elements(FA.normalize_position, return_dtype=pl.Utf8),
        pl.col("team").map_elements(CW.normalize_team, return_dtype=pl.Utf8),
    )


def load(
    season: int = SEASON,
    week: int | None = None,
    my_team: str = MY_TEAM_NAME,
    free_agent_keep: int = 120,
    offline: bool | None = None,
) -> LeagueState:
    """Read the whole league as of ``week``. The read half of every job.

    Fails loudly on anything that would make a recommendation wrong rather than
    merely incomplete: an unresolvable ROSTERED player is fatal (that is M1's
    gate, and a rostered player we cannot price silently distorts my own lineup),
    while an unresolvable FREE AGENT is reported and skipped.
    """
    from ff_agent.data import espn as ESPN
    from ff_agent.data import byes as BY
    from ff_agent.inseason import clock as CK
    from ff_agent.inseason import freeagents as FA

    notes: list[str] = []
    kw = {} if offline is None else {"offline": offline}

    if week is None:
        week = CK.current_week(season) or 1

    rosters_raw = _normalize(ESPN.current_rosters(season, **kw))
    rosters, ros_bad = resolve(rosters_raw, "rosters")
    if ros_bad.height:
        # M1's gate, unchanged: a rostered player who cannot be priced distorts
        # MY lineup, which is the one thing every job depends on.
        raise StateError(
            f"{ros_bad.height} ROSTERED player(s) could not be resolved to a "
            f"canonical id.\n{unresolved_note(ros_bad, 'rostered players')}\n"
            f"  Unlike a free agent, a rostered player cannot be skipped — he is "
            f"in somebody's starting lineup and every number here depends on it."
        )

    proj = _normalize(FA.playable(ESPN.player_projections(season, **kw)))
    rostered_ids = set(rosters_raw["espn_id"].to_list())
    fa_raw = FA.pool(proj, rostered_ids).unique(subset=["espn_id"], keep="first")
    free_agents, fa_bad = resolve(fa_raw, "free agents")
    note = unresolved_note(fa_bad, "free agents")
    if note:
        notes.append(note)

    try:
        waivers = ESPN.waiver_order(season, **kw)
    except Exception as exc:
        waivers = pl.DataFrame(schema={
            "team_id": pl.Int64, "fantasy_team": pl.Utf8, "waiver_rank": pl.Int64})
        notes.append(
            f"waiver priority unavailable ({type(exc).__name__}) — claims are "
            f"ordered but their odds are not."
        )

    completed = ESPN.weekly_results(season, through_week=max(week - 1, 0), **kw)
    if week > 1:
        assert_covers_completed_week(completed, week - 1, "ESPN results")

    byes = BY.bye_weeks(season).select(
        pl.col("team"), pl.col("bye_week").cast(pl.Int64))
    rosters = rosters.join(byes, on="team", how="left")
    free_agents = free_agents.join(byes, on="team", how="left")

    return LeagueState(
        season=season, week=week, rosters=rosters, free_agents=free_agents,
        unresolved=fa_bad, waiver_order=waivers, completed=completed,
        my_team=my_team, notes=notes,
    )


def with_values(
    frame: pl.DataFrame, ros_frame: pl.DataFrame, label: str = "roster"
) -> pl.DataFrame:
    """Join a roster or FA pool to its rest-of-season numbers.

    Separate from ``load`` on purpose: identity and value are different
    questions, and keeping them apart is what lets every engine be tested with
    made-up numbers against a real roster shape, or vice versa.
    """
    cols = [c for c in (
        "ros_points", "weekly_points", "anchor_points", "anchor_weekly",
        "sack_correction", "kicker_correction", "games_remaining",
        "espn_projection",
    ) if c in ros_frame.columns]
    out = frame.join(
        ros_frame.select("canonical_id", *cols), on="canonical_id", how="left"
    )
    if out.height != frame.height:
        raise StateError(
            f"{label}: the value join changed the row count ({frame.height} -> "
            f"{out.height}). §0.2 — a fan-out breaks one-row-per-player as "
            f"thoroughly as an unresolved id."
        )
    from ff_agent.inseason.ros import normalize_schema

    # `priced` separates "ESPN says he will score nothing" from "we have no
    # number for him". Both arrive as a zero once filled, and they mean opposite
    # things: the first is a legitimate drop candidate, the second is a player we
    # know nothing about and must not offer to cut. Filling silently made an
    # unpriced starter the single most attractive thing on the roster to drop.
    out = out.with_columns(
        pl.col("weekly_points").is_not_null().alias("priced")
    ).with_columns(
        pl.col("weekly_points").fill_null(0.0),
        pl.col("ros_points").fill_null(0.0),
    )
    # Same reason as ros.normalize_schema: rosters and free agents are spliced
    # together constantly, and a left join can widen a column's dtype.
    return normalize_schema(out)


# ─── The column contract the engines splice against ──────────────────────────
ENGINE_COLUMNS = (
    "canonical_id", "name", "position", "team", "weekly_points", "ros_points",
    "bye_week", "anchor_points", "anchor_weekly", "sack_correction",
    "kicker_correction", "games_remaining", "lineup_slot", "priced",
    "espn_projection",
)
"""Exactly what a roster row and a free-agent row must BOTH carry.

The waiver engine splices a free agent into a roster frame to score an (add,
drop) pair, and the trade engine splices two rosters together. Those frames come
from different ESPN endpoints with different columns — the roster carries
``team_id``/``manager``, the free agent carries ``week``/``projected_points`` —
so without a contract the splice fails on the first real run with a
ColumnNotFoundError from deep inside polars. Which is how this was found.
"""


def unpriced(frame: pl.DataFrame) -> pl.DataFrame:
    """Rostered players we have no projection for. Never droppable, always said."""
    if "priced" not in frame.columns:
        return frame.head(0)
    return frame.filter(~pl.col("priced").fill_null(False))


def align(frame: pl.DataFrame, label: str = "frame") -> pl.DataFrame:
    """Reduce a frame to the engine contract, filling anything absent.

    Fills rather than raises, because the two sides legitimately differ: a free
    agent has no ``lineup_slot`` and never will. What must not differ is the
    SHAPE.
    """
    if frame.is_empty():
        return pl.DataFrame(schema={c: _ENGINE_DTYPES[c] for c in ENGINE_COLUMNS})
    out = frame
    for col in ENGINE_COLUMNS:
        if col not in out.columns:
            out = out.with_columns(
                pl.lit(None, dtype=_ENGINE_DTYPES[col]).alias(col))
    return out.select(list(ENGINE_COLUMNS)).with_columns([
        pl.col(c).cast(t) for c, t in _ENGINE_DTYPES.items()
    ])


_ENGINE_DTYPES = {
    "canonical_id": pl.Utf8, "name": pl.Utf8, "position": pl.Utf8,
    "team": pl.Utf8, "weekly_points": pl.Float64, "ros_points": pl.Float64,
    "bye_week": pl.Int64, "anchor_points": pl.Float64,
    "anchor_weekly": pl.Float64, "sack_correction": pl.Float64,
    "kicker_correction": pl.Float64, "games_remaining": pl.Int64,
    "lineup_slot": pl.Utf8, "priced": pl.Boolean,
    "espn_projection": pl.Float64,
}
