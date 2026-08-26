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

    # A collision — two ESPN players on one canonical id — is NOT a soft failure.
    # It silently merges two people, which is worse than not resolving one.
    collisions = ok.group_by("canonical_id").len().filter(pl.col("len") > 1)
    if collisions.height:
        rows = ok.filter(pl.col("canonical_id").is_in(
            collisions["canonical_id"].to_list())).sort("canonical_id")
        raise StateError(
            f"{label}: {collisions.height} canonical id(s) claimed by more than "
            f"one ESPN player — two people silently merged into one.\n"
            f"{rows.select('espn_id', 'name', 'position', 'canonical_id')}\n"
            f"  Fix with an explicit override row (§0.2); never by picking one."
        )
    return ok, bad


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
