"""The in-season free-agent pool, and §0.2's harder in-season shape.

The pool is simply everyone ESPN knows about who is not on one of the nine
rosters. What makes it interesting is that it is **not a finite, known
population** the way M1's five test sets were: it grows every week with
practice-squad callups and street free agents nflverse has never issued a
``gsis_id`` for.

§0.2 does not bend for them, but it does bend in exactly one direction: an
unresolvable free agent is **reported by name** and never evaluated, rather than
blocking the whole digest. Refusing to send a Tuesday claim list because a
promoted tight end appeared on the wire would be the tool failing at its job;
quietly name-matching him would be the failure §0.2 exists to prevent.
"""

from __future__ import annotations

import polars as pl

from ff_agent.config import FANTASY_POSITIONS

ESPN_POSITION_ALIASES = {"D/ST": "DST", "DST": "DST", "PK": "K", "K": "K"}
"""ESPN spells a team defense ``D/ST``; every engine here says ``DST``.

Enumerated rather than inferred, for the same reason the name suffixes in M9
were: a fixed, listed transformation is not the fuzzy matching §0.2 forbids.
"""


def normalize_position(pos: str | None) -> str | None:
    if not pos:
        return None
    p = pos.strip().upper()
    return ESPN_POSITION_ALIASES.get(p, p)


def playable(df: pl.DataFrame) -> pl.DataFrame:
    """Drop positions this project does not project.

    §10 and M9's IDP refusal: the projections cover QB/RB/WR/TE/K/D-ST, so a
    linebacker on the wire gets left alone rather than given a confident wrong
    number.
    """
    return df.filter(pl.col("position").is_in(sorted(FANTASY_POSITIONS)))


def pool(
    projections: pl.DataFrame,
    rostered_espn_ids: set[str],
) -> pl.DataFrame:
    """Everyone with a projection who is not on a roster.

    Keyed on ESPN id rather than name, obviously — but worth stating, because
    the one place this could go wrong is a player who was just dropped and whose
    projection row still carries a stale ``on_team_id``. Membership in the LIVE
    roster set is the authority, not the projection's own idea of who owns him.
    """
    if projections.is_empty():
        return projections
    return projections.filter(~pl.col("espn_id").is_in(list(rostered_espn_ids)))


def rank_by_upside(fa: pl.DataFrame, keep: int = 60) -> pl.DataFrame:
    """Trim the pool to what a claim could plausibly be made on.

    §3.2 is why this is a trim and not a filter on ownership: "draft for upside,
    not floor — a safe pick is replaceable from the wire, a league-winner isn't."
    Ranking on projected points keeps the league-winners; ranking on percent
    owned would keep whoever is already popular, which is the control (F4), not
    the strategy.
    """
    if fa.is_empty():
        return fa
    return fa.sort("ros_points", descending=True, nulls_last=True).head(keep)
