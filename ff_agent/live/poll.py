"""Read-only poll of ESPN's draft feed.

§6 is explicit that ESPN's live draft feed is flaky, so **this is a convenience,
never the source of truth**. The human typing a name is the primary path; this
just saves typing when it happens to work. Every failure mode here is designed
to degrade to "nothing new", never to a crash and never to a wrong pick.

§0.1: there is no write path in this module. It reads ``league.draft`` and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ff_agent.draft import pool as PL


@dataclass
class PollResult:
    """What the feed says, already mapped onto pool indices."""

    indices: list[int]          # pool indices, in ESPN's pick order
    n_feed_picks: int           # how many picks the feed reported
    unresolved: list[str]       # feed picks we could not place (names)
    error: str = ""             # populated when the feed could not be read

    @property
    def ok(self) -> bool:
        return not self.error


def espn_index_map(pool: PL.DraftPool) -> dict[str, int]:
    """espn_id -> pool index, for both players and D/ST.

    Built once at startup. Players go through the §0.2 crosswalk
    (espn_id -> gsis_id -> the pool's canonical_id); D/ST are team entities and
    take the separate negative-id path M1 built for them.
    """
    from ff_agent.data import crosswalk as cw

    canon = pool.df["canonical_id"].to_list()
    by_canon = {c: i for i, c in enumerate(canon)}
    out: dict[str, int] = {}

    players = cw.canonical_players()
    have = {"espn_id", "gsis_id"} <= set(players.columns)
    if have:
        sub = players.filter(
            pl.col("espn_id").is_not_null() & pl.col("gsis_id").is_not_null()
        )
        for espn_id, gsis in zip(sub["espn_id"].to_list(), sub["gsis_id"].to_list()):
            i = by_canon.get(str(gsis))
            if i is not None:
                out[str(espn_id)] = i

    # D/ST are team entities, not players (§0.2's corollary), and the draft
    # feed stores them as ESPN's negative -16000-proTeamId ids. dst_crosswalk
    # already carries the same canonical_id the pool uses, so this is a direct
    # join rather than a second guess at the team spelling.
    xw = cw.dst_crosswalk()
    if {"espn_id", "canonical_id"} <= set(xw.columns):
        for espn_id, canon in zip(xw["espn_id"].to_list(), xw["canonical_id"].to_list()):
            i = by_canon.get(str(canon))
            if i is not None:
                out[str(espn_id)] = i
    return out


def poll_picks(season: int, index_map: dict[str, int], pool: PL.DraftPool) -> PollResult:
    """Ask ESPN what has been drafted. Never raises — failure is a message.

    Returns pool indices in ESPN's own pick order. The caller decides what to do
    with them; this deliberately does not touch DraftState, because a feed that
    disagrees with what a human typed must be surfaced, not silently applied.
    """
    try:
        from ff_agent.data.espn import get_league

        lg = get_league(season)
        picks = list(getattr(lg, "draft", None) or [])
    except Exception as exc:                      # offline, 401, ESPN 500, ...
        return PollResult([], 0, [], error=f"{type(exc).__name__}: {exc}")

    indices: list[int] = []
    unresolved: list[str] = []
    for pk in picks:
        pid = str(getattr(pk, "playerId", "") or "")
        i = index_map.get(pid)
        if i is None:
            unresolved.append(getattr(pk, "playerName", None) or f"espn_id {pid}")
            continue
        indices.append(i)
    return PollResult(indices, len(picks), unresolved)
