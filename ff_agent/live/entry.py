"""Typing a pick at the table.

§6 and §0.1 make manual entry the DESIGN, not a fallback: "ESPN's live draft
feed is flakier than Sleeper's — build and rehearse manual entry first, treat
polling as a bonus."

**This is not the fuzzy matching §0.2 forbids, and the distinction matters.**
§0.2 bans silently joining datasets on names, because a wrong join produces a
board that recommends a retired player and nobody finds out. Here a human types
a fragment, *sees the candidates*, and confirms one. No canonical id is ever
resolved without a person looking at it. So the rule this module enforces is:

    ambiguity always asks, never guesses.

A single exact match on a full name is taken directly. Anything else — several
prefix hits, a misspelling, nothing at all — comes back as a question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ff_agent.data.crosswalk import (
    CURRENT_TEAMS, normalize_team, team_from_nickname,
)
from ff_agent.draft import pool as PL

MAX_SUGGESTIONS = 8

SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
"""Generational suffixes, which the sources disagree about.

The 2025 replay found this the hard way: ESPN's draft board says "James Cook
III" and the projection board says "James Cook", and a query LONGER than the
stored name fails prefix, surname and substring matching alike — so typing the
name exactly as ESPN shows it matched nothing at all.

Stripping an enumerable set of suffix tokens from both sides is not the fuzzy
matching §0.2 forbids. It is a fixed, listed transformation, ambiguity still
prompts, and no id resolves without a human confirming it."""


@dataclass
class Match:
    """Either one resolved player, or a question for the human."""

    index: int | None
    candidates: list[int]
    query: str
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.index is not None

    @property
    def ambiguous(self) -> bool:
        return self.index is None and bool(self.candidates)

    @property
    def missing(self) -> bool:
        return self.index is None and not self.candidates


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _strip_suffix(s: str) -> str:
    parts = [w for w in _norm(s).split() if w not in SUFFIXES]
    return " ".join(parts)


class Resolver:
    """Name fragment -> pool index, with the human in the loop."""

    def __init__(self, pool: PL.DraftPool):
        self.pool = pool
        names = pool.df["name"].to_list()
        self.names = names
        self.norm = [_strip_suffix(n) for n in names]
        self.positions = [PL.POSITIONS[p] for p in pool.pos_idx]
        self.teams = [t or "" for t in pool.df["team"].to_list()]
        self.last = [n.split()[-1] if n else "" for n in self.norm]

    def resolve(
        self, query: str, available: set[int] | None = None
    ) -> Match:
        q = _strip_suffix(query)
        if not q:
            return Match(None, [], query, "empty entry")

        pool_ids = range(self.pool.n)
        if available is not None:
            pool_ids = [i for i in pool_ids if i in available]

        # a team abbreviation or nickname means that team's defense, however
        # the two sources happen to spell it ("Texans D/ST" vs "HOU D/ST")
        # normalize_team() echoes anything it does not recognise, so it is
        # always truthy — check it names a real team before trusting it, or the
        # nickname path is short-circuited and never runs.
        abbr = normalize_team(query.strip())
        team = abbr if abbr in CURRENT_TEAMS else team_from_nickname(query)
        if team:
            dst = [i for i in pool_ids
                   if self.positions[i] == "DST" and normalize_team(self.teams[i]) == team]
            if len(dst) == 1:
                return Match(dst[0], [], query, "team -> D/ST")

        exact = [i for i in pool_ids if self.norm[i] == q]
        if len(exact) == 1:
            return Match(exact[0], [], query, "exact name")
        if len(exact) > 1:
            return Match(None, exact, query, "several players share that exact name")

        prefix = [i for i in pool_ids if self.norm[i].startswith(q)]
        surname = [i for i in pool_ids if self.last[i] == q]
        contains = [i for i in pool_ids if q in self.norm[i]]

        for hits, why in ((prefix, "name prefix"), (surname, "surname"),
                          (contains, "name contains")):
            if len(hits) == 1:
                return Match(hits[0], [], query, why)
            if hits:
                return Match(None, hits[:MAX_SUGGESTIONS], query,
                             f"{len(hits)} players match on {why}")

        return Match(None, [], query, "no player matches — check the spelling")

    def describe(self, i: int) -> str:
        return f"{self.names[i]} ({self.positions[i]}, {self.teams[i] or 'FA'})"
