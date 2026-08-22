"""Draft state as it actually unfolds.

Deliberately dumb and fully testable: a list of pool indices in pick order, and
everything else derived. The expensive thinking lives in ``advise.py``; this is
the thing that has to be *right* while somebody is reading a clock.

§0.1 governs this whole package. Nothing here writes to ESPN, and nothing here
can: the only inputs are the pool, the human's typing, and (optionally, and
read-only) a poll of the draft feed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ff_agent.config import N_TEAMS, POSITION_MAXIMA, ROSTER_TOTAL
from ff_agent.draft import engine as E
from ff_agent.draft import pool as PL


@dataclass
class DraftState:
    pool: PL.DraftPool
    managers: list[str]
    my_index: int
    rounds: int = ROSTER_TOTAL
    history: list[int] = field(default_factory=list)

    @property
    def n_teams(self) -> int:
        return len(self.managers)

    @property
    def total_picks(self) -> int:
        return self.n_teams * self.rounds

    @property
    def order(self) -> np.ndarray:
        return E.snake_order(self.n_teams, self.rounds)

    @property
    def next_pick(self) -> int:
        """1-based overall pick about to happen. ``total_picks + 1`` when done."""
        return len(self.history) + 1

    @property
    def complete(self) -> bool:
        return len(self.history) >= self.total_picks

    @property
    def on_the_clock(self) -> int:
        """Team index whose pick is next; -1 when the draft is over."""
        return -1 if self.complete else int(self.order[len(self.history)])

    @property
    def my_turn(self) -> bool:
        return self.on_the_clock == self.my_index

    def my_turns(self) -> np.ndarray:
        """1-based pick numbers that belong to me."""
        return np.where(self.order == self.my_index)[0] + 1

    def picks_until_my_turn(self) -> int | None:
        """0 if I am on the clock now; None if I have no picks left."""
        later = self.my_turns()[self.my_turns() >= self.next_pick]
        return int(later[0] - self.next_pick) if later.size else None

    def taken(self) -> set[int]:
        return set(self.history)

    def available_mask(self) -> np.ndarray:
        m = np.ones(self.pool.n, dtype=bool)
        if self.history:
            m[np.array(self.history)] = False
        return m

    def roster_of(self, team: int) -> list[int]:
        o = self.order
        return [p for k, p in enumerate(self.history) if o[k] == team]

    def my_roster(self) -> list[int]:
        return self.roster_of(self.my_index)

    def position_counts(self, team: int | None = None) -> dict[str, int]:
        idx = self.my_roster() if team is None else self.roster_of(team)
        out = {p: 0 for p in PL.POSITIONS}
        for i in idx:
            out[PL.POSITIONS[self.pool.pos_idx[i]]] += 1
        return out

    def counts_array(self, team: int | None = None) -> np.ndarray:
        c = self.position_counts(team)
        return np.array([[c[p] for p in PL.POSITIONS]], dtype=np.int16)

    # ── mutation ─────────────────────────────────────────────────────────────
    def record(self, player: int) -> None:
        """Append one pick. Refuses anything that would corrupt the board."""
        if self.complete:
            raise ValueError("the draft is already complete")
        if player < 0 or player >= self.pool.n:
            raise ValueError(f"{player} is not a valid pool index")
        if player in self.taken():
            name = self.pool.df["name"].to_list()[player]
            raise ValueError(
                f"{name} has already been drafted (pick "
                f"{self.history.index(player) + 1}). If the earlier entry was "
                "wrong, undo it rather than recording him twice."
            )
        self.history.append(player)

    def undo(self) -> int:
        if not self.history:
            raise ValueError("nothing to undo")
        return self.history.pop()

    # ── §10 alarms, live ─────────────────────────────────────────────────────
    def alarms(self) -> list[str]:
        """§10's sanity alarms, about MY roster, checked as the draft happens."""
        c = self.position_counts()
        rnd = (self.next_pick - 1) // self.n_teams + 1
        out = []
        if c["K"] and rnd < 13:
            out.append(f"a kicker is on my roster in round {rnd} (§10: not before 13)")
        if rnd > 11 and c["QB"] <= 1:
            out.append(f"only {c['QB']} QB after round 11 (§10) — this is a 2-QB league")
        for pos, cap in POSITION_MAXIMA.items():
            if c.get(pos, 0) > cap:
                out.append(f"{c[pos]} {pos} exceeds ESPN's maximum of {cap}")
        return out

    def summary(self) -> str:
        c = self.position_counts()
        shape = " ".join(f"{p}{c[p]}" for p in PL.POSITIONS if c[p])
        return (
            f"pick {min(self.next_pick, self.total_picks)}/{self.total_picks} · "
            f"round {(self.next_pick - 1) // self.n_teams + 1} · "
            f"my roster {len(self.my_roster())}/{self.rounds} [{shape or 'empty'}]"
        )


def new_state(
    pool: PL.DraftPool, slot: int, managers: list[str], rounds: int = ROSTER_TOTAL
) -> DraftState:
    if not 1 <= slot <= len(managers):
        raise ValueError(f"slot {slot} is outside 1..{len(managers)}")
    return DraftState(pool=pool, managers=managers, my_index=slot - 1, rounds=rounds)
