"""Vectorized Monte Carlo draft (§5, §11 step 7).

**The model here is M5's, unchanged.** ``opponents/pickmodel.py`` fits three
terms — ADP proximity, per-manager positional tilt, run pressure — and this
module reproduces them exactly. Only the execution differs, and it has to:
M5 walks one 136-pick draft with a polars filter per pick, which is fine for
scoring a single draft and hopeless at 10,000 sims x 9 slots x 153 picks = 13.8
million picks.

So every draft advances in lockstep. Availability is an ``(n_sims, n_players)``
boolean mask, and sampling from the softmax is the Gumbel-max trick — argmax of
``utility + Gumbel(0,1)`` is an exact draw from ``softmax(utility)``, with no
per-row normalisation. Measured: ~11s per slot at 10,000 sims.

``test_draft.py`` asserts this path reproduces ``pick_probabilities`` to float
tolerance on fixed input, so the fast path cannot silently drift away from the
fitted one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ff_agent.config import N_TEAMS, ROSTER_TOTAL
from ff_agent.draft import pool as PL
from ff_agent.opponents import pickmodel as PM
from ff_agent.opponents import tendencies as TD

N_POS = len(PL.POSITIONS)
PHASE_NAMES = tuple(p[0] for p in TD.PHASES)


def snake_order(n_teams: int = N_TEAMS, rounds: int = ROSTER_TOTAL) -> np.ndarray:
    """Team index (0-based) on the clock at each overall pick.

    §5's gap table falls out of this: slot 1 waits 17 picks then picks twice in
    a row, slot 5 in a 9-team draft waits exactly 9 every time.
    """
    order = np.empty(n_teams * rounds, dtype=np.int16)
    for r in range(rounds):
        row = np.arange(n_teams) if r % 2 == 0 else np.arange(n_teams)[::-1]
        order[r * n_teams:(r + 1) * n_teams] = row
    return order


def turns_for_slot(slot: int, n_teams: int = N_TEAMS, rounds: int = ROSTER_TOTAL) -> np.ndarray:
    """1-based overall pick numbers belonging to ``slot`` (1-based)."""
    order = snake_order(n_teams, rounds)
    return np.where(order == slot - 1)[0] + 1


def _phase_index(pick_fraction: float) -> int:
    for i, (_, lo, hi) in enumerate(TD.PHASES):
        if lo <= pick_fraction < hi:
            return i
    return len(TD.PHASES) - 1


def tilt_array(tilt_lookup: dict, managers: list[str]) -> np.ndarray:
    """(n_managers, n_phases, n_positions) of log(manager rate / league rate).

    Same quantity ``pick_probabilities`` computes per player; hoisted out of the
    inner loop because it only depends on (manager, phase, position).
    """
    out = np.zeros((len(managers), len(TD.PHASES), N_POS), dtype=np.float32)
    for mi, m in enumerate(managers):
        for pi, (ph, _, _) in enumerate(TD.PHASES):
            for qi, pos in enumerate(PL.POSITIONS):
                rate, lg = tilt_lookup.get((m, ph, pos), (None, None))
                if rate and lg and lg > 0:
                    out[mi, pi, qi] = np.log(max(rate, 1e-6) / lg)
    return out


def opponent_utility(
    cf: np.ndarray,
    pos_idx: np.ndarray,
    pick_fraction: float,
    reach: float,
    tilt_row: np.ndarray,
    adp_sigma: float = PM.ADP_SIGMA,
) -> np.ndarray:
    """The manager-independent part of one pick's utility, over the whole pool.

    Deliberately a free function: the parity test calls exactly this and compares
    against ``pickmodel.pick_probabilities``.
    """
    target = pick_fraction + reach
    return (-np.abs(cf - target) / adp_sigma + tilt_row[pos_idx]).astype(np.float32)


@dataclass
class DraftResult:
    picks: np.ndarray            # (n_sims, total_picks) pool index
    team_at_pick: np.ndarray     # (total_picks,) team index
    managers: list[str]
    my_index: int
    pool: PL.DraftPool
    n_teams: int
    rounds: int

    @property
    def n_sims(self) -> int:
        return self.picks.shape[0]

    def roster_of(self, team: int) -> np.ndarray:
        """(n_sims, rounds) pool indices drafted by ``team``."""
        cols = np.where(self.team_at_pick == team)[0]
        return self.picks[:, cols]

    def my_roster(self) -> np.ndarray:
        return self.roster_of(self.my_index)

    def pick_number_of(self, player: int) -> np.ndarray:
        """(n_sims,) 1-based overall pick where ``player`` went; 0 = undrafted."""
        hit = self.picks == player
        any_hit = hit.any(axis=1)
        return np.where(any_hit, hit.argmax(axis=1) + 1, 0)


def simulate_drafts(
    pool: PL.DraftPool,
    managers: list[str],
    my_index: int,
    policy,
    tilt_lookup: dict,
    reach: dict[str, float],
    n_sims: int = 10_000,
    rounds: int = ROSTER_TOTAL,
    seed: int = 11,
    adp_sigma: float = PM.ADP_SIGMA,
    pos_offsets: np.ndarray | None = None,
) -> DraftResult:
    """Run ``n_sims`` drafts. ``managers[my_index]`` is me and uses ``policy``.

    ``pos_offsets`` is an (n_phases, n_positions) log-offset on the league's
    aggregate positional mix — see ``draft/calibrate.py``. M5's manager tilts are
    measured *against the league rate*, so they sum to roughly zero and cannot
    move the aggregate at all: without this term the simulated mix is whatever
    superflex ECR implies, which the M7 gate showed is not what this league does.
    """
    n_teams = len(managers)
    total = n_teams * rounds
    if total > pool.n:
        raise ValueError(
            f"pool has {pool.n} players but the draft needs {total} picks. "
            "A pool that runs dry silently produces short rosters."
        )

    from ff_agent.draft import policy as POLICY   # circular at module scope

    order = snake_order(n_teams, rounds)
    tilt = tilt_array(tilt_lookup, managers)
    reach_v = np.array([reach.get(m, 0.0) for m in managers], dtype=np.float32)

    rng = np.random.default_rng(seed)
    avail = np.ones((n_sims, pool.n), dtype=bool)
    picks = np.zeros((n_sims, total), dtype=np.int32)
    rows = np.arange(n_sims)

    # circular buffer of the last RUN_WINDOW pick positions, per sim
    recent = np.zeros((n_sims, PM.RUN_WINDOW), dtype=np.int8)
    n_recent = 0

    counts = np.zeros((n_sims, n_teams, N_POS), dtype=np.int16)
    my_turns = np.where(order == my_index)[0]
    if my_turns.size and policy is None:
        raise ValueError("a seat was assigned to me but no policy was given")
    my_counts = counts[:, my_index] if my_turns.size else None
    ctx = PolicyState(pool=pool, rounds=rounds, n_teams=n_teams, my_turns=my_turns)

    for k in range(total):
        team = int(order[k])
        pf = (k + 1) / total

        if team == my_index:
            ctx.pick_no = k + 1
            ctx.round_no = k // n_teams + 1
            ctx.available = avail
            ctx.counts = my_counts
            ctx.rng = rng
            choice = policy.choose(ctx)
        else:
            ph = _phase_index(pf)
            row = tilt[team, ph]
            if pos_offsets is not None:
                row = row + pos_offsets[ph]
            base = opponent_utility(
                pool.consensus_fraction, pool.pos_idx, pf,
                float(reach_v[team]), row, adp_sigma,
            )
            util = np.broadcast_to(base, (n_sims, pool.n)) + rng.gumbel(
                size=(n_sims, pool.n)
            ).astype(np.float32)
            if n_recent:
                util = util + PM.RUN_BOOST * _run_boost(recent, n_recent, pool.pos_idx)
            # M5's model knows nothing about roster legality — see
            # policy.allowed_positions for why that has to be imposed here
            allowed = POLICY.allowed_positions(
                counts[:, team], k // n_teams + 1, rounds, kdst_last_rounds=None
            )
            legal = avail & allowed[:, pool.pos_idx]
            dead = ~legal.any(axis=1)
            if dead.any():
                legal[dead] = avail[dead]
            util = np.where(legal, util, -np.inf)
            choice = util.argmax(axis=1).astype(np.int32)

        picks[:, k] = choice
        avail[rows, choice] = False
        cpos = pool.pos_idx[choice]
        counts[rows, team, cpos] += 1
        recent[:, k % PM.RUN_WINDOW] = cpos
        n_recent = min(n_recent + 1, PM.RUN_WINDOW)

    return DraftResult(picks, order, managers, my_index, pool, n_teams, rounds)


def _run_boost(recent: np.ndarray, n_recent: int, pos_idx: np.ndarray) -> np.ndarray:
    """(n_sims, n_players) 1.0 where that player's position is mid-run.

    §3.1's named residual risk: once two or three managers grab their second QB,
    the rest stampede. Picks are not independent and the model says so.
    """
    n_sims = recent.shape[0]
    counts = np.zeros((n_sims, N_POS), dtype=np.float32)
    rows = np.arange(n_sims)
    for j in range(n_recent):
        np.add.at(counts, (rows, recent[:, j]), 1.0)
    return (counts / n_recent >= 0.4)[:, pos_idx].astype(np.float32)


@dataclass
class PolicyState:
    """Everything a policy sees. Mutated in place each turn — never stored."""

    pool: PL.DraftPool
    rounds: int
    n_teams: int
    my_turns: np.ndarray             # 0-based overall pick indices that are mine
    pick_no: int = 0                 # 1-based
    round_no: int = 0                # 1-based
    available: np.ndarray | None = None   # (n_sims, n_players)
    counts: np.ndarray | None = None      # (n_sims, N_POS) my roster so far
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        # pool indices per position, already in VOR order — the whole reason
        # "the j-th best available at position p" is one cumulative sum
        self.members = [
            np.where(self.pool.pos_idx == p)[0] for p in range(N_POS)
        ]

    @property
    def n_sims(self) -> int:
        return self.available.shape[0]

    def picks_until_next_turn(self) -> int:
        """0 if this is my last pick. Drives every wait-or-take decision."""
        later = self.my_turns[self.my_turns > self.pick_no - 1]
        return int(later[0] - (self.pick_no - 1)) if later.size else 0
