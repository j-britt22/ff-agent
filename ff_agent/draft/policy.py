"""How *I* pick, inside the simulation.

§5 says "you pick per the §7.2 rule", but §7.2 is a projection rule, not a draft
policy — it says how to value a player, not when to take him. So the policy has
to be written down explicitly, and once it is written down it can be *compared*,
which is the real reason to own a draft simulator. Four of them:

* ``best_vor`` — highest VOR that is legal. The baseline everything must beat.
* ``need_weighted`` — VOR plus a bonus for unfilled starting slots.
* ``tier_aware`` — a two-turn lookahead. Take the position where waiting costs
  the most, which is §7.4's tier cliff expressed as points rather than a label.
* ``qb_by(pick)`` — force the second QB by a given pick. A family, so §3.1's
  "once two or three managers grab their second QB the rest stampede" gets an
  answer instead of an argument.

Every policy shares ``legal_mask``, which enforces the roster rules and §3.5's
"never before the last two rounds" for K and D/ST. §10's sanity alarms are then
re-checked against the simulated drafts — if a policy takes a kicker in round 8,
that is a bug here, not a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ff_agent.config import POSITION_MAXIMA
from ff_agent.draft import pool as PL
from ff_agent.draft.engine import N_POS, PolicyState, _phase_index
from ff_agent.opponents import tendencies as TD

REQUIRED_STARTERS = {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "DST": 1, "K": 1}
"""§1's ten starting slots minus the FLEX, which is handled separately."""

FLEX_POOL = ("RB", "WR", "TE")
FLEX_MIN_TOTAL = 6
"""RB2 + WR2 + TE1 + one FLEX from the same three."""

KDST_LAST_ROUNDS = 2
"""§3.5: 'Still never before the last two rounds.'"""

NEED_BONUS = 15.0
"""Points added per unfilled starting slot in ``need_weighted``. Roughly a tier."""

_MAXIMA = np.array(
    [POSITION_MAXIMA.get(p, 99) for p in PL.POSITIONS], dtype=np.int16
)
_REQUIRED = np.array(
    [REQUIRED_STARTERS.get(p, 0) for p in PL.POSITIONS], dtype=np.int16
)
_FLEX_IDX = np.array([PL.POS_INDEX[p] for p in FLEX_POOL])
_K = PL.POS_INDEX["K"]
_DST = PL.POS_INDEX["DST"]
_QB = PL.POS_INDEX["QB"]
_SKILL_IDX = np.array([PL.POS_INDEX[p] for p in ("QB", "RB", "WR", "TE")])


def _pos_mask(state: PolicyState, allowed: np.ndarray) -> np.ndarray:
    """(n_sims, n_players) True where the player's position is allowed."""
    return allowed[:, state.pool.pos_idx]


def allowed_positions(
    counts: np.ndarray,
    round_no: int,
    rounds: int,
    kdst_last_rounds: int | None = KDST_LAST_ROUNDS,
    maxima: np.ndarray | None = None,
) -> np.ndarray:
    """(n_sims, N_POS) — which positions a team may legally take right now.

    Two rules, and they apply to **every** team, not just mine:

    * ESPN's ``POSITION_MAXIMA``. Nobody drafts a fifth quarterback.
    * Enough picks left to still field §1's ten starters. When the remaining
      picks exactly equal the remaining requirements, the choice collapses.

    That second rule is why this function exists at all. M5's pick model was
    fitted to score picks that had *already happened*, so it never needed to know
    a roster was illegal — and run forward without these constraints it happily
    drafted eight quarterbacks and no tight end, then fielded a lineup with empty
    slots. The gap that opened against a constrained policy was ~50 points a
    week, which is how the omission announced itself.

    ``kdst_last_rounds=None`` lifts the §3.5 restriction on early K/D-ST, which
    is right for opponents: *their* timing comes from their fitted consensus
    position, not from my discipline.
    """
    caps = _MAXIMA if maxima is None else maxima
    allowed = counts < caps[None, :]
    picks_left = rounds - round_no + 1

    if kdst_last_rounds is not None:
        first_kdst_round = rounds - kdst_last_rounds + 1
        if round_no < first_kdst_round:
            allowed = allowed.copy()
            allowed[:, _K] = False
            allowed[:, _DST] = False

    deficit = np.maximum(_REQUIRED[None, :] - counts, 0)
    after_min = counts[:, _FLEX_IDX] + deficit[:, _FLEX_IDX]
    extra_flex = np.maximum(FLEX_MIN_TOTAL - after_min.sum(axis=1), 0)
    hard = deficit.sum(axis=1)
    need = hard + extra_flex

    forced = need >= picks_left
    if forced.any():
        want = deficit > 0
        # The FLEX is the lowest-priority need: any RB/WR/TE fills it, so it is
        # always satisfiable later. Adding it to `want` alongside a hard
        # positional deficit lets a fourth wide receiver through while the roster
        # still has no kicker. Only offer it once the hard deficits fit.
        loose = hard < picks_left
        want[:, _FLEX_IDX] |= ((extra_flex > 0) & loose)[:, None]
        allowed = np.where(forced[:, None], allowed & want, allowed)
    return allowed


def caps_array(caps: dict[str, int] | None) -> np.ndarray | None:
    """Tighter-than-ESPN roster caps, as a positional array."""
    if not caps:
        return None
    out = _MAXIMA.copy()
    for pos, n in caps.items():
        out[PL.POS_INDEX[pos]] = n
    return out


def legal_mask(state: PolicyState, caps: dict[str, int] | None = None) -> np.ndarray:
    """Available, under the maxima, and consistent with fielding a legal lineup."""
    allowed = allowed_positions(
        state.counts, state.round_no, state.rounds, maxima=caps_array(caps)
    )
    legal = state.available & _pos_mask(state, allowed)
    # never leave a sim with nothing legal — fall back to plain availability
    dead = ~legal.any(axis=1)
    if dead.any():
        legal[dead] = state.available[dead]
    return legal


def _nth_best(state: PolicyState, legal: np.ndarray, p: int, j: int):
    """VOR and pool index of the j-th best legal player at position ``p``.

    ``j`` is 1-based. Returns ``(-inf, -1)`` where fewer than j are legal. This
    is one cumulative sum only because the pool is sorted by VOR.
    """
    mem = state.members[p]
    if mem.size == 0 or j < 1:
        n = state.n_sims
        return np.full(n, -np.inf, dtype=np.float32), np.full(n, -1, dtype=np.int32)
    sub = legal[:, mem]
    hit = sub & (np.cumsum(sub, axis=1) == j)
    any_hit = hit.any(axis=1)
    col = hit.argmax(axis=1)
    idx = np.where(any_hit, mem[col], -1).astype(np.int32)
    val = np.where(any_hit, state.pool.vor[np.maximum(idx, 0)], -np.inf).astype(np.float32)
    return val, idx


def _argmax_legal(state: PolicyState, score: np.ndarray, legal: np.ndarray) -> np.ndarray:
    return np.where(legal, score, -np.inf).argmax(axis=1).astype(np.int32)


@dataclass
class BestVOR:
    """Highest legal VOR. Deterministic given availability.

    ``caps`` tightens the roster limits below ESPN's. It exists because of what
    this policy does unconstrained: it takes **four quarterbacks in the first
    five rounds**, every time. That is not a bug in the policy — the board really
    does rank Allen, Jackson, Maye and Burrow at VOR 167/148/147/122, ahead of
    everything else. It is a flaw in using VOR for a bench pick. VOR is measured
    against the STARTER replacement level (QB18), so it prices a third and fourth
    quarterback as though they would start, and in a league that starts two they
    will not. §3.6 wants 2-3; §3.2 says do not hoard.
    """

    caps: dict[str, int] | None = None
    name: str = "best_vor"

    def choose(self, state: PolicyState) -> np.ndarray:
        legal = legal_mask(state, self.caps)
        score = np.broadcast_to(state.pool.vor, (state.n_sims, state.pool.n))
        return _argmax_legal(state, score, legal)


@dataclass
class NeedWeighted:
    """VOR plus ``bonus`` per unfilled starting slot at that position."""

    bonus: float = NEED_BONUS
    caps: dict[str, int] | None = None
    name: str = "need_weighted"

    def choose(self, state: PolicyState) -> np.ndarray:
        legal = legal_mask(state, self.caps)
        unfilled = np.maximum(_REQUIRED[None, :] - state.counts, 0).astype(np.float32)
        flex_short = np.maximum(
            FLEX_MIN_TOTAL - state.counts[:, _FLEX_IDX].sum(axis=1), 0
        ).astype(np.float32)
        unfilled[:, _FLEX_IDX] += flex_short[:, None] / len(_FLEX_IDX)
        score = state.pool.vor[None, :] + self.bonus * unfilled[:, state.pool.pos_idx]
        return _argmax_legal(state, score, legal)


@dataclass
class TierAware:
    """Two-turn lookahead: what does waiting actually cost at each position?

    For every position, value taking its best player now and then receiving, at
    my next turn, the best of what survives everywhere else. The position whose
    total is highest wins. A tier cliff shows up as a large drop between the best
    available and the fallback — §7.4's point, priced in points rather than
    asserted by a tier label.

    Degenerates to ``best_vor`` on the final pick, where there is no next turn.
    """

    caps: dict[str, int] | None = None
    name: str = "tier_aware"

    def choose(self, state: PolicyState) -> np.ndarray:
        legal = legal_mask(state, self.caps)
        gap = state.picks_until_next_turn()
        if gap <= 0:
            score = np.broadcast_to(state.pool.vor, (state.n_sims, state.pool.n))
            return _argmax_legal(state, score, legal)

        pf = state.pick_no / (state.rounds * state.n_teams)
        rates = _league_rates()[_phase_index(pf)]

        best = np.empty((state.n_sims, N_POS), dtype=np.float32)
        best_idx = np.empty((state.n_sims, N_POS), dtype=np.int32)
        fb = np.empty((state.n_sims, N_POS), dtype=np.float32)
        fb_after = np.empty((state.n_sims, N_POS), dtype=np.float32)
        for p in range(N_POS):
            expected_gone = int(round(gap * float(rates[p])))
            best[:, p], best_idx[:, p] = _nth_best(state, legal, p, 1)
            fb[:, p], _ = _nth_best(state, legal, p, 1 + expected_gone)
            fb_after[:, p], _ = _nth_best(state, legal, p, 2 + expected_gone)

        total = np.empty((state.n_sims, N_POS), dtype=np.float32)
        for p in range(N_POS):
            alt = fb.copy()
            alt[:, p] = fb_after[:, p]           # I just took one from p
            total[:, p] = best[:, p] + np.nanmax(
                np.where(np.isfinite(alt), alt, -np.inf), axis=1
            )
        total = np.where(np.isfinite(best), total, -np.inf)
        pick_pos = total.argmax(axis=1)
        chosen = best_idx[np.arange(state.n_sims), pick_pos]

        bad = chosen < 0
        if bad.any():
            fallback = _argmax_legal(
                state, np.broadcast_to(state.pool.vor, (state.n_sims, state.pool.n)), legal
            )
            chosen = np.where(bad, fallback, chosen)
        return chosen.astype(np.int32)


@dataclass
class BestConsensus:
    """Take the best available by CONSENSUS rank, with no noise at all.

    Not a policy anyone should draft — a control. It separates two things that
    otherwise arrive fused in ``best_vor``'s margin over the opponent model:

    * **Board edge** — how much my blended projections disagree with superflex
      ECR, and profitably. That is real, and M3 measured it as small.
    * **Noise exploitation** — how much I gain purely because M5's opponents
      pick with an 8-pick standard deviation while I do not. That is an artifact
      of a weak opponent model, and it belongs in the caveats, not the plan.

    ``best_consensus`` has zero board edge by construction, so whatever margin it
    still shows over the model is entirely the second kind.
    """

    name: str = "best_consensus"

    def choose(self, state: PolicyState) -> np.ndarray:
        legal = legal_mask(state)
        cf = state.pool.consensus_fraction
        score = np.broadcast_to(-cf.astype(np.float32), (state.n_sims, state.pool.n))
        return _argmax_legal(state, score, legal)


@dataclass
class ModelSeat:
    """I draft exactly like the opponent model does. The zero point.

    Any measured edge has to be read against this, not against nothing.
    """

    name: str = "model_seat"
    tilt: dict | None = None
    reach: dict | None = None
    manager: str = ""
    pos_offsets: np.ndarray | None = None
    adp_sigma: float | None = None

    def choose(self, state: PolicyState) -> np.ndarray:
        from ff_agent.draft.engine import opponent_utility, tilt_array
        from ff_agent.opponents import pickmodel as PM

        legal = legal_mask(state)
        total = state.rounds * state.n_teams
        pf = state.pick_no / total
        ph = _phase_index(pf)
        row = tilt_array(self.tilt or {}, [self.manager])[0, ph]
        if self.pos_offsets is not None:
            row = row + self.pos_offsets[ph]
        base = opponent_utility(
            state.pool.consensus_fraction, state.pool.pos_idx, pf,
            (self.reach or {}).get(self.manager, 0.0), row,
            PM.ADP_SIGMA if self.adp_sigma is None else self.adp_sigma,
        )
        util = np.broadcast_to(base, (state.n_sims, state.pool.n)) + state.rng.gumbel(
            size=(state.n_sims, state.pool.n)
        ).astype(np.float32)
        return _argmax_legal(state, util, legal)


@dataclass
class QBBy:
    """``best_vor``, except the second QB is taken no later than ``pick``.

    §3.1 names the QB run as this league's specific structural risk. Sweeping
    ``pick`` turns that risk into a measured curve.
    """

    pick: int = 40
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"qb_by_{self.pick}"

    def choose(self, state: PolicyState) -> np.ndarray:
        legal = legal_mask(state)
        score = np.array(
            np.broadcast_to(state.pool.vor, (state.n_sims, state.pool.n)), dtype=np.float32
        )
        if state.pick_no >= self.pick:
            short = state.counts[:, _QB] < REQUIRED_STARTERS["QB"]
            if short.any():
                qb_only = legal & (state.pool.pos_idx == _QB)[None, :]
                usable = short & qb_only.any(axis=1)
                legal = np.where(usable[:, None], qb_only, legal)
        return _argmax_legal(state, score, legal)


_RATES: np.ndarray | None = None


def _league_rates() -> np.ndarray:
    """(n_phases, n_positions) league P(position | phase), from M5's own fit.

    K and D/ST are absent from ``positional_tendency`` (it filters to skill), so
    their share comes from the same measured history ``pool.py`` uses. Rows are
    renormalised so each phase sums to one.
    """
    global _RATES
    if _RATES is not None:
        return _RATES

    from ff_agent.opponents import history as H
    import polars as pl

    h = H.draft_history().with_columns(
        pl.col("pick_fraction").alias("_pf")
    )
    out = np.zeros((len(TD.PHASES), N_POS), dtype=np.float32)
    for pi, (_, lo, hi) in enumerate(TD.PHASES):
        seg = h.filter((pl.col("_pf") >= lo) & (pl.col("_pf") < hi))
        n = max(seg.height, 1)
        for qi, pos in enumerate(PL.POSITIONS):
            out[pi, qi] = seg.filter(pl.col("position") == pos).height / n
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-9)
    _RATES = out
    return out


ALL_POLICIES = (BestVOR(), NeedWeighted(), TierAware())
