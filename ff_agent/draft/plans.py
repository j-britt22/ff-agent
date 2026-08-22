"""Per-turn plans (§5) — what is realistically there when I am on the clock.

§5 lists exactly what each ``plan_{slot}.json`` must carry: my exact pick
numbers, expected best-available tier per position at each turn *with
probabilities*, recommended positional shape by round, branch points, and the
distribution of final roster strength. All of it falls out of an M7
``DraftResult`` almost for free — availability for all seventeen turns costs
0.03 seconds, because it is one scatter into a "when was this player taken"
matrix and then comparisons against it.

The number that actually decides a pick is **conditional survival**:

    P(still available at my NEXT turn | available now)

Not raw availability. "Lamar Jackson is 93% available at pick 14" is a fact
about the draft; "if I pass on him now he comes back 61% of the time" is a fact
about my decision, and it is the one §5 means when it says survival probability
dominates at the wheel slots. Slot 1 waits 17 picks and slot 5 waits 9, so the
same player has a very different survival number in the two plans — which is
also why the nine plans are not one list re-indexed.

Everything here is expressed in availability and positional shape rather than
title probability. That is deliberate: M7's gate calibrated the first two and
showed the third is inflated in absolute terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ff_agent.draft import engine as E
from ff_agent.draft import pool as PL

SHORTLIST_SIZE = 10
AVAILABLE_THRESHOLD = 0.35
"""Below this, a player is not a realistic option at that turn and listing him
is padding. §5 wants a plan, not a ranked board reprinted seventeen times."""

TAKE_NOW_SURVIVAL = 0.5
"""If a player available now returns less than half the time, passing on him is
a decision rather than a deferral."""


def gone_matrix(res: E.DraftResult) -> np.ndarray:
    """(n_sims, n_players) 1-based pick at which each player went.

    Undrafted players get ``total_picks + 1``, so "still available at pick t" is
    a single comparison and undrafted never reads as "taken at pick 0".
    """
    total = res.picks.shape[1]
    out = np.full((res.n_sims, res.pool.n), total + 1, dtype=np.int32)
    rows = np.arange(res.n_sims)[:, None]
    out[rows, res.picks] = np.arange(1, total + 1)[None, :]
    return out


def availability(gone: np.ndarray, turn: int) -> np.ndarray:
    """(n_players,) P(available when pick ``turn`` is on the clock)."""
    return (gone >= turn).mean(axis=0)


def conditional_survival(
    gone: np.ndarray, turn: int, next_turn: int, my_pick: np.ndarray | None = None
) -> np.ndarray:
    """(n_players,) P(available at ``next_turn`` | available now **and I passed**).

    The counterfactual matters. Without ``my_pick``, every player the policy
    actually takes at this turn scores a survival of zero — which is true, and
    useless: he is gone because *I* took him. The pass-or-take question is
    whether he comes back if I DON'T, so sims where I took him are excluded
    rather than counted as evidence that he vanishes.

    Still not a perfect counterfactual — in the surviving sims I spent the pick
    on somebody else, which changes nothing about this player but does change
    the board. Close enough to be the right number, and stated rather than
    assumed.
    """
    here = gone >= turn
    if my_pick is not None:
        took = my_pick[:, None] == np.arange(gone.shape[1])[None, :]
        here = here & ~took
    n_here = here.sum(axis=0)
    both = (here & (gone >= next_turn)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_here > 0, both / np.maximum(n_here, 1), np.nan)


def best_available_tier(
    res: E.DraftResult, gone: np.ndarray, turn: int, position: str
) -> dict:
    """§5's "expected best-available tier per position, with probabilities".

    Returns the distribution over tiers of the BEST available player at that
    position, plus his expected VOR. The pool is VOR-sorted, so "best available"
    is the first surviving member.
    """
    members = np.where(res.pool.pos_idx == PL.POS_INDEX[position])[0]
    if members.size == 0:
        return {"position": position, "p_any": 0.0, "tiers": {}}
    sub = gone[:, members] >= turn
    has = sub.any(axis=1)
    idx = members[sub.argmax(axis=1)]
    tiers = res.pool.tier[idx][has]
    vors = res.pool.vor[idx][has]
    counts = {}
    for t, c in zip(*np.unique(tiers, return_counts=True)):
        counts[int(t)] = round(float(c) / res.n_sims, 4)
    return {
        "position": position,
        "p_any": round(float(has.mean()), 4),
        "modal_tier": int(np.bincount(tiers).argmax()) if tiers.size else None,
        "expected_vor": round(float(vors.mean()), 2) if vors.size else None,
        "vor_p10_p90": (
            [round(float(np.percentile(vors, 10)), 2),
             round(float(np.percentile(vors, 90)), 2)] if vors.size else None
        ),
        "tiers": counts,
    }


CANDIDATE_FLOOR = 0.01
"""A player the policy takes less than 1% of the time is not a candidate."""


def _player_row(res, i, p, surv, taken_by_me, next_turn) -> dict:
    names = res.pool.df["name"].to_list()
    ids = res.pool.df["canonical_id"].to_list()
    teams = res.pool.df["team"].to_list()
    sv = float(surv[i]) if next_turn else float("nan")
    # NaN means the counterfactual never occurs: in every simulation where he
    # was available, the policy took him. That is the STRONGEST take-now signal
    # there is, and as a bare NaN it compares False against the threshold and
    # would print as "not urgent".
    always = next_turn is not None and not np.isfinite(sv)
    return {
        "canonical_id": ids[i],
        "name": names[i],
        "position": PL.POSITIONS[res.pool.pos_idx[i]],
        "team": teams[i],
        "vor": round(float(res.pool.vor[i]), 1),
        "tier": int(res.pool.tier[i]),
        "p_available": round(float(p[i]), 3),
        "p_i_take_him_here": round(float((taken_by_me == i).mean()), 3),
        "survival_to_next_turn": (
            None if (next_turn is None or always) else round(sv, 3)
        ),
        "never_passed_on": bool(always),
        "take_now": bool(next_turn is not None and (always or sv < TAKE_NOW_SURVIVAL)),
        "free_bye": bool(res.pool.free_bye[i]),
    }


def shortlist(
    res: E.DraftResult,
    gone: np.ndarray,
    turn: int,
    next_turn: int | None,
    size: int = SHORTLIST_SIZE,
) -> dict:
    """The two lists a draft sheet actually needs.

    Ranking by VOR alone produces a useless sheet here, and the reason is worth
    recording: the board is QB-heavy (M4, M7) and the simulated league drafts
    quarterbacks late, so at a mid draft turn the five highest-VOR available
    players are all quarterbacks the policy takes 0% of the time. A list of
    players I will not pick is padding.

    So, two lists with separate jobs:

    * ``candidates`` — who the policy actually takes here, with probability.
      This is what M9 hands over.
    * ``slipping_away`` — available now, valuable, and does not come back.
      Passing on these is a decision rather than a deferral, whether or not the
      policy currently takes them.
    """
    p = availability(gone, turn)
    taken_by_me = res.picks[:, turn - 1]
    surv = (
        conditional_survival(gone, turn, next_turn, taken_by_me)
        if next_turn else np.zeros(res.pool.n)
    )

    counts = np.bincount(taken_by_me, minlength=res.pool.n) / res.n_sims
    cand_idx = np.where(counts >= CANDIDATE_FLOOR)[0]
    cand_idx = cand_idx[np.argsort(-counts[cand_idx])][:size]
    candidates = [
        _player_row(res, int(i), p, surv, taken_by_me, next_turn) for i in cand_idx
    ]

    slipping: list[dict] = []
    if next_turn:
        # a below-replacement player who disappears is not a loss
        live = (p >= AVAILABLE_THRESHOLD) & (res.pool.vor > 0)
        gone_soon = live & (
            ~np.isfinite(surv) | (np.nan_to_num(surv, nan=0.0) < TAKE_NOW_SURVIVAL)
        )
        idx = np.where(gone_soon)[0]
        idx = idx[np.argsort(-res.pool.vor[idx])][:size]
        chosen = {c["canonical_id"] for c in candidates}
        slipping = [
            r for r in (
                _player_row(res, int(i), p, surv, taken_by_me, next_turn) for i in idx
            ) if r["canonical_id"] not in chosen
        ]

    return {"candidates": candidates, "slipping_away": slipping}


def recommended_shape(res: E.DraftResult, turn: int) -> dict:
    """What my policy actually did at this turn, as a distribution.

    Empirical rather than prescribed: "takes RB 62% of the time" is a statement
    about the policy meeting a real draft, not a rule someone wrote down.
    """
    col = turn - 1
    pos = res.pool.pos_idx[res.picks[:, col]]
    counts = np.bincount(pos, minlength=len(PL.POSITIONS)) / res.n_sims
    d = {PL.POSITIONS[i]: round(float(counts[i]), 4)
         for i in range(len(PL.POSITIONS)) if counts[i] > 0}
    return {
        "distribution": dict(sorted(d.items(), key=lambda kv: -kv[1])),
        "modal_position": PL.POSITIONS[int(counts.argmax())],
        "concentration": round(float(counts.max()), 4),
    }


# ── branch points ────────────────────────────────────────────────────────────
MIN_BRANCH_PROB = 0.15
"""A branch that fires one time in ten is not a plan, it is a footnote."""


def _qb_gone_before(res: E.DraftResult, turn: int) -> np.ndarray:
    qb = res.pool.pos_idx[res.picks[:, : turn - 1]] == PL.POS_INDEX["QB"]
    return qb.sum(axis=1)


def _top_tier_gone(res: E.DraftResult, gone: np.ndarray, turn: int, position: str) -> np.ndarray:
    """True where no tier-1 player at that position survives to my pick."""
    members = np.where(
        (res.pool.pos_idx == PL.POS_INDEX[position]) & (res.pool.tier == 1)
    )[0]
    if members.size == 0:
        return np.zeros(res.n_sims, dtype=bool)
    return ~(gone[:, members] >= turn).any(axis=1)


def branches(res: E.DraftResult, gone: np.ndarray, turn: int) -> list[dict]:
    """§5's "if two QBs go before pick 14, pivot to X" — derived, not written.

    Split the simulations on something I can actually observe at the table, then
    compare what the policy does on each side. **Only splits that change the
    recommendation are reported.** A branch point that does not change the
    recommendation is not a branch point, and printing it would pad the sheet
    with advice that is really just noise.
    """
    col = turn - 1
    mine = res.pool.pos_idx[res.picks[:, col]]
    out = []

    qb_gone = _qb_gone_before(res, turn)
    for k in range(1, 8):
        cond = qb_gone >= k
        b = _compare(cond, mine, f"{k}+ QBs gone before my pick")
        if b:
            out.append(b)
            break                      # report the first threshold that bites

    for position in ("RB", "WR", "TE", "QB"):
        cond = _top_tier_gone(res, gone, turn, position)
        b = _compare(cond, mine, f"no tier-1 {position} survives to my pick")
        if b:
            out.append(b)
    return out


def _compare(cond: np.ndarray, mine: np.ndarray, label: str) -> dict | None:
    p = float(cond.mean())
    if not (MIN_BRANCH_PROB <= p <= 1 - MIN_BRANCH_PROB):
        return None
    a, b = mine[cond], mine[~cond]
    if a.size == 0 or b.size == 0:
        return None
    ha = np.bincount(a, minlength=len(PL.POSITIONS))
    hb = np.bincount(b, minlength=len(PL.POSITIONS))
    if int(ha.argmax()) == int(hb.argmax()):
        return None                    # same call either way: not a branch
    return {
        "condition": label,
        "p_condition": round(p, 3),
        "if_true_take": PL.POSITIONS[int(ha.argmax())],
        "if_true_share": round(float(ha.max() / a.size), 3),
        "if_false_take": PL.POSITIONS[int(hb.argmax())],
        "if_false_share": round(float(hb.max() / b.size), 3),
    }


# ── assembling one slot's plan ───────────────────────────────────────────────
@dataclass
class SlotPlan:
    slot: int
    turns: list[int]
    per_turn: list[dict]
    roster_strength: dict
    shape_by_round: list[dict]

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "pick_numbers": self.turns,
            "snake_gaps": np.diff(self.turns).tolist(),
            "per_turn": self.per_turn,
            "roster_strength": self.roster_strength,
            "shape_by_round": self.shape_by_round,
        }


def build_slot_plan(res: E.DraftResult, slot: int, value: np.ndarray) -> SlotPlan:
    gone = gone_matrix(res)
    turns = (E.turns_for_slot(slot, res.n_teams, res.rounds)).tolist()

    per_turn = []
    for i, t in enumerate(turns):
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        per_turn.append({
            "pick": int(t),
            "round": i + 1,
            "picks_until_next_turn": int(nxt - t) if nxt else None,
            "shortlist": shortlist(res, gone, int(t), nxt),
            "best_available_by_position": [
                best_available_tier(res, gone, int(t), p)
                for p in ("QB", "RB", "WR", "TE")
            ],
            "recommended": recommended_shape(res, int(t)),
            "branches": branches(res, gone, int(t)),
        })

    shape = []
    for i, t in enumerate(turns):
        r = recommended_shape(res, int(t))
        shape.append({"round": i + 1, "pick": int(t), **r})

    return SlotPlan(
        slot=slot,
        turns=[int(t) for t in turns],
        per_turn=per_turn,
        roster_strength={
            "mean_weekly_points": round(float(value.mean()), 2),
            "p10": round(float(np.percentile(value, 10)), 2),
            "p50": round(float(np.percentile(value, 50)), 2),
            "p90": round(float(np.percentile(value, 90)), 2),
        },
        shape_by_round=shape,
    )


# ── §11 step 8: do the plans actually differ? ────────────────────────────────
def shape_matrix(plans: dict[int, SlotPlan]) -> np.ndarray:
    """(n_slots, n_rounds, n_positions) recommended shape, for comparison."""
    slots = sorted(plans)
    n_r = len(plans[slots[0]].shape_by_round)
    out = np.zeros((len(slots), n_r, len(PL.POSITIONS)))
    for si, s in enumerate(slots):
        for ri, row in enumerate(plans[s].shape_by_round):
            for pi, p in enumerate(PL.POSITIONS):
                out[si, ri, pi] = row["distribution"].get(p, 0.0)
    return out


def divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Mean total-variation distance between two shape profiles."""
    return float(np.abs(a - b).sum(axis=-1).mean() / 2)


def plans_differ(
    plans: dict[int, SlotPlan], noise_pairs: list[tuple[SlotPlan, SlotPlan]]
) -> dict:
    """§11 step 8, with the obvious trap closed.

    "Plans differ meaningfully; slot 5 shouldn't read like slot 1." Nine plans
    built from Monte Carlo will differ *somewhat* whatever happens, so the
    comparison that matters is **between-slot divergence against within-slot
    noise** — the same slot re-run under a different seed. If between is not
    clearly larger than within, the nine plans are one list re-indexed and this
    reports exactly that instead of pointing at decimals.
    """
    m = shape_matrix(plans)
    slots = sorted(plans)
    between = []
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            between.append(divergence(m[i], m[j]))
    within = [
        divergence(shape_matrix({0: a})[0], shape_matrix({0: b})[0])
        for a, b in noise_pairs
    ]
    mb = float(np.mean(between)) if between else 0.0
    mw = float(np.mean(within)) if within else 0.0

    extremes = None
    if 1 in plans and 5 in plans and 9 in plans:
        i1, i5, i9 = slots.index(1), slots.index(5), slots.index(9)
        extremes = {
            "slot1_vs_slot5": round(divergence(m[i1], m[i5]), 4),
            "slot1_vs_slot9": round(divergence(m[i1], m[i9]), 4),
            "slot5_vs_slot9": round(divergence(m[i5], m[i9]), 4),
        }

    ratio = mb / mw if mw > 0 else float("inf")
    return {
        "mean_between_slot_divergence": round(mb, 4),
        "mean_within_slot_noise": round(mw, 4),
        "ratio": round(ratio, 2),
        "extremes": extremes,
        "verdict": (
            f"PASS: between-slot divergence {mb:.4f} is {ratio:.1f}x the "
            f"re-seed noise {mw:.4f} — the plans carry slot-specific content."
            if ratio >= 2.0 else
            f"FAIL: between-slot divergence {mb:.4f} against re-seed noise "
            f"{mw:.4f} ({ratio:.1f}x). The nine plans are close to one list "
            "re-indexed; §5 explicitly warns against shipping that."
        ),
    }
