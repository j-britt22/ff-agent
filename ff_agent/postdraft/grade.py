"""Every pick, graded against what was actually on the board at the time.

Two design decisions carry this module, and both are inherited from findings
that were expensive to make.

**1. A pick is scored over a PAIR of picks, not one.** M8:

    "Josh Allen has the highest VOR on the board (167.5) and is the THIRD best
    choice, because he comes back 97% of the time."

A pick is not good because the player was the best available; it is good because
no better *sequence* was available. So for a pick at ``k`` by a manager whose
next pick is ``k'``, this asks which legal pair (a at k, b at k') was worth the
most, and how far short of it the manager fell. Live, that needs the opponent
model to guess who survives to ``k'``. Afterwards it does not: **the survivors
are observed.** The one assumption is that the other managers would have picked
the same players regardless — unknowable, standard for any post-mortem, and
printed on the output rather than buried here.

**2. Value is the STARTING LINEUP, never a sum of VOR.** The first version of
this module summed VOR and handed round one a wall of Ds, because it rediscovered
M7's finding from the other side:

    "UNCAPPED VOR TAKES FOUR QUARTERBACKS IN THE FIRST FIVE ROUNDS, EVERY TIME.
    Not a policy bug — the board really does rank Allen/Jackson/Maye/Burrow at
    VOR 167/148/147/122. The flaw is using VOR for a bench pick at all: VOR is
    measured against the STARTER replacement (QB18), so it prices QB3 and QB4 as
    though they would start."

Grading Ja'Marr Chase at 1.06 against a quarterback nobody could start was that
bug wearing a different hat. Pairs are now scored with ``surrogate.lineup_stats``
— the same optimal-lineup solver M7 used — against the manager's own play weeks,
so a third quarterback is worth what a third quarterback is actually worth, and
§2.1's bye weeks price themselves.

The comparison holds the manager's OTHER fifteen picks fixed. That is hindsight,
deliberately: the question a post-mortem should answer is "given the team you
ended up with, was this pick the right one", and that cannot be asked without
knowing the team you ended up with.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ff_agent.config import MY_GAME_WEEKS
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.draft import surrogate as SUR
from ff_agent.postdraft.results import DraftRecord

TOP_BY_VALUE = 25
"""How many of the best legal players by VOR to consider at each pick."""

TOP_PER_POSITION = 4
"""Plus this many at EVERY legal position, on top of the VOR-ranked shortlist.

Without it the candidate set inherits exactly the bias the module exists to
avoid: the board is QB-heavy at the top (M4: 12 QB in the top 30), so a pure
VOR shortlist can reach twenty-five deep without offering the tight end a roster
with no tight end obviously needs."""

SPREAD_FLOOR = 0.75
"""Minimum denominator, in weekly points. Late in the draft the board flattens
and the ratio would otherwise divide by ~0 and manufacture an F out of noise."""

GRADE_PERCENTILES = (
    (0.10, "A+"), (0.25, "A"), (0.40, "B+"), (0.55, "B"),
    (0.70, "C+"), (0.85, "C"), (0.95, "D"),
)
"""Where a pick's regret sits among all picks in THIS draft -> letter.

This is a curve, which the first version of the module deliberately was not, and
the reason for changing is worth stating because it is the same reason M7 gives
for its own headline numbers:

    "Compare policies and slots to each other; do not read the absolute level."

**Hindsight regret has no meaningful zero.** The "better pair" is chosen knowing
exactly who survived to the manager's next turn, so it beats what a human could
have known essentially always: measured on the 2025 draft, the MEDIAN pick came
out 8-10 weekly points short of it. Absolute thresholds over that distribution
graded four picks in five at D or F, which says nothing about any of them.

So the letter reports rank and the points report magnitude. A pick graded A is
one of the better decisions made in this draft; it is not a decision that beat
an oracle, because nothing does. ``regret_weekly_points`` is printed alongside
precisely so the curve cannot hide how big or small the real gap was.

The curve runs **within each round**, not across the whole draft, because the
ratio carries a structural trend that has nothing to do with decision quality:
measured on 2025 the median ratio climbs from 0.74 in round one to 1.7 by round
fifteen, as the value spread between the best and the median option collapses
and the denominator with it. Curving globally would hand the late rounds a wall
of Ds for being late. The cost of curving by round is real and worth naming: a
round in which all nine managers picked well still produces a bottom pick, so a
single low grade means "the weakest of nine reasonable picks" at least as often
as it means a mistake. Read the points column before reacting to a letter."""

FAIL = "F"

GRADE_POINTS = {
    "A+": 4.3, "A": 4.0, "B+": 3.3, "B": 3.0,
    "C+": 2.3, "C": 2.0, "D": 1.0, "F": 0.0,
}


def _letters_for_group(ratios: np.ndarray) -> list[str]:
    """Percentile grades within one round, with TIES SHARING A GRADE.

    Ordinal ranking looked fine until the counterfactuals got good enough that
    most picks in a round came back at exactly zero regret — at which point the
    tie-break was row order, and two identical picks one seat apart were handed
    an A+ and a C. Averaging the rank over a tie group is the fix: a pick can
    only be graded below another if it was actually worse.
    """
    r = np.asarray(ratios, dtype=float)
    n = len(r)
    if n == 0:
        return []
    order = np.argsort(r, kind="stable")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and r[order[j + 1]] == r[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    out = []
    for q in (ranks + 0.5) / n:
        for cut, g in GRADE_PERCENTILES:
            if q <= cut:
                out.append(g)
                break
        else:
            out.append(FAIL)
    return out


def letters_from(ratios: np.ndarray, rounds: np.ndarray) -> list[str]:
    """Grade each pick against the other picks made in the SAME ROUND."""
    ratios = np.asarray(ratios, dtype=float)
    rounds = np.asarray(rounds)
    out: list[str] = [FAIL] * len(ratios)
    for r in np.unique(rounds):
        idx = np.flatnonzero(rounds == r)
        for i, g in zip(idx, _letters_for_group(ratios[idx])):
            out[i] = g
    return out


def gpa(grades: list[str]) -> float:
    return round(float(np.mean([GRADE_POINTS[g] for g in grades])), 2) if grades else 0.0


def _ecr_rank(pool: PL.DraftPool) -> np.ndarray:
    """Consensus rank over the pool, 1-based, unranked at the back.

    M3 measured our own model as materially WORSE than consensus alone, which is
    why the board blends at 0.12 rather than replacing it. The same caution
    applies to grading: a pick that reaches on our board but sits fine on ECR is
    a different animal from one that reaches on both, and reporting only our own
    number would hide the difference.
    """
    cf = pool.consensus_fraction.astype(float).copy()
    cf[~np.isfinite(cf)] = 9.9
    return np.argsort(np.argsort(cf, kind="stable"), kind="stable") + 1


def _legal(av: np.ndarray, counts: np.ndarray, round_no: int, rounds: int,
           pos_idx: np.ndarray) -> np.ndarray:
    """Available AND legal for this roster at this point in the draft.

    §3.5's "never a kicker before the last two rounds" is KEPT here, and the
    round trip that established it is worth recording. It was lifted first, on
    the argument that the rule is my discipline and grading a rival against it is
    unfair. What that produced was kicker worship: with the ban gone, the best
    pair at nine consecutive picks of mine came back as "Texans D/ST + Brandon
    Aubrey", because the rank->points curve gives the top kicker a real edge over
    the twentieth and nothing stopped the counterfactual from banking it fifteen
    rounds early.

    §3.5 exists precisely because that edge does not survive contact with a
    season: D/ST and K "reward active management", the advice is to stream both
    weekly, and the difference between kickers is not something you spend a
    seventh-round pick to own. A counterfactual allowed to draft them early is
    not a better yardstick, it is a worse one.

    The reason it was lifted turned out to be a misdiagnosis. Rounds 15-16 did
    show inflated regret — but on a TRUNCATED corpus (a replay log stopping at
    round 16, in which no team ever drafted a defense), so the last rounds were
    being graded against rosters that structurally could not be finished. On a
    full draft the feasibility branch of ``allowed_positions`` handles it: three
    picks left and two of them owed to K and D/ST still leaves round 15 free.
    """
    allowed = POL.allowed_positions(counts, round_no, rounds)[0]
    return av & allowed[pos_idx]


def _candidates(vor: np.ndarray, legal: np.ndarray, pos_idx: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(legal)
    if idx.size == 0:
        return idx
    order = idx[np.argsort(-vor[idx], kind="stable")]
    keep = list(order[:TOP_BY_VALUE])
    for p in range(len(PL.POSITIONS)):
        at_p = order[pos_idx[order] == p][:TOP_PER_POSITION]
        keep.extend(at_p.tolist())
    return np.array(sorted(set(int(i) for i in keep)), dtype=np.int64)


def _counts_after(counts: np.ndarray, pos: int) -> np.ndarray:
    out = counts.copy()
    out[0, pos] += 1
    return out


def analyse(
    record: DraftRecord,
    play_weeks: dict[str, tuple[int, ...]] | None = None,
    rosters: dict[str, list[int]] | None = None,
) -> pl.DataFrame:
    """One row per pick, with its regret and the pair that would have beaten it.

    ``rosters`` is the context each pick is scored inside — normally the
    manager's real final roster, but on a draft still in progress it should be
    the roster EXTENDED with their remaining picks (``finish.project_remaining``).
    Scoring against a half-built roster does not merely add noise, it inverts the
    answer: an unfilled kicker slot makes a kicker the most valuable addition at
    every pick in the draft.
    """
    pool = record.pool
    vor = pool.vor.astype(float)
    pos_idx = pool.pos_idx
    names = pool.df["name"].to_list()
    positions = pool.df["position"].to_list()
    source = (
        pool.df["value_source"].to_list()
        if "value_source" in pool.df.columns
        else ["projected"] * pool.n
    )
    ecr = _ecr_rank(pool)
    board_rank = np.argsort(np.argsort(-vor, kind="stable"), kind="stable") + 1

    picks = record.picks.sort("overall").to_dicts()
    rounds = record.rounds
    weeks = play_weeks or {}

    # availability before each pick; each manager's counts before each pick;
    # each manager's FINAL roster and where in it every pick sits
    avail = np.ones(pool.n, dtype=bool)
    taken_before, counts_before = [], []
    counts = {m: np.zeros((1, len(PL.POSITIONS)), dtype=np.int16)
              for m in record.managers}
    real_roster: dict[str, list[int]] = {}
    slot_of: list[int] = []
    for r in picks:
        m = r["manager"]
        taken_before.append(avail.copy())
        counts_before.append(
            counts.setdefault(m, np.zeros((1, len(PL.POSITIONS)), dtype=np.int16)).copy()
        )
        seat = real_roster.setdefault(m, [])
        slot_of.append(len(seat))
        seat.append(int(r["pool_index"]))
        avail[int(r["pool_index"])] = False
        counts[m][0, pos_idx[int(r["pool_index"])]] += 1

    next_of = _next_pick_index(picks)
    rows = []
    for k, r in enumerate(picks):
        p = int(r["pool_index"])
        mgr = r["manager"]
        rnd = int(r["round"])
        base = np.array((rosters or real_roster)[mgr], dtype=np.int64)
        wk = tuple(weeks.get(mgr, MY_GAME_WEEKS))

        legal_a = _legal(taken_before[k], counts_before[k], rnd, rounds, pos_idx)
        cand_a = _candidates(vor, legal_a, pos_idx)

        j = next_of[k]
        if j is None:
            best, best_names, actual, spread = _best_single(
                pool, base, slot_of[k], cand_a, p, wk
            )
        else:
            nxt = picks[j]
            av_next = taken_before[j].copy()
            av_next[p] = True          # holding others' picks fixed, he survives
            best, best_names, actual, spread = _best_pair(
                pool, base, slot_of[k], slot_of[j], cand_a, av_next,
                counts_before[k], int(nxt["round"]), rounds, p,
                int(nxt["pool_index"]), vor, pos_idx, wk,
            )

        regret = max(float(best - actual), 0.0)
        ratio = regret / max(float(spread), SPREAD_FLOOR)
        rows.append({
            "overall": int(r["overall"]),
            "round": rnd,
            "pick_in_round": int(r["pick_in_round"]),
            "manager": mgr,
            "name": r["name"],
            "canonical_id": r["canonical_id"],
            "position": positions[p],
            "nfl_team": r.get("nfl_team"),
            "vor": round(float(vor[p]), 2),
            "board_rank": int(board_rank[p]),
            "ecr_rank": int(ecr[p]),
            "value_vs_board": int(board_rank[p]) - int(r["overall"]),
            "value_vs_ecr": int(ecr[p]) - int(r["overall"]),
            "regret_weekly_points": round(regret, 2),
            "regret_ratio": round(float(ratio), 3),
            "better_pair": " + ".join(best_names) if best_names else None,
            "next_own_pick": names[int(picks[j]["pool_index"])] if j is not None else None,
            "would_have_survived": (
                bool(taken_before[j][p]) if j is not None else None
            ),
            "off_board": source[p] == "off_board",
            "filled_starter_need": _filled_need(counts_before[k], int(pos_idx[p])),
            "pick_value_spread": round(float(spread), 2),
        })
    out = pl.DataFrame(rows)
    return out.with_columns(
        pl.Series("grade", letters_from(
            out["regret_ratio"].to_numpy(), out["round"].to_numpy()
        ))
    )


def _lineup_values(pool, rosters: np.ndarray, wk) -> np.ndarray:
    """Expected weekly starting points for each candidate roster."""
    return SUR.lineup_stats(pool, rosters, play_weeks=wk).mean


def _best_single(pool, base, slot_a, cand_a, taken_a, wk):
    """The manager's last pick: one substitution, no follow-on."""
    if cand_a.size == 0:
        return 0.0, [], 0.0, SPREAD_FLOOR
    rosters = np.repeat(base[None, :], cand_a.size, axis=0)
    rosters[:, slot_a] = cand_a
    vals = _lineup_values(pool, rosters, wk)
    actual = float(_lineup_values(pool, base[None, :], wk)[0])
    b = int(np.argmax(vals))
    names = pool.df["name"].to_list()
    spread = _spread_from(vals)
    return float(vals[b]), ([names[int(cand_a[b])]] if cand_a[b] != taken_a else []), \
        actual, spread


def _best_pair(pool, base, slot_a, slot_b, cand_a, av_next, counts, round_b,
               rounds, taken_a, taken_b, vor, pos_idx, wk):
    """Best legal (a here, b at my next pick), holding the rest of the roster fixed.

    Legality at ``k'`` depends on ``a`` only through the position it consumed, so
    the b-shortlist is built once per position rather than once per candidate.
    """
    names = pool.df["name"].to_list()
    if cand_a.size == 0:
        actual = float(_lineup_values(pool, base[None, :], wk)[0])
        return actual, [], actual, SPREAD_FLOOR

    cand_b_for_pos: dict[int, np.ndarray] = {}
    for pa in set(int(x) for x in pos_idx[cand_a]):
        allowed_b = POL.allowed_positions(
            _counts_after(counts, pa), round_b, rounds
        )[0]
        cand_b_for_pos[pa] = _candidates(vor, av_next & allowed_b[pos_idx], pos_idx)

    pairs_a, pairs_b = [], []
    for a in cand_a:
        for b in cand_b_for_pos[int(pos_idx[a])]:
            if b == a:
                continue
            pairs_a.append(int(a))
            pairs_b.append(int(b))
    if not pairs_a:
        actual = float(_lineup_values(pool, base[None, :], wk)[0])
        return actual, [], actual, SPREAD_FLOOR

    rosters = np.repeat(base[None, :], len(pairs_a), axis=0)
    rosters[:, slot_a] = np.array(pairs_a)
    rosters[:, slot_b] = np.array(pairs_b)
    vals = _lineup_values(pool, rosters, wk)
    best = int(np.argmax(vals))
    actual = float(_lineup_values(pool, base[None, :], wk)[0])

    chosen = []
    if pairs_a[best] != taken_a or pairs_b[best] != taken_b:
        chosen = [names[pairs_a[best]], names[pairs_b[best]]]

    return float(vals[best]), chosen, actual, _spread_from(vals)


def _spread_from(vals: np.ndarray) -> float:
    """How much was at stake at this pick: the best option minus the MEDIAN one.

    The denominator has to be built from the same distribution as the numerator
    or the ratio is meaningless. An earlier version measured the gap between the
    best option and the ninth best, which sounds like "one round's worth of
    board" and is not: those top options all pair with the same strong follow-on
    pick, so they bunch within a point of each other while the manager's actual
    pair can sit ten points below. Best-minus-median is the honest scale — it
    asks how much the CHOICE was worth, not how tightly the top of the shortlist
    is packed.
    """
    if vals.size < 2:
        return SPREAD_FLOOR
    return float(np.max(vals) - np.median(vals))


def _next_pick_index(picks: list[dict]) -> list[int | None]:
    out: list[int | None] = [None] * len(picks)
    last: dict[str, int] = {}
    for i in range(len(picks) - 1, -1, -1):
        m = picks[i]["manager"]
        out[i] = last.get(m)
        last[m] = i
    return out


def _filled_need(counts: np.ndarray, pos: int) -> bool:
    name = PL.POSITIONS[pos]
    return bool(counts[0, pos] < POL.REQUIRED_STARTERS.get(name, 0))
