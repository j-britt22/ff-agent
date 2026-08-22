"""What to do, right now, given what has actually happened.

§4 gives this one second. The measured cost of re-simulating the REMAINDER of
the draft is 0.61s at 500 sims from pick 40 and 0.27s from pick 100, so M9
re-simulates rather than filtering M8's precomputed plan. That matters more than
it sounds: a plan is built for the expected draft, and ten picks in, reality has
already diverged from it.

Two things fall out of conditioning on the real board that a static plan cannot
do.

**The QB scenario resolves itself.** M8's honest headline was that only 7 to 10
of 17 turns survive all three quarterback-timing scenarios, because the league
has run exactly one 2-QB draft. But you do not have to stay uncertain — you
*watch* which world you are in. Comparing quarterbacks actually taken against
each scenario's fitted rate identifies it within about two rounds, and the sim
is then run under that scenario instead of the blend.

**Advice follows MY roster, not the policy's.** I will deviate; the engine
replays the real history, so the simulated version of me starts from the roster
I actually have.

Nothing here writes anything anywhere (§0.1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ff_agent.draft import calibrate as CB
from ff_agent.draft import engine as E
from ff_agent.draft import plans as PN
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.draft import scenarios as SC
from ff_agent.live.state import DraftState

LATENCY_BUDGET = 0.60
"""Seconds. §4 allows one; the margin absorbs a slow laptop and the print."""

THROUGHPUT = 99_000.0
"""Measured pick-simulations per second (500/2000-sim runs both landed here).
Used to size the sim count for the budget, then corrected from what actually
happened."""

SURVEY_SHARE = 0.35
"""Share of the budget spent on the survey run (availability, tiers, branches).
The rest prices the concrete options, which is what actually decides the pick."""

MIN_SIMS = 150
MAX_SIMS = 4_000
SCENARIO_MIN_PICKS = 9
"""Below one full round there is not enough evidence to call the QB scenario."""


@dataclass
class Option:
    """One concrete choice, and what taking it is worth."""

    index: int
    name: str
    position: str
    team: str
    vor: float
    tier: int
    free_bye: bool
    survival_to_next_turn: float | None
    never_passed_on: bool
    roster_value: float
    delta_vs_best: float
    n_sims: int

    def line(self) -> str:
        d = "  best" if self.delta_vs_best >= -1e-9 else f"{self.delta_vs_best:+6.2f}"
        bye = " *bye5/14" if self.free_bye else ""
        back = (
            "never comes back" if self.never_passed_on
            else ("—" if self.survival_to_next_turn is None
                  else f"back {self.survival_to_next_turn:.0%}")
        )
        return (f"  {d}  {self.name[:22]:22s} {self.position:3s} "
                f"vor {self.vor:6.1f}  {back}{bye}")


@dataclass
class Advice:
    pick: int
    round_no: int
    scenario: str
    scenario_evidence: dict
    n_sims: int
    elapsed: float
    candidates: list[dict]
    slipping_away: list[dict]
    best_by_position: list[dict]
    recommended: dict
    branches: list[dict]
    options: list[Option] = field(default_factory=list)
    alarms: list[str] = field(default_factory=list)
    degraded: str = ""

    def headline(self) -> str:
        if self.options:
            o = self.options[0]
            second = (
                f"; next best {self.options[1].name} "
                f"({self.options[1].delta_vs_best:+.2f})"
                if len(self.options) > 1 else ""
            )
            return f"TAKE {o.name} ({o.position}, {o.team}){second}"
        if not self.candidates:
            return "no candidate — the pool is exhausted or the roster is full"
        top = self.candidates[0]
        return f"{top['name']} ({top['position']}) — {top['p_i_take_him_here']:.0%}"


def sims_for_budget(picks_remaining: int, budget: float = LATENCY_BUDGET) -> int:
    if picks_remaining <= 0:
        return MIN_SIMS
    n = int(budget * THROUGHPUT / picks_remaining)
    return int(np.clip(n, MIN_SIMS, MAX_SIMS))


def detect_scenario(
    state: DraftState, targets: dict[str, np.ndarray]
) -> tuple[str, dict]:
    """Which QB-timing world is this draft actually in?

    Each scenario was fitted to a target P(position | phase), so the quarterbacks
    it expects by pick k is just those rates summed over the picks so far. The
    scenario whose expectation is closest to what has happened is the one to
    simulate under. Before a full round there is not enough evidence and this
    says so rather than guessing.
    """
    n = len(state.history)
    observed = sum(
        1 for i in state.history if PL.POSITIONS[state.pool.pos_idx[i]] == "QB"
    )
    total = state.total_picks
    expected = {}
    for name, target in targets.items():
        e = 0.0
        for k in range(n):
            ph = E._phase_index((k + 1) / total)
            e += float(target[ph, PL.POS_INDEX["QB"]])
        expected[name] = round(e, 2)

    evidence = {
        "picks_seen": n,
        "qbs_taken": observed,
        "expected_by_scenario": expected,
        "confident": n >= SCENARIO_MIN_PICKS,
    }
    if n < SCENARIO_MIN_PICKS:
        evidence["note"] = (
            f"fewer than {SCENARIO_MIN_PICKS} picks seen — using the shipped "
            "blend until the draft says otherwise"
        )
        return "shipped", evidence

    best = min(expected, key=lambda k: abs(expected[k] - observed))
    evidence["note"] = (
        f"{observed} QBs taken in {n} picks; closest is {best} "
        f"(expects {expected[best]})"
    )
    return best, evidence


OPTION_COUNT = 5
"""How many choices to price. The budget is split between them, so this trades
directly against the precision of each."""


def candidate_options(state: DraftState, k: int = OPTION_COUNT) -> list[int]:
    """The choices worth pricing: best legal available, by VOR, per position.

    One per position rather than the top k overall, because the board is
    QB-heavy (M4, M7) and a flat VOR ranking would price five quarterbacks
    against each other while ignoring the running back about to disappear.
    """
    avail = state.available_mask()
    allowed = POL.allowed_positions(
        state.counts_array(),
        round_no=(state.next_pick - 1) // state.n_teams + 1,
        rounds=state.rounds,
    )[0]
    out = []
    for pi, pos in enumerate(PL.POSITIONS):
        if not allowed[pi]:
            continue
        mask = avail & (state.pool.pos_idx == pi)
        if not mask.any():
            continue
        out.append(int(np.flatnonzero(mask)[0]))     # pool is VOR-sorted
    out.sort(key=lambda i: -state.pool.vor[i])
    return out[:k]


def price_options(
    state: DraftState,
    tilt: dict,
    reach: dict,
    offsets: np.ndarray,
    n_sims: int,
    qb_cap: int | None,
    seed: int,
) -> list[Option]:
    """What is each choice actually worth? Same budget, split between them.

    A deterministic policy has exactly ONE answer at a known board, so "a shortlist with
    probabilities" collapses to 100% at the current turn — true, and useless
    while a clock runs. What decides the pick is the counterfactual: force each
    candidate as my pick, simulate the rest of the draft, and compare the roster
    I end up with. Splitting the sims between k options costs the same total
    pick-simulations as one undivided run, so this fits the budget unchanged.
    """
    from ff_agent.draft import surrogate as SUR

    cands = candidate_options(state)
    if not cands:
        return []
    per = max(n_sims // len(cands), 40)

    pol = POL.NeedWeighted(caps={"QB": qb_cap} if qb_cap else None)
    names = state.pool.df["name"].to_list()
    teams = state.pool.df["team"].to_list()

    scored = []
    for j, c in enumerate(cands):
        res = E.simulate_drafts(
            state.pool, state.managers, my_index=state.my_index, policy=pol,
            tilt_lookup=tilt, reach=reach, n_sims=per, rounds=state.rounds,
            seed=seed + 7919 * j, pos_offsets=offsets,
            history=list(state.history) + [c],
        )
        val = SUR.lineup_stats(state.pool, res.my_roster())
        scored.append((c, float(val.mean.mean()), per))

    best = max(v for _, v, _ in scored)
    out = [
        Option(
            index=c,
            name=names[c],
            position=PL.POSITIONS[state.pool.pos_idx[c]],
            team=teams[c] or "FA",
            vor=round(float(state.pool.vor[c]), 1),
            tier=int(state.pool.tier[c]),
            free_bye=bool(state.pool.free_bye[c]),
            survival_to_next_turn=None,
            never_passed_on=False,
            roster_value=round(v, 2),
            delta_vs_best=round(v - best, 2),
            n_sims=n,
        )
        for c, v, n in scored
    ]
    out.sort(key=lambda o: -o.roster_value)
    return out


def advise(
    state: DraftState,
    tilt: dict,
    reach: dict,
    scen: dict[str, SC.Scenario],
    targets: dict[str, np.ndarray],
    qb_cap: int | None = 3,
    budget: float = LATENCY_BUDGET,
    seed: int = 4242,
) -> Advice:
    """One turn's recommendation, conditioned on the real board."""
    if state.complete:
        raise ValueError("the draft is complete")

    pick = state.next_pick
    remaining = state.total_picks - len(state.history)
    # The budget covers BOTH runs. Option pricing is what decides the pick, so
    # it gets the larger share; the survey run only has to produce availability,
    # tier survival and branch points, which are far less sensitive to n.
    n_sims = sims_for_budget(remaining, budget * SURVEY_SHARE)
    n_option_sims = sims_for_budget(remaining, budget * (1 - SURVEY_SHARE))
    name, evidence = detect_scenario(state, targets)

    pol = POL.NeedWeighted(caps={"QB": qb_cap} if qb_cap else None)
    pol.name = f"need_weighted_qb{qb_cap}" if qb_cap else "need_weighted"

    t0 = time.perf_counter()
    res = E.simulate_drafts(
        state.pool, state.managers, my_index=state.my_index, policy=pol,
        tilt_lookup=tilt, reach=reach, n_sims=n_sims, rounds=state.rounds,
        seed=seed + pick, pos_offsets=scen[name].offsets, history=list(state.history),
    )
    elapsed = time.perf_counter() - t0

    gone = PN.gone_matrix(res)
    later = state.my_turns()[state.my_turns() > pick]
    nxt = int(later[0]) if later.size else None

    sl = PN.shortlist(res, gone, pick, nxt)

    options = price_options(
        state, tilt, reach, scen[name].offsets, n_option_sims, qb_cap, seed + 31
    )
    # attach the survival numbers the counterfactual run cannot see
    if nxt:
        surv = PN.conditional_survival(gone, pick, nxt, res.picks[:, pick - 1])
        for o in options:
            v = float(surv[o.index])
            o.never_passed_on = not np.isfinite(v)
            o.survival_to_next_turn = None if o.never_passed_on else round(v, 3)

    elapsed = time.perf_counter() - t0
    return Advice(
        pick=pick,
        round_no=(pick - 1) // state.n_teams + 1,
        scenario=name,
        scenario_evidence=evidence,
        n_sims=n_sims,
        elapsed=round(elapsed, 3),
        options=options,
        candidates=sl["candidates"],
        slipping_away=sl["slipping_away"],
        best_by_position=[
            PN.best_available_tier(res, gone, pick, p) for p in ("QB", "RB", "WR", "TE")
        ],
        recommended=PN.recommended_shape(res, pick),
        branches=PN.branches(res, gone, pick),
        alarms=state.alarms(),
        degraded="" if elapsed <= budget * 1.5 else (
            f"took {elapsed:.2f}s against a {budget:.2f}s budget — "
            f"{n_sims} sims may be too many for this machine"
        ),
    )


def load_context(season: int, pool: PL.DraftPool, managers: list[str], rounds: int):
    """The slow, once-per-session setup: manager terms and the three scenarios.

    Runs BEFORE the draft starts. On the clock, nothing here is repeated.
    """
    import polars as pl

    from ff_agent.opponents import pickmodel as PM
    from ff_agent.opponents import tendencies as TD

    from ff_agent.config import ARTIFACTS_DIR

    tilt, reach = PM.build_lookups(
        TD.picks_with_consensus().filter(pl.col("canonical_id").is_not_null())
    )
    # Fitting the three scenarios costs ~45s. `cli plans` writes them; draft
    # morning must not pay that, and with the WiFi off it may not be able to.
    scen = SC.load(ARTIFACTS_DIR)
    if scen is None:
        scen = SC.fit_all(pool, managers, tilt, reach, rounds, season)
        SC.save(scen, ARTIFACTS_DIR)
    prior = (2023, 2024)
    targets = {
        "qb_cold": CB.target_rates(seasons=prior, target_two_qb=True),
        "shipped": CB.target_rates(target_two_qb=True),
        "qb_hot": CB.target_rates(seasons=(2025,), target_two_qb=True),
    }
    return tilt, reach, scen, targets
