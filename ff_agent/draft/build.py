"""draft_sim.json (§5) — every slot, every policy, scored in title probability.

This is the first milestone that composes the whole pipeline: M4's board is the
value, M5's opponents are the environment, M6's season simulator is the
objective. §5 asks for 10,000+ simulated drafts per slot, all nine slots, "scored
by championship probability from the season simulator, not raw points".

Three things are deliberately reported rather than smoothed over.

**Seat assignment is unknown.** I learn my slot an hour before the draft; I never
learn where the other eight sit. Opponents fill the remaining seats in a fixed
rotation and ``seat_sensitivity`` measures what that simplification costs by
reshuffling them. If the spread is small the shortcut is earned; if not, the
number says so.

**Absolute title probabilities are optimistic.** M5's model puts the actual pick
at a median rank of 13, so a policy that optimises exactly against it beats it by
more than it would beat eight humans. Policy-vs-policy and slot-vs-slot
comparisons are sound — every arm faces the same opponents — and the absolute
level is not. It is labelled that way in the artifact, not just here.

**The projections are the binding constraint** (M6). This tells you *shape* —
when the quarterback run arrives, where the cliffs are, what actually survives to
your turn — and it is weak at separating players of similar projected value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from ff_agent.config import (
    ARTIFACTS_DIR, MY_TEAM_NAME, N_TEAMS, OPPONENTS, ROSTER_TOTAL, SEASON,
    normalize_team_name,
)
from ff_agent.draft import calibrate as CB
from ff_agent.draft import engine as E
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.draft import surrogate as SUR
from ff_agent.opponents import history as H
from ff_agent.opponents import pickmodel as PM
from ff_agent.opponents import tendencies as TD

MY_MANAGER = "Jordan Britt"

GRID_STEP = 5.0
GRID_PAD = 12.0
"""The grid is sized from the simulated range, not hardcoded.

The first version fixed it at -30..+30 and every draft outcome above +30 was
silently clamped to the edge, scoring 0.21 absolute error on P(title) while
looking like an interpolation result. Samples inside the grid were accurate to
0.003. Bilinear interpolation clips without complaining, so the grid has to be
built around what the simulator actually produces."""

GRID_SDS = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 12.0])
"""Season-mean uncertainty. §2.4's ceiling axis."""


def managers_for_slot(slot: int, n_teams: int = N_TEAMS) -> list[str]:
    """Me at ``slot`` (1-based); opponents fill the rest in config order."""
    others = [H.canonical_manager(o["manager"]) for o in OPPONENTS]
    if len(others) != n_teams - 1:
        raise ValueError(f"{len(others)} opponents for a {n_teams}-team league")
    out = others[: slot - 1] + [MY_MANAGER] + others[slot - 1:]
    return out[:n_teams]


def team_name_for(manager: str) -> str:
    """Manager -> fantasy team name, spelled the way the SCHEDULE spells it.

    §1 stores "Hodor\u2019s Hodors" with a curly apostrophe; the league schedule
    runs every name through ``normalize_team_name`` first. Skipping that here
    produced a KeyError deep inside the season simulator after all 27 slot runs
    had already finished.
    """
    if manager == MY_MANAGER:
        return normalize_team_name(MY_TEAM_NAME)
    for o in OPPONENTS:
        if H.canonical_manager(o["manager"]) == manager:
            return normalize_team_name(o["team"])
    raise KeyError(manager)


def assert_teams_match_schedule(season: int = SEASON) -> None:
    """Fail before the expensive part, not after it."""
    from ff_agent.season import schedule as SCH

    known = set(SCH.games_played(season)["team"].to_list())
    mine = {team_name_for(m) for m in managers_for_slot(1)}
    missing = sorted(mine - known)
    if missing:
        raise RuntimeError(
            f"team names not present in the {season} league schedule: {missing}\n"
            f"  schedule has: {sorted(known)}"
        )


@dataclass
class SlotRun:
    slot: int
    policy: str
    result: E.DraftResult
    value: SUR.RosterValue
    delta: np.ndarray
    opponent_means: dict[str, float]


def _fit_shipped_offsets(
    pool: PL.DraftPool, season: int, n_sims: int = 1_500
) -> tuple[np.ndarray, pl.DataFrame]:
    """Offsets for the season being drafted, on all history, format-weighted.

    Unlike the backtest's strict arm this uses 2025 as well — it is the only
    2-QB draft this league has ever run, and ``FORMAT_MATCH_WEIGHT`` gives it
    triple weight. There is no way to validate that out of sample (validating it
    would need a second 2-QB season), so the backtest reports the strict number
    and this ships. The gap between the two is the honest error bar.
    """
    managers = managers_for_slot(1)
    tilt, reach = PM.build_lookups(
        TD.picks_with_consensus().filter(pl.col("canonical_id").is_not_null())
    )
    target = CB.target_rates(target_two_qb=season in H.TWO_QB_SEASONS)
    return CB.fit_offsets(
        pool, managers, tilt, reach, target, ROSTER_TOTAL, n_sims=n_sims
    )


def run_slot(
    pool: PL.DraftPool,
    slot: int,
    policy,
    tilt: dict,
    reach: dict,
    offsets: np.ndarray | None,
    n_sims: int = 10_000,
    season: int = SEASON,
    seed: int = 31,
    managers: list[str] | None = None,
) -> SlotRun:
    mgrs = managers or managers_for_slot(slot)
    res = E.simulate_drafts(
        pool, mgrs, my_index=mgrs.index(MY_MANAGER), policy=policy,
        tilt_lookup=tilt, reach=reach, n_sims=n_sims, rounds=ROSTER_TOTAL,
        seed=seed + slot * 17, pos_offsets=offsets,
    )
    val = SUR.lineup_stats(pool, res.my_roster())
    opp = {}
    for t, m in enumerate(mgrs):
        if m == MY_MANAGER:
            continue
        opp[team_name_for(m)] = float(SUR.lineup_stats(pool, res.roster_of(t)).mean.mean())
    league_other = float(np.mean(list(opp.values())))
    return SlotRun(slot, policy.name, res, val, val.mean - league_other, opp)


def grid_deltas(runs: list["SlotRun"]) -> np.ndarray:
    """Cover every delta the simulator produced, padded, so nothing clamps."""
    lo = min(float(r.delta.min()) for r in runs) - GRID_PAD
    hi = max(float(r.delta.max()) for r in runs) + GRID_PAD
    lo = np.floor(lo / GRID_STEP) * GRID_STEP
    hi = np.ceil(hi / GRID_STEP) * GRID_STEP
    return np.arange(lo, hi + GRID_STEP, GRID_STEP)


def fit_surrogate(
    runs: list[SlotRun], season: int = SEASON, n_sims: int = 8_000
) -> SUR.Surrogate:
    """One grid for every slot and policy, on the averaged opponent profile."""
    all_opp: dict[str, list[float]] = {}
    for r in runs:
        for k, v in r.opponent_means.items():
            all_opp.setdefault(k, []).append(v)
    means = {k: float(np.mean(v)) for k, v in all_opp.items()}
    base = float(np.mean(list(means.values())))
    offsets = {k: v - base for k, v in means.items()}
    opp_sd = float(np.mean([r.value.mean_sd.mean() for r in runs]))
    return SUR.fit(
        base_mean=base, opponent_offsets=offsets, opponent_mean_sd=opp_sd,
        deltas=grid_deltas(runs), sds=GRID_SDS, season=season, n_sims=n_sims,
    )


def score(run: SlotRun, sur: SUR.Surrogate) -> dict:
    p = sur.evaluate(run.delta, run.value.mean_sd)
    pos = np.array(run.result.pool.df["position"].to_list())
    mine = pos[run.result.my_roster()]
    shape = {q: float((mine == q).sum(axis=1).mean()) for q in PL.POSITIONS}
    return {
        "slot": run.slot,
        "policy": run.policy,
        "mean_weekly_points": round(float(run.value.mean.mean()), 2),
        "mean_vs_league": round(float(run.delta.mean()), 2),
        "mean_sd": round(float(run.value.mean_sd.mean()), 2),
        "bye_cost": round(float(run.value.bye_cost.mean()), 2),
        "p_title": round(float(p["p_title"].mean()), 4),
        "p_top2": round(float(p["p_top2"].mean()), 4),
        "p_playoffs": round(float(p["p_playoffs"].mean()), 4),
        "p_title_p10": round(float(np.percentile(p["p_title"], 10)), 4),
        "p_title_p90": round(float(np.percentile(p["p_title"], 90)), 4),
        "roster_shape": {k: round(v, 2) for k, v in shape.items()},
        "sanity_alarms": _alarms(run.result),
    }


def _alarms(res: E.DraftResult) -> list[str]:
    from ff_agent.draft import backtest as BT

    return BT.sanity_alarms(res)


def decompose_edge(
    pool: PL.DraftPool,
    tilt: dict,
    reach: dict,
    offsets: np.ndarray | None,
    slot: int = 5,
    n_sims: int = 4_000,
    season: int = SEASON,
) -> pl.DataFrame:
    """How much of my simulated margin is real, and how much is the model's?

    Three reference points, in order:

    * ``model_seat`` — I draft exactly like the opponent model. Must land at
      zero; if it does not, the simulator is not symmetric and nothing else here
      means anything.
    * ``best_consensus`` — I optimise, but against superflex ECR, so my board
      contributes nothing. Whatever margin survives is bought purely from M5's
      opponents picking with an 8-pick standard deviation while I do not.
    * ``best_vor`` and friends — the real policies.

    The gap between the second and third rows is the board's actual contribution.
    The gap between the first and second is the part that will not show up on
    draft day.
    """
    pols = [
        POL.ModelSeat(tilt=tilt, reach=reach, manager=MY_MANAGER, pos_offsets=offsets),
        POL.BestConsensus(),
        *POL.ALL_POLICIES,
    ]
    rows = []
    for p in pols:
        r = run_slot(pool, slot, p, tilt, reach, offsets, n_sims=n_sims, season=season)
        rows.append({
            "policy": p.name,
            "mean_weekly_points": round(float(r.value.mean.mean()), 2),
            "vs_league": round(float(r.delta.mean()), 2),
            "mean_sd": round(float(r.value.mean_sd.mean()), 2),
        })
    df = pl.DataFrame(rows)
    base = float(df.filter(pl.col("policy") == "model_seat")["vs_league"][0])
    consensus = float(df.filter(pl.col("policy") == "best_consensus")["vs_league"][0])
    return df.with_columns(
        (pl.col("vs_league") - consensus).round(2).alias("board_edge"),
        pl.lit(round(consensus - base, 2)).alias("noise_exploitation"),
    )


def qb_cap_sweep(
    pool: PL.DraftPool,
    tilt: dict,
    reach: dict,
    offsets: np.ndarray | None,
    slot: int = 5,
    caps: tuple[int, ...] = (2, 3, 4),
    n_sims: int = 4_000,
    season: int = SEASON,
) -> pl.DataFrame:
    """How many quarterbacks should I actually take? §2.1's question, simulated.

    This exists because of what an uncapped VOR policy does: it takes **four
    quarterbacks in the first five rounds**, in every single simulation. The
    board is not malfunctioning — Allen, Jackson, Maye and Burrow really do carry
    VOR 167/148/147/122, ahead of every other player. The flaw is using VOR for a
    bench pick at all. VOR is measured against the STARTER replacement (QB18), so
    it prices QB3 and QB4 as if they would start, in a league that starts two.

    M4 answered this question TWO, by comparing a specific pair (Allen wk 7,
    Jackson wk 13, neither bye free) against the best upside stash: 17.9 points
    versus 26.4. M7 answers it over the DISTRIBUTION of quarterbacks I actually
    end up with, and scores expected weekly points rather than ceiling. The two
    do not measure the same thing and they do not have to agree; the gap between
    them is a real open question, not a rounding difference.
    """
    rows = []
    for cap in caps:
        for cls, label in ((POL.BestVOR, "best_vor"), (POL.NeedWeighted, "need_weighted")):
            pol = cls(caps={"QB": cap})
            pol.name = f"{label}_qb{cap}"
            r = run_slot(pool, slot, pol, tilt, reach, offsets,
                         n_sims=n_sims, season=season)
            pos = np.array(pool.df["position"].to_list())
            mine = pos[r.result.my_roster()]
            rows.append({
                "qb_cap": cap,
                "base_policy": label,
                "mean_weekly_points": round(float(r.value.mean.mean()), 2),
                "vs_league": round(float(r.delta.mean()), 2),
                **{f"n_{q}": round(float((mine == q).sum(axis=1).mean()), 2)
                   for q in ("RB", "WR", "TE")},
            })
    return pl.DataFrame(rows).sort(["base_policy", "qb_cap"])


def seat_sensitivity(
    pool: PL.DraftPool,
    tilt: dict,
    reach: dict,
    offsets: np.ndarray | None,
    slot: int = 5,
    shuffles: int = 5,
    n_sims: int = 2_000,
    season: int = SEASON,
) -> pl.DataFrame:
    """What does not knowing the opponents' seats actually cost?

    I learn my own slot an hour before the draft and never learn theirs. The
    shipped runs put opponents in config order; this reshuffles them and reports
    the spread, so the shortcut is measured rather than assumed harmless.
    """
    base = managers_for_slot(slot)
    others = [m for m in base if m != MY_MANAGER]
    rng = np.random.default_rng(4)
    rows = []
    for i in range(shuffles):
        perm = list(rng.permutation(others)) if i else others
        mgrs = perm[: slot - 1] + [MY_MANAGER] + perm[slot - 1:]
        r = run_slot(pool, slot, POL.BestVOR(), tilt, reach, offsets,
                     n_sims=n_sims, season=season, managers=mgrs)
        rows.append({
            "seating": "config order" if i == 0 else f"shuffle {i}",
            "mean_weekly_points": round(float(r.value.mean.mean()), 2),
            "mean_vs_league": round(float(r.delta.mean()), 2),
        })
    return pl.DataFrame(rows)


def build(
    season: int = SEASON,
    n_sims: int = 10_000,
    policies=None,
    grid_sims: int = 8_000,
    validate_samples: int = 40,
) -> dict:
    assert_teams_match_schedule(season)
    pool = PL.build_pool(season, N_TEAMS)
    tilt, reach = PM.build_lookups(
        TD.picks_with_consensus().filter(pl.col("canonical_id").is_not_null())
    )
    offsets, trace = _fit_shipped_offsets(pool, season)
    pols = list(policies or POL.ALL_POLICIES)

    runs = [
        run_slot(pool, slot, p, tilt, reach, offsets, n_sims=n_sims, season=season)
        for slot in range(1, N_TEAMS + 1)
        for p in pols
    ]
    sur = fit_surrogate(runs, season=season, n_sims=grid_sims)
    scored = [score(r, sur) for r in runs]

    edge = decompose_edge(pool, tilt, reach, offsets, season=season)
    qb_caps = qb_cap_sweep(pool, tilt, reach, offsets, season=season)
    seats = seat_sensitivity(pool, tilt, reach, offsets, season=season)
    val = SUR.validate(
        sur, _validation_samples(runs, sur, validate_samples), season=season,
        n_sims=grid_sims,
    )
    err = float(val["abs_err_title"].max())
    spread = _policy_spread(scored)

    return {
        "season": season,
        "my_team": MY_TEAM_NAME,
        "n_sims_per_slot": n_sims,
        "pool_size": pool.n,
        "policies": [p.name for p in pols],
        "positional_offset_fit": trace.to_dicts(),
        "positional_offsets": CB.offsets_table(offsets).to_dicts(),
        "by_slot_and_policy": scored,
        "edge_decomposition": edge.to_dicts(),
        "qb_cap_sweep": qb_caps.to_dicts(),
        "seat_sensitivity": seats.to_dicts(),
        "surrogate": {
            "grid_deltas": sur.deltas.tolist(),
            "grid_sds": sur.sds.tolist(),
            "clamped_fraction": round(
                float(np.mean([r["clamped"] for r in val.to_dicts()])), 4
            ),
            "variance_slope": sur.variance_slope().to_dicts(),
            "variance_crossover": sur.variance_crossover(),
            "validation": val.to_dicts(),
            "max_abs_error_p_title": round(err, 4),
            "policy_spread_p_title": round(spread, 4),
            "verdict": _surrogate_verdict(err, spread),
        },
        "caveats": {
            "absolute_level": (
                "Title probabilities here are OPTIMISTIC, and edge_decomposition "
                "says by how much. best_consensus carries ZERO board edge by "
                "construction and still beats the opponent model by most of "
                "best_vor's margin — that difference is bought from M5's "
                "opponents drafting with an 8-pick standard deviation, not from "
                "anything that will happen on draft day. Compare policies and "
                "slots to each other; do not read the absolute level."
            ),
            "binding_constraint": (
                "M6 showed the projections, not the simulator, are the limit. This "
                "is reliable on draft SHAPE and weak on fine distinctions between "
                "players of similar projected value."
            ),
            "seat_assignment": (
                "Opponents' slots are unknown and are filled in config order. See "
                "seat_sensitivity for what that costs."
            ),
        },
    }


def _validation_samples(runs: list[SlotRun], sur: SUR.Surrogate, k: int) -> list[dict]:
    """Real simulated draft outcomes, with their REAL opponent means."""
    rng = np.random.default_rng(17)
    out = []
    for _ in range(k):
        r = runs[int(rng.integers(len(runs)))]
        i = int(rng.integers(r.result.n_sims))
        means = dict(r.opponent_means)
        means[sur.my_team] = float(r.value.mean[i])
        msds = {t: sur.opponent_mean_sd for t in means}
        msds[sur.my_team] = float(r.value.mean_sd[i])
        out.append({"team_means": means, "team_mean_sds": msds})
    return out


def _policy_spread(scored: list[dict]) -> float:
    by_slot: dict[int, list[float]] = {}
    for s in scored:
        by_slot.setdefault(s["slot"], []).append(s["p_title"])
    return float(np.mean([max(v) - min(v) for v in by_slot.values()]))


def _surrogate_verdict(err: float, spread: float) -> str:
    if spread <= 0:
        return "policies are indistinguishable; nothing to rank."
    if err >= spread:
        return (
            f"DO NOT RANK POLICIES: surrogate error ({err:.4f}) is at least as "
            f"large as the mean gap between policies ({spread:.4f}). Any ordering "
            "would be noise."
        )
    return (
        f"USABLE: surrogate error {err:.4f} against a mean policy gap of "
        f"{spread:.4f} ({err / spread:.1f}x). Rank with that ratio in mind."
    )


def write(season: int = SEASON, n_sims: int = 10_000, **kw) -> str:
    path = ARTIFACTS_DIR / "draft_sim.json"
    path.write_text(json.dumps(build(season, n_sims, **kw), indent=2, default=str))
    return str(path)
