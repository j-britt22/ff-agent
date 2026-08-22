"""MILESTONE 7 GATE (§11 step 7): "10k sims from a known slot reproduce last
year's draft's broad shape."

Replays the **2025** draft — 8 teams, the real draft order, my real slot 7 —
with the opponent model fitted on **2023-24 only**. That is M5's strict split,
so no season being predicted contributed to the parameters predicting it.

Two things make this a harder test than M5's evaluation, and both are deliberate:

* **The pool is wider than the players who were actually drafted.** M5 scored
  against a 136-player pool consisting of exactly who went. For a forward
  simulation that is lookahead — the simulator would be told the answer. Here it
  gets the top of the 2025 preseason superflex board plus K/D-ST ranked by 2024
  production, and has to find the 136 by itself.
* **Every seat runs the opponent model, mine included.** The 2025 draft had a
  human in seat 7, not my policy. Putting my policy there would test my policy,
  not the model being gated.

"Broad shape" is made operational as four measures, because a phrase that vague
can be declared passed by anyone:

a. **Positional composition by round** against actual, per round and summarised.
b. **QB timing** — first QB off the board and QBs inside three rounds. M5 found
   this is the single most distinctive thing about this league (2025: pick 7,
   and five).
c. **Calibration** — what share of actual picks land inside the simulated 80%
   interval for that player's pick number. Well calibrated is ~80%. This is the
   real test: (a) and (b) can both look fine while the simulator is confidently
   wrong about every individual player.
d. **§10 sanity alarms inside the simulated drafts.**

Nothing here is tuned to pass. ``sigma_sensitivity`` reports how the calibration
moves with the ADP kernel width precisely so that a miss can be attributed to
the model's strength rather than argued about.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ff_agent.draft import engine as E
from ff_agent.draft import pool as PL
from ff_agent.opponents import history as H
from ff_agent.opponents import pickmodel as PM
from ff_agent.opponents import tendencies as TD

BACKTEST_SEASON = 2025
POOL_TOP_N = 250
"""Skill players offered to the simulator. The 2025 draft used 136 picks total,
so the simulator must locate them inside a pool nearly twice that size."""


class NoOpPolicy:
    """Every seat uses the fitted model — see the module docstring."""

    name = "model"

    def __init__(self, tilt, reach, manager: str):
        self.tilt, self.reach, self.manager = tilt, reach, manager

    def choose(self, state):  # pragma: no cover - replaced by engine wiring
        raise NotImplementedError


def draft_order(season: int = BACKTEST_SEASON) -> list[str]:
    """Managers by draft slot, from round 1 of the real draft."""
    h = H.draft_history().filter(pl.col("season") == season)
    r1 = h.filter(pl.col("round") == 1).sort("round_pick")
    return r1["manager"].to_list()


def actual_picks(season: int = BACKTEST_SEASON) -> pl.DataFrame:
    picks = TD.picks_with_consensus().filter(pl.col("season") == season)
    return picks.sort("overall_pick").select(
        "overall_pick", "round", "round_pick", "manager", "canonical_id",
        "name", "position",
    )


def _fit_lookups(season: int):
    picks = TD.picks_with_consensus().filter(pl.col("canonical_id").is_not_null())
    return PM.build_lookups(picks.filter(pl.col("season") < season))


def build_backtest_pool(season: int = BACKTEST_SEASON, n_teams: int = 8) -> PL.DraftPool:
    from ff_agent.board import build as BB

    board = BB.build(season)
    return PL.build_pool(
        season=season, n_teams=n_teams, board=board,
        top_n=POOL_TOP_N, kdst_source="prior_season",
    )


def fit_strict_offsets(
    season: int = BACKTEST_SEASON,
    pool: PL.DraftPool | None = None,
    n_sims: int = 1_500,
) -> tuple[np.ndarray, pl.DataFrame]:
    """League positional offsets fitted on seasons STRICTLY BEFORE ``season``.

    For the 2025 backtest that means 2023-24 — both ONE-QB leagues. The offsets
    will therefore teach the simulator to take quarterbacks late, into a season
    that took five in three rounds. That is not a bug to be fixed by peeking; it
    is M5's central finding ("2023 and 2024 were one-QB leagues") measured from a
    second direction, and the gate reports it as such.
    """
    from ff_agent.draft import calibrate as CB

    managers = draft_order(season)
    p = pool if pool is not None else build_backtest_pool(season, len(managers))
    rounds = int(actual_picks(season)["round"].max())
    tilt, reach = _fit_lookups(season)
    prior = tuple(s for s in H.HISTORY_SEASONS if s < season)
    target = CB.target_rates(seasons=prior, target_two_qb=season in H.TWO_QB_SEASONS)
    return CB.fit_offsets(p, managers, tilt, reach, target, rounds, n_sims=n_sims)


def run(
    season: int = BACKTEST_SEASON,
    n_sims: int = 10_000,
    seed: int = 21,
    adp_sigma: float = PM.ADP_SIGMA,
    pool: PL.DraftPool | None = None,
    pos_offsets: np.ndarray | None = None,
) -> E.DraftResult:
    """Simulate the draft with every seat on the fitted model."""
    managers = draft_order(season)
    n_teams = len(managers)
    rounds = int(actual_picks(season)["round"].max())
    p = pool if pool is not None else build_backtest_pool(season, n_teams)
    tilt, reach = _fit_lookups(season)

    # my_index = -1 => no seat uses a policy
    return E.simulate_drafts(
        p, managers, my_index=-1, policy=None, tilt_lookup=tilt, reach=reach,
        n_sims=n_sims, rounds=rounds, seed=seed, adp_sigma=adp_sigma,
        pos_offsets=pos_offsets,
    )


# ── (a) positional composition by round ──────────────────────────────────────
def composition_by_round(res: E.DraftResult, season: int = BACKTEST_SEASON) -> pl.DataFrame:
    pos = np.array(res.pool.df["position"].to_list())
    n_teams, rounds = res.n_teams, res.rounds
    act = actual_picks(season)

    rows = []
    for r in range(rounds):
        cols = slice(r * n_teams, (r + 1) * n_teams)
        sim = pos[res.picks[:, cols]]
        a = act.filter(pl.col("round") == r + 1)
        n_a = max(a.height, 1)
        for p in PL.POSITIONS:
            rows.append({
                "round": r + 1,
                "position": p,
                "sim_share": round(float((sim == p).mean()), 4),
                "actual_share": round(
                    a.filter(pl.col("position") == p).height / n_a, 4
                ),
            })
    return (
        pl.DataFrame(rows)
        .with_columns((pl.col("sim_share") - pl.col("actual_share")).abs().alias("abs_err"))
        .sort(["round", "position"])
    )


def composition_summary(comp: pl.DataFrame) -> dict:
    return {
        "mean_abs_share_error": round(float(comp["abs_err"].mean()), 4),
        "max_abs_share_error": round(float(comp["abs_err"].max()), 4),
        "worst": comp.sort("abs_err", descending=True).head(3).to_dicts(),
    }


# ── (b) QB timing ────────────────────────────────────────────────────────────
def qb_timing(res: E.DraftResult, season: int = BACKTEST_SEASON) -> dict:
    """§3.1's structural risk, checked against what the league actually did."""
    pos = np.array(res.pool.df["position"].to_list())
    is_qb = pos[res.picks] == "QB"
    first = np.where(is_qb.any(axis=1), is_qb.argmax(axis=1) + 1, 0)
    three_rounds = res.n_teams * 3
    in_three = is_qb[:, :three_rounds].sum(axis=1)

    act = actual_picks(season)
    a_qb = act.filter(pl.col("position") == "QB")
    a_first = int(a_qb["overall_pick"].min()) if a_qb.height else 0
    a_three = int(a_qb.filter(pl.col("round") <= 3).height)
    return {
        "actual_first_qb_pick": a_first,
        "sim_first_qb_pick_median": int(np.median(first)),
        "sim_first_qb_pick_p10_p90": [int(np.percentile(first, 10)),
                                      int(np.percentile(first, 90))],
        "actual_first_qb_in_sim_interval": bool(
            np.percentile(first, 10) <= a_first <= np.percentile(first, 90)
        ),
        "actual_qbs_in_first_3_rounds": a_three,
        "sim_qbs_in_first_3_rounds_median": int(np.median(in_three)),
        "sim_qbs_in_first_3_rounds_p10_p90": [int(np.percentile(in_three, 10)),
                                              int(np.percentile(in_three, 90))],
        "actual_qb_count_in_sim_interval": bool(
            np.percentile(in_three, 10) <= a_three <= np.percentile(in_three, 90)
        ),
    }


# ── (c) calibration ──────────────────────────────────────────────────────────
def calibration(
    res: E.DraftResult,
    season: int = BACKTEST_SEASON,
    interval: float = 0.80,
    skill_only: bool = True,
) -> pl.DataFrame:
    """For each actual pick: where did the simulator think that player would go?

    Undrafted-in-sim counts as outside the interval — the alternative is to drop
    it, which would quietly grade the simulator only on the players it happened
    to find.
    """
    ids = res.pool.df["canonical_id"].to_list()
    index = {cid: i for i, cid in enumerate(ids)}
    act = actual_picks(season)
    if skill_only:
        act = act.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))

    lo_q = (1 - interval) / 2 * 100
    hi_q = 100 - lo_q
    total = res.picks.shape[1]

    rows = []
    for r in act.to_dicts():
        i = index.get(r["canonical_id"])
        if i is None:
            rows.append({**_row(r), "in_pool": False, "sim_median": None,
                         "sim_lo": None, "sim_hi": None, "inside": False,
                         "p_undrafted": None})
            continue
        got = res.pick_number_of(i)
        undrafted = float((got == 0).mean())
        seen = got[got > 0]
        if seen.size == 0:
            rows.append({**_row(r), "in_pool": True, "sim_median": None,
                         "sim_lo": None, "sim_hi": None, "inside": False,
                         "p_undrafted": round(undrafted, 4)})
            continue
        # undrafted sims sit past the end of the draft, not at zero
        filled = np.where(got == 0, total + 1, got)
        lo, med, hi = (float(np.percentile(filled, q)) for q in (lo_q, 50, hi_q))
        rows.append({
            **_row(r), "in_pool": True,
            "sim_median": int(med), "sim_lo": int(lo), "sim_hi": int(hi),
            "inside": bool(lo <= r["overall_pick"] <= hi),
            "p_undrafted": round(undrafted, 4),
        })
    return pl.DataFrame(rows)


def _row(r: dict) -> dict:
    return {
        "overall_pick": r["overall_pick"], "name": r["name"],
        "position": r["position"], "manager": r["manager"],
    }


def calibration_summary(
    cal: pl.DataFrame, interval: float = 0.80, total_picks: int | None = None
) -> dict:
    n = cal.height
    inside = int(cal["inside"].sum())
    width = cal.filter(pl.col("sim_hi").is_not_null()).with_columns(
        (pl.col("sim_hi") - pl.col("sim_lo")).alias("w")
    )
    w = int(width["w"].median()) if width.height else 0
    frac = w / total_picks if total_picks else None
    return {
        "n_picks": n,
        "nominal_interval": interval,
        "observed_coverage": round(inside / n, 4) if n else None,
        "in_pool": int(cal["in_pool"].sum()),
        "median_interval_width_picks": w or None,
        "median_interval_width_as_share_of_draft": (
            round(frac, 3) if frac is not None else None
        ),
        "verdict": _verdict(inside / n if n else 0.0, interval, w, frac),
    }


VACUOUS_WIDTH_SHARE = 0.40
"""An interval covering 40% of the draft says almost nothing about a player."""


def _verdict(coverage: float, interval: float, width: int, frac: float | None) -> str:
    if coverage < interval - 0.10:
        return ("UNDER-COVERED: the simulator is more confident than it has any "
                "right to be. Actual picks fall outside its intervals too often.")
    if frac is not None and frac > VACUOUS_WIDTH_SHARE:
        return (f"COVERED BUT VACUOUS: {coverage:.0%} of picks land inside an "
                f"interval {width} picks wide — {frac:.0%} of the whole draft. "
                "Coverage bought with vagueness is not a passing grade.")
    return (f"CALIBRATED: {coverage:.0%} coverage against a {interval:.0%} target, "
            f"on intervals {width} picks wide"
            + (f" ({frac:.0%} of the draft)." if frac is not None else "."))


# ── (d) §10 sanity alarms, and the league-level version of the same checks ───
def sanity_alarms(res: E.DraftResult, team: int | None = None) -> list[str]:
    """§10's alarms, for **one** roster.

    §10 lists these as alarms on *my* decisions — "a K before round 13", "only 1
    QB rostered after round 11". Running them across all eight simulated
    opponents, as an earlier version did, changes what they mean: it is not an
    alarm that a rival is thin at quarterback, it is information. Use
    ``league_diagnostics`` for the league-wide view.
    """
    t = res.my_index if team is None else team
    if t < 0:
        return []
    pos = np.array(res.pool.df["position"].to_list())
    cols = np.where(res.team_at_pick == t)[0]
    out = []

    early = cols[cols < 12 * res.n_teams]
    if early.size:
        share = float((pos[res.picks[:, early]] == "K").any(axis=1).mean())
        if share > 0.01:
            out.append(f"takes a kicker before round 13 in {share:.1%} of drafts")

    after_11 = cols[cols < 11 * res.n_teams]
    if after_11.size:
        n_qb = (pos[res.picks[:, after_11]] == "QB").sum(axis=1)
        share = float((n_qb <= 1).mean())
        if share > 0.05:
            out.append(
                f"holds one QB or fewer after round 11 in {share:.1%} of drafts"
            )
    return out


def league_diagnostics(res: E.DraftResult, season: int = BACKTEST_SEASON) -> pl.DataFrame:
    """League-wide behaviour, simulated vs what actually happened.

    Not alarms — evidence. Whether the simulated league drafts kickers and
    second quarterbacks on the same clock the real one does is part of "broad
    shape", and it is checkable against three real drafts.
    """
    pos = np.array(res.pool.df["position"].to_list())
    act = actual_picks(season)
    n_teams, rounds = res.n_teams, res.rounds
    rows = []

    r13 = 12 * n_teams
    sim_k = float((pos[res.picks[:, :r13]] == "K").sum(axis=1).mean())
    act_k = float(act.filter((pl.col("position") == "K") & (pl.col("round") <= 12)).height)
    rows.append({"measure": "kickers taken before round 13",
                 "simulated": round(sim_k, 2), "actual": act_k})

    r11 = 11 * n_teams
    thin = []
    for t in range(n_teams):
        cols = np.where(res.team_at_pick == t)[0]
        cols = cols[cols < r11]
        thin.append((pos[res.picks[:, cols]] == "QB").sum(axis=1) <= 1)
    sim_thin = float(np.stack(thin, axis=1).sum(axis=1).mean())
    a_thin = 0
    for m in act["manager"].unique().to_list():
        n = act.filter(
            (pl.col("manager") == m) & (pl.col("position") == "QB") & (pl.col("round") <= 11)
        ).height
        a_thin += int(n <= 1)
    rows.append({"measure": "teams on <=1 QB after round 11",
                 "simulated": round(sim_thin, 2), "actual": float(a_thin)})

    sim_dst = float((pos[res.picks[:, :r13]] == "DST").sum(axis=1).mean())
    act_dst = float(act.filter((pl.col("position") == "DST") & (pl.col("round") <= 12)).height)
    rows.append({"measure": "defenses taken before round 13",
                 "simulated": round(sim_dst, 2), "actual": act_dst})

    return pl.DataFrame(rows).with_columns(
        (pl.col("simulated") - pl.col("actual")).round(2).alias("diff")
    )


def sigma_sensitivity(
    season: int = BACKTEST_SEASON,
    sigmas: tuple[float, ...] = (0.02, 0.035, 0.055, 0.08, 0.12),
    n_sims: int = 2_000,
) -> pl.DataFrame:
    """Is the calibration a kernel-width story or a model-structure story?

    ``ADP_SIGMA`` was reasoned, not fitted — "roughly one round". Widening it
    buys coverage and pays in width, one for one, and that trade is all it does.
    Running the same sweep with the positional offsets in place separates the
    two: if offsets reach a coverage the width knob cannot reach at any width,
    they are adding structure rather than vagueness.
    """
    p = build_backtest_pool(season, len(draft_order(season)))
    strict, _ = fit_strict_offsets(season, pool=p)
    rows = []
    for label, off in (("no offsets", None), ("strict offsets", strict)):
        for s in sigmas:
            res = run(season, n_sims=n_sims, adp_sigma=s, pool=p, pos_offsets=off)
            cal = calibration(res, season)
            summ = calibration_summary(cal, total_picks=res.picks.shape[1])
            qb = qb_timing(res, season)
            rows.append({
                "arm": label,
                "adp_sigma": s,
                "coverage_80": summ["observed_coverage"],
                "width_picks": summ["median_interval_width_picks"],
                "width_share": summ["median_interval_width_as_share_of_draft"],
                "sim_first_qb": qb["sim_first_qb_pick_median"],
                "shipped_sigma": s == PM.ADP_SIGMA,
            })
    return pl.DataFrame(rows)


def _arm(res: E.DraftResult, season: int) -> dict:
    comp = composition_by_round(res, season)
    cal = calibration(res, season)
    return {
        "composition": composition_summary(comp),
        "qb_timing": qb_timing(res, season),
        "calibration": calibration_summary(cal, total_picks=res.picks.shape[1]),
        "league_diagnostics": league_diagnostics(res, season).to_dicts(),
    }


def in_sample_offsets(
    season: int = BACKTEST_SEASON,
    pool: PL.DraftPool | None = None,
    n_sims: int = 1_500,
) -> np.ndarray:
    """Offsets fitted on the TEST season itself. **Lookahead, on purpose.**

    This is M6's ceiling trick: if a simulator that has been told the answer
    still cannot reproduce the draft, the problem is mechanical. If it can, the
    gap between it and the strict arm measures exactly what the format change
    costs, rather than leaving that cost as an assertion.
    """
    from ff_agent.draft import calibrate as CB

    managers = draft_order(season)
    p = pool if pool is not None else build_backtest_pool(season, len(managers))
    rounds = int(actual_picks(season)["round"].max())
    tilt, reach = _fit_lookups(season)
    target = CB.target_rates(seasons=(season,), target_two_qb=season in H.TWO_QB_SEASONS)
    off, _ = CB.fit_offsets(p, managers, tilt, reach, target, rounds, n_sims=n_sims)
    return off


def report(season: int = BACKTEST_SEASON, n_sims: int = 10_000) -> dict:
    """The gate, in three arms. Only the middle one is the grade."""
    managers = draft_order(season)
    pool = build_backtest_pool(season, len(managers))
    strict, trace = fit_strict_offsets(season, pool=pool)

    arms = {
        "shipped_model_no_offsets": _arm(run(season, n_sims=n_sims, pool=pool), season),
        "strict_offsets_fit_on_prior_seasons": _arm(
            run(season, n_sims=n_sims, pool=pool, pos_offsets=strict), season
        ),
        "in_sample_offsets_LOOKAHEAD_ceiling": _arm(
            run(season, n_sims=n_sims, pool=pool,
                pos_offsets=in_sample_offsets(season, pool=pool)),
            season,
        ),
    }
    return {
        "season": season,
        "n_sims": n_sims,
        "n_teams": len(managers),
        "rounds": int(actual_picks(season)["round"].max()),
        "pool_size": pool.n,
        "actual_picks": actual_picks(season).height,
        "graded_arm": "strict_offsets_fit_on_prior_seasons",
        "offset_fit_trace": trace.to_dicts(),
        "arms": arms,
    }
