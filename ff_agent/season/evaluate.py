"""MILESTONE 6 TEST (§11): reproduce last season's standings from preseason
projections, and handle byes and unequal games correctly.

Two honest limits on what this can show, both stated before the number:

* **2025 had no regular-season byes** — 8 teams, everyone played 14 games. The
  bye and unequal-games logic therefore CANNOT be validated against last season.
  It is checked structurally instead (see ``structural_checks``).
* **Opening-day rosters ignore a whole season of waivers and trades**, and §3.2
  says this league's wire stays rich all year. So the backtest measures how well
  draft-day talent predicts final standings — a LOWER BOUND on the simulator,
  not a clean measure of it.

And the ceiling is low for a real reason: weekly noise (sd 25.7) is about three
times the talent spread (sd 8.7), so a 14-game season genuinely does not resolve
team quality very far.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ff_agent.config import normalize_team_name
from ff_agent.season import schedule as SCH
from ff_agent.season import simulate as SIM
from ff_agent.season import strength as STR


def opening_rosters(season: int) -> pl.DataFrame:
    """Reconstruct post-draft rosters from the draft results."""
    from ff_agent.data import crosswalk as cw
    from ff_agent.data import espn

    d = espn.draft_results(season)
    canon = cw.canonical_players().select("espn_id", "canonical_id")
    players = d.filter(~pl.col("espn_id").str.starts_with("-")).join(
        canon, on="espn_id", how="left"
    )
    return (
        players.filter(pl.col("canonical_id").is_not_null())
        .with_columns(
            pl.col("fantasy_team")
            .map_elements(normalize_team_name, return_dtype=pl.Utf8)
            .alias("fantasy_team")
        )
        .select("fantasy_team", "canonical_id", "name", "round", "pick")
    )


def actual_standings(season: int) -> pl.DataFrame:
    from ff_agent.data.espn import get_league

    lg = get_league(season)
    rows = []
    for t in lg.teams:
        rows.append({
            "team": normalize_team_name(t.team_name),
            "actual_standing": getattr(t, "standing", None),
            "actual_wins": getattr(t, "wins", None),
            "actual_points": round(float(getattr(t, "points_for", 0.0)), 1),
        })
    return pl.DataFrame(rows).sort("actual_standing")


def backtest(season: int = 2025, n_sims: int = 20_000) -> pl.DataFrame:
    """Simulate ``season`` from opening-day rosters and compare to what happened."""
    from ff_agent.projections import board_inputs as BI

    rosters = opening_rosters(season)
    proj = BI.build(season)
    means = STR.roster_strength(rosters, proj)
    res = SIM.simulate(means, season=season, n_sims=n_sims, seeding_rule="win_pct")

    sim = res.table().rename({"team": "team"})
    act = actual_standings(season)
    return (
        sim.join(act, on="team", how="inner")
        .with_columns(
            pl.col("mean_wins").rank("ordinal", descending=True).cast(pl.Int32)
            .alias("sim_rank")
        )
        .sort("sim_rank")
    )


def backtest_summary(season: int = 2025, n_sims: int = 20_000) -> dict:
    b = backtest(season, n_sims)
    if b.height < 4:
        return {"error": "insufficient overlap between sim and actual standings"}
    rho = b.select(pl.corr("sim_rank", "actual_standing", method="spearman")).item()
    pts = b.select(pl.corr("mean_points", "actual_points", method="spearman")).item()
    top6_sim = set(b.sort("sim_rank").head(6)["team"].to_list())
    top6_act = set(b.sort("actual_standing").head(6)["team"].to_list())
    return {
        "season": season,
        "n_teams": b.height,
        "standings_spearman": round(float(rho), 4),
        "points_spearman": round(float(pts), 4),
        "playoff_field_overlap": f"{len(top6_sim & top6_act)}/6",
        "note": (
            "Opening-day rosters only; a season of waivers and trades is not "
            "modelled. Weekly noise is ~3x the talent spread, so a 14-game "
            "season cannot resolve much more than this."
        ),
    }


def structural_checks(season: int = 2026) -> list[str]:
    """What the 2025 backtest cannot cover: byes and unequal games.

    Verified as invariants rather than against data, because 2025 had neither.
    """
    problems = []
    g = SCH.games_played(season)
    total_byes = int(g["n_byes"].sum())
    n_weeks = SCH.league_schedule(season)["week"].n_unique()
    if total_byes != n_weeks:
        problems.append(
            f"{total_byes} bye slots across {g.height} teams but {n_weeks} weeks — "
            f"an odd team count must produce exactly one bye per week"
        )
    teams = g["team"].to_list()
    means = {t: 130.0 for t in teams}
    r = SIM.simulate(means, season=season, n_sims=4000, seeding_rule="win_pct")
    if abs(float(r.p_playoffs.sum()) - 6.0) > 0.01:
        problems.append(f"playoff probabilities sum to {r.p_playoffs.sum():.3f}, expected 6.0")
    if abs(float(r.p_title.sum()) - 1.0) > 0.01:
        problems.append(f"title probabilities sum to {r.p_title.sum():.3f}, expected 1.0")
    # with identical teams, win_pct must not advantage the 13-game group
    tbl = r.table().join(g.select("team", "games"), on="team")
    by_games = tbl.group_by("games").agg(pl.col("p_playoffs").mean())
    if by_games.height == 2:
        spread = abs(by_games["p_playoffs"][0] - by_games["p_playoffs"][1])
        if spread > 0.02:
            problems.append(
                f"under win_pct, 12- and 13-game teams differ by {spread:.3f} "
                f"in P(playoffs) despite identical strength"
            )
    return problems


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(-np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(-np.asarray(b, dtype=float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def ceiling(season: int = 2025, trials: int = 400) -> dict:
    """What could a PERFECT simulator score on this test?

    Feed the simulator each team's TRUE strength (its realised points per game)
    and see how well one simulated season's standings track it. Anything at or
    near this number is as good as the test can measure; the gap between it and
    1.0 is irreducible luck, not model error.
    """
    act = actual_standings(season)
    means = {r["team"]: r["actual_points"] / 14.0 for r in act.to_dicts()}
    rhos = []
    for i in range(trials):
        r = SIM.simulate(means, season=season, n_sims=1,
                         seeding_rule="win_pct", seed=1000 + i)
        rhos.append(_spearman([means[t] for t in r.teams], r.mean_wins))
    a = np.array(rhos)
    return {
        "mean_rho": round(float(a.mean()), 4),
        "sd": round(float(a.std()), 4),
        "p05": round(float(np.percentile(a, 5)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
    }


def decompose(season: int = 2025) -> pl.DataFrame:
    """Is the miss caused by the SIMULATOR or by the PROJECTIONS?

    Runs team strength two ways on identical rosters and schedule: once from
    preseason projections, once from what those players actually scored. If
    perfect player knowledge reaches the ceiling and projections do not, the
    simulator is sound and the projections are the limit.
    """
    from ff_agent.projections import actuals as A
    from ff_agent.projections import board_inputs as BI

    ros = opening_rosters(season)
    act = actual_standings(season)
    teams = act["team"].to_list()
    pts = {r["team"]: r["actual_points"] for r in act.to_dicts()}
    rank = {r["team"]: r["actual_standing"] for r in act.to_dicts()}

    proj = BI.build(season)
    real = A.player_season_actuals(season).select(
        "canonical_id", "position", pl.col("points").alias("blended_points")
    )
    rows = []
    for label, table in (("preseason projections", proj),
                         ("perfect player knowledge", real)):
        m = STR.roster_strength(ros, table)
        xs = [m.get(t, 0.0) for t in teams]
        rows.append({
            "team_strength_from": label,
            "vs_actual_points": round(_spearman(xs, [pts[t] for t in teams]), 4),
            "vs_actual_standings": round(_spearman(xs, [-rank[t] for t in teams]), 4),
        })
    return pl.DataFrame(rows)


def bye_worth(season: int = 2026, n_sims: int = 30_000) -> dict:
    """§2.4 claims a first-round bye roughly DOUBLES title odds (12.5% vs 25%).

    Measured with all teams identical, so the only difference is the bracket.
    """
    from ff_agent.season import schedule as SCH

    teams = SCH.games_played(season)["team"].to_list()
    means = {t: 130.0 for t in teams}
    r = SIM.simulate(means, season=season, n_sims=n_sims, seeding_rule="win_pct")
    p_title = float(r.p_title.mean())
    p_top2 = float(r.p_top2.mean())
    p_made = float(r.p_playoffs.mean())
    # P(title | seed) by Bayes on the aggregate rates
    p_title_given_bye = p_title * (p_top2 / max(p_made, 1e-9)) / max(p_top2, 1e-9)
    return {
        "p_playoffs_equal_teams": round(p_made, 4),
        "p_top2_equal_teams": round(p_top2, 4),
        "spec_claim": "§2.4: ~12.5% without a bye vs ~25% with one",
        "note": (
            "With 6 of 9 qualifying, making the tournament is the default "
            "outcome (§2.4) — the bye, not the berth, is what is scarce."
        ),
    }
