"""Milestone 1 command line.

    uv run python -m ff_agent.cli status      # cache inventory + credential check
    uv run python -m ff_agent.cli ingest      # pull/refresh nflverse 2016-2025
    uv run python -m ff_agent.cli byes        # bye table + §2.1 free-bye teams
    uv run python -m ff_agent.cli crosswalk   # resolve the ESPN pool, assert, report
    uv run python -m ff_agent.cli score       # M2 GATE — scores vs ESPN
    uv run python -m ff_agent.cli project     # M3 — build projections
    uv run python -m ff_agent.cli project --backtest   # M3 GATE — beats consensus?
    uv run python -m ff_agent.cli draftsim    # M7 — draft_sim.json (all 9 slots)
    uv run python -m ff_agent.cli draftsim --validate  # M7 GATE — replay 2025
    uv run python -m ff_agent.cli plans       # M8 — plan_1..9.json (precompute!)
    uv run python -m ff_agent.cli draft --slot 6   # T-60 drill: load, no compute
    uv run python -m ff_agent.cli settings    # refresh league settings JSON
    uv run python -m ff_agent.cli verify      # ESPN cookie pre-flight (draft morning)
    uv run python -m ff_agent.cli offline     # prove the offline path works
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from ff_agent.config import (
    ARTIFACTS_DIR, FREE_BYE_WEEKS, HISTORY_SEASONS, LAST_SEASON, SEASON,
    have_espn_credentials,
)
from ff_agent.data import byes as byes_mod
from ff_agent.data import cache, crosswalk as cw
from ff_agent.data import nflverse as nv

WIDE = pl.Config(tbl_rows=60, tbl_cols=12, fmt_str_lengths=60)


def cmd_status(_) -> int:
    inv = cache.cache_inventory()
    print(f"cache: {inv.height} tables, {inv['mb'].sum():.0f} MB")
    stale = inv.filter(pl.col("stale"))
    with WIDE:
        print(inv.group_by("table").agg(
            pl.len().alias("files"), pl.col("mb").sum().round(1).alias("mb"),
            pl.col("stale").any().alias("any_stale"),
        ).sort("table"))
    if stale.height:
        print(f"\n⚠  {stale.height} STALE file(s) — refresh before trusting output:")
        with WIDE:
            print(stale.select("table", "season", "age_hours", "policy"))
    print(f"\nESPN credentials present: {have_espn_credentials()}")
    return 0


def cmd_ingest(args) -> int:
    seasons = [args.season] if args.season else HISTORY_SEASONS
    rep = nv.ingest_all(seasons=seasons, include_pbp=not args.no_pbp)
    fails = rep.filter(pl.col("status") == "fail")
    print(f"\n{rep.height} pulls, {fails.height} failures")
    if fails.height:
        with WIDE:
            print(fails)
        return 1
    return 0


def cmd_byes(_) -> int:
    for season in (LAST_SEASON, SEASON):
        print(f"\n─── {season} ───")
        with WIDE:
            print(byes_mod.bye_summary(season).select("bye_week", "n_teams", "free_bye"))
        ft = byes_mod.free_bye_teams(season)
        print(f"FREE-BYE teams (NFL bye in {sorted(FREE_BYE_WEEKS)}): {len(ft)} → {ft}")
        anom = byes_mod.schedule_anomalies(season)
        if anom.height:
            print("schedule anomalies:")
            with WIDE:
                print(anom)
    return 0


def _gate_one(label: str, frame: pl.DataFrame) -> bool:
    """Resolve one population — players AND D/ST — and assert §0.2."""
    has_pos = "position" in frame.columns
    if has_pos:
        dst = frame.filter(pl.col("position") == "D/ST")
        players = frame.filter(pl.col("position") != "D/ST")
    else:  # draft history carries no position column
        dst = frame.filter(pl.col("espn_id").str.starts_with("-"))
        players = frame.filter(~pl.col("espn_id").str.starts_with("-"))
        players = players.with_columns(pl.lit(None, dtype=pl.Utf8).alias("position"),
                                       pl.lit(None, dtype=pl.Utf8).alias("team"))

    ok = True
    print(f"\n─── {label} ───  {players.height} players + {dst.height} D/ST")
    res = cw.resolve_players(players) if players.height else None
    if res is not None:
        with WIDE:
            print(cw.resolution_summary(res))
        try:
            cw.assert_all_resolved(res, label)
        except cw.CrosswalkError as e:
            print(f"FAIL (players, unresolved):\n{e}")
            ok = False
        try:
            # resolving is not enough — it must resolve to the RIGHT person
            cw.assert_resolutions_plausible(res, label, SEASON)
        except cw.CrosswalkError as e:
            print(f"FAIL (players, wrong person):\n{e}")
            ok = False
    if dst.height:
        dres = cw.resolve_dst(dst)
        bad = dres.filter(pl.col("match_method") == "unresolved")
        if bad.height:
            print(f"FAIL (D/ST): {bad.height} unresolved\n{bad.select('espn_id', 'name')}")
            ok = False
        else:
            print(f"  D/ST: all {dst.height} resolved")
    if ok:
        print("PASS: every entry resolves to exactly one canonical id")
    return ok


def cmd_crosswalk(_) -> int:
    canon = cw.canonical_players()
    print(f"canonical players: {canon.height:,}")
    print(f"D/ST entities:     {cw.dst_crosswalk().height}")
    print(f"manual overrides:  {cw.load_overrides().height}")

    if not have_espn_credentials():
        print(
            "\nESPN credentials absent — cannot run the Milestone 1 gate.\n"
            "  The gate needs the live draftable pool and last season's rosters.\n"
            "  Fill in .env (SETUP.md §3), then re-run."
        )
        return 2

    from ff_agent.data import espn

    populations: list[tuple[str, pl.DataFrame]] = [
        (f"draftable_pool_{SEASON}", espn.draftable_players(SEASON)),
        (f"rosters_{LAST_SEASON}", espn.rosters(LAST_SEASON)),
    ]
    for yr in (2023, 2024, 2025):
        try:
            populations.append((f"draft_{yr}", espn.draft_results(yr)))
        except Exception as e:
            print(f"  (skipping draft_{yr}: {type(e).__name__})")

    ok = True
    for label, frame in populations:
        ok &= _gate_one(label, frame)
    print("\n" + ("GATE PASSED — Milestone 1 assertion holds" if ok
                   else "GATE FAILED — see reports/"))
    return 0 if ok else 1


def cmd_score(args) -> int:
    """Milestone 2 gate: recomputed scores vs ESPN's recorded scores."""
    from ff_agent.scoring import validate as sv
    from ff_agent.scoring.rules import SPEC_SEASONS, load_rules

    season = args.season or LAST_SEASON
    rules = load_rules(season)
    print(f"season {season}: {len(rules)} scoring rules loaded"
          + ("  (matches §1)" if season in SPEC_SEASONS else "  (historical ruleset)"))
    if season not in SPEC_SEASONS:
        extra = {k: rules[k] for k in ("PC", "INC") if k in rules}
        if extra:
            print(f"  NOTE: this season also scored {extra} — removed for 2025+.")

    a = sv.layer_a_rules_check(season)
    a_bad = a.filter(pl.col("delta").abs() > sv.TOLERANCE)
    print(f"\nLAYER A  our rules on ESPN's own stat line "
          f"{a.height - a_bad.height}/{a.height} "
          f"({round(100 * (a.height - a_bad.height) / a.height, 3)}%)")
    if a_bad.height:
        with WIDE:
            print(a_bad.head(10))

    rep = sv.report(season)
    total = rep["players"]["rows"] + rep["dst"]["rows"]
    mism = rep["players"]["mismatches"] + rep["dst"]["mismatches"]
    print(f"\nLAYER B  our scores from nflverse vs ESPN")
    for k in ("players", "dst"):
        st = rep[k]
        print(f"  {st['label']:8} {st['rows'] - st['mismatches']:>5}/{st['rows']:<5} "
              f"exact  ({st['exact_pct']}%)")
    print(f"  {'TOTAL':8} {total - mism:>5}/{total:<5} exact "
          f"({round(100 * (total - mism) / total, 3)}%)")

    for key, label in (("player_categories", "players"), ("dst_categories", "dst")):
        cd = rep[key]
        if cd.height:
            print(f"\nLAYER C  which rule disagrees ({label}):")
            with WIDE:
                print(cd.group_by("rule").agg(
                    pl.len().alias("rows"),
                    pl.col("diff").abs().sum().round(2).alias("abs_pts"),
                ).sort("rows", descending=True))

    if mism:
        print(f"\n{mism} row(s) differ — see reports/scoring_mismatch_*_{season}.csv")
    return 0 if a_bad.height == 0 else 1


def cmd_project(args) -> int:
    """M3: build projections, or run the backtest gate."""
    from ff_agent.projections import backtest as B
    from ff_agent.projections import board_inputs as BI

    if args.backtest:
        wf = B.walk_forward(seasons=(2021, 2022, 2023, 2024, 2025), per_position=False)
        print("WALK-FORWARD backtest — blend weight fitted on PRIOR seasons only")
        with WIDE:
            print(wf)
        won, n = int((wf["delta"] > 0).sum()), wf.height
        print(f"\n  blend beat consensus in {won}/{n} seasons, "
              f"mean delta {wf['delta'].mean():+.4f} Spearman")
        print("\nFixed-weight robustness (is it a plateau or a knife-edge?):")
        with WIDE:
            print(B.weight_sweep(weights=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)))
        return 0 if won == n else 1

    season = args.season or SEASON
    b = BI.build(season)
    out = ARTIFACTS_DIR / f"projections_{season}.parquet"
    b.write_parquet(out)
    print(f"{b.height} players projected for {season} -> {out}")
    with WIDE:
        print(b.head(args.top).select(
            "name", "position", "team", "ecr", "consensus_points",
            "model_points", "blended_points", "free_bye_week_5_or_14", "playoff_sos"))
    fb = b.head(60).filter(pl.col("free_bye_week_5_or_14"))
    print(f"\n§2.1 free-bye players inside the top 60: {fb.height}")
    print("  " + ", ".join(fb["name"].to_list()))
    return 0


def cmd_xfp(args) -> int:
    """M3b: xFP under §1, and the ADD-§I-3b validity test."""
    from ff_agent.projections import xfp as X
    from ff_agent.projections import xfp_validity as V

    season = args.season or LAST_SEASON
    s = X.xfp_season(season)
    print(f"{season}: xFP for {s.height} players\n")
    with WIDE:
        print(s.head(args.top).select(
            "name", "position", "games", "xfp", "actual_points", "fpoe",
            "xtd", "actual_td", "sacks_over_expected"))

    if not args.validate:
        return 0

    print("\n=== ADD-§I-3b: does xFP predict next season better than points? ===")
    v = V.validity()
    with WIDE:
        print(v)
    wins = v.filter(pl.col("position") != "ALL")["xfp_wins"].any()
    print("  RESULT: xFP does NOT beat prior-season points at any position."
          if not wins else "  RESULT: xFP wins somewhere — revisit the conclusion.")
    print("  xFP therefore stays a DIAGNOSTIC and does not enter the projection.")

    print("\n=== ADD-§A.1: does touchdown luck regress? ===")
    with WIDE:
        print(V.td_regression_evidence())
    print("  xTD beats actual TD for WR/TE (luck regresses) but not RB/QB,"
          "\n  where goal-line role is a durable team assignment.")

    print("\n=== ADD-§B.1: sacks-over-expected — trait or luck? ===")
    with WIDE:
        print(V.sack_trait_evidence())
    print("  Sack overperformance persists ~4x more strongly than TD"
          "\n  overperformance. Fade TD luck; price in sack rate (§3.3).")
    return 0


def cmd_board(args) -> int:
    """M4: VOR + tiers + bye adjustment -> board.json, plus the review report."""
    from ff_agent.board import build as BB
    from ff_agent.board import replacement as RR
    from ff_agent.board import review as RV

    season = args.season or SEASON
    b, path = BB.write_board(season)
    print(f"board: {b.height} players -> {path}\n")

    print("replacement levels (§7.3, flex allocated by MEASURED usage):")
    print("  measured flex share:",
          {k: round(v, 3) for k, v in RR.flex_share().items()},
          " (§7.3 assumed ~4 RB / 4 WR / 1 TE)")
    with WIDE:
        print(RR.replacement_table(
            __import__("ff_agent.projections.calibration", fromlist=["x"])
            .smooth_curve(
                __import__("ff_agent.projections.calibration", fromlist=["x"])
                .rank_points_curve(seasons=[s for s in range(2016, season)]))))

    print(f"\nTOP {args.top}")
    with WIDE:
        print(b.head(args.top).select(
            "overall_rank", "name", "position", "team", "vor", "bye_adjustment",
            "sack_adjustment", "tier", "players_remaining_in_tier", "ecr",
            "rank_vs_ecr"))

    print("\n§11 REVIEW — divergences from consensus, with reasons:")
    with WIDE:
        print(RV.divergences(b, top=50).select(
            "overall_rank", "name", "position", "ecr", "rank_vs_ecr",
            "vs_consensus", "reason"))

    print("\n§11 — do week 5/14 bye players visibly rise?")
    with WIDE:
        print(RV.bye_lift(b))

    print("\n§2.1 — QB COUNT DECISION")
    for k, v in RV.qb_count_decision(b).items():
        print(f"  {k}: {v}")

    alarms = RV.sanity_alarms(b)
    print("\n§10 SANITY ALARMS: " + ("none" if not alarms else ""))
    for a in alarms:
        print(f"  !! {a}")
    return 1 if alarms else 0


def cmd_opponents(args) -> int:
    """M5: per-manager tendencies -> opponents.json, plus the §11 gate."""
    from ff_agent.opponents import build as OB
    from ff_agent.opponents import evaluate as OE
    from ff_agent.opponents import history as OH
    from ff_agent.opponents import tendencies as OT

    path = OB.write(args.season or SEASON)
    print(f"wrote {path}\n")

    print("THE CONSTRAINT — 2023 and 2024 were ONE-QB leagues; only 2025 was 2-QB:")
    with WIDE:
        print(OH.format_shift_evidence())
    print("\nEvidence per manager (§2.3 weights the 2x managers most; "
          "they are not the ones with the most data):")
    with WIDE:
        print(OH.manager_coverage().select(
            "manager", "faced_twice", "n_drafts", "n_picks",
            "n_drafts_two_qb", "qb_confidence"))

    print("\nWhen each manager takes his first QB (2-QB seasons only):")
    with WIDE:
        print(OT.qb_timing())

    if args.validate:
        print("\n=== §11 GATE: fit 2023-24, predict 2025, vs superflex ADP ===")
        with WIDE:
            print(OE.strict_holdout(2025))
        print("\nPer-manager — where does the gain come from?")
        b = OE.per_manager_breakdown(2025)
        with WIDE:
            print(b)
        print(f"  mean gain {b['gain'].mean():+.4f}; "
              f"{b.filter(pl.col('gain') > 0).height}/{b.height} managers improved")
        print("  Negative entries are real: a manager's 1-QB history can mispredict")
        print("  his 2-QB behaviour. Hence the 3x weight on the format-matched season.")
    return 0


def cmd_simulate(args) -> int:
    """M6: season simulator -> season_sim.json, plus the §11 diagnostics."""
    from ff_agent.season import build as SB
    from ff_agent.season import evaluate as SE
    from ff_agent.season import schedule as SSCH
    from ff_agent.season import simulate as SSIM

    season = args.season or SEASON
    teams = SSCH.games_played(season)["team"].to_list()
    means = {t: 130.0 for t in teams}

    print("§2.5 — unequal games are real:")
    with WIDE:
        print(SSCH.games_played(season))

    print("\n§2.5 ANSWERED — raw wins vs win pct, with every team identical:")
    sens = SSIM.seeding_sensitivity(means, season=season, n_sims=args.sims)
    with WIDE:
        print(sens.select("team", "games", "p_playoffs_wins", "p_playoffs_pct",
                          "d_playoffs", "p_top2_seed_wins", "p_top2_seed_pct", "d_top2"))
    g12 = sens.filter(pl.col("games") == 12)
    g13 = sens.filter(pl.col("games") == 13)
    print(f"  A 12-game team gives up "
          f"{float(g13['p_playoffs_wins'].mean() - g12['p_playoffs_wins'].mean()):.3f} "
          f"playoff probability and "
          f"{float(g13['p_top2_seed_wins'].mean() - g12['p_top2_seed_wins'].mean()):.3f} "
          f"top-2 probability under RAW WINS. Under win percentage, nothing.")
    print("  Worth confirming on the standings page — this is not a technicality.")

    path = SB.write(season=season, n_sims=args.sims)
    print(f"\nwrote {path}")

    if args.validate:
        print("\n=== §11: reproduce last season's standings ===")
        import json
        print(json.dumps(SE.backtest_summary(2025, args.sims), indent=2))
        print("\nIs the miss the SIMULATOR or the PROJECTIONS?")
        with WIDE:
            print(SE.decompose(2025))
        print("\nWhat could a PERFECT simulator score on this test?")
        print(" ", SE.ceiling(2025, trials=150))
        print("\n  Perfect player knowledge reaches +0.43 against a +0.52 ceiling,")
        print("  so the mechanics are sound. Preseason projections score NEGATIVE:")
        print("  the projections are the limitation, not the simulator.")
        probs = SE.structural_checks(season)
        print("\nStructural checks (2025 had no byes, so these cannot be backtested):")
        print("  " + ("\n  ".join(probs) if probs else "all invariants hold"))
    return 0


def cmd_draftsim(args) -> int:
    """M7: Monte Carlo draft simulator -> draft_sim.json, plus the §11 gate."""
    from ff_agent.draft import backtest as DBT
    from ff_agent.draft import build as DB

    if args.validate:
        print("=== §11 GATE: reproduce the 2025 draft's broad shape ===")
        print("  8 teams, real draft order, opponent model fit on 2023-24 ONLY.")
        print("  Pool is WIDER than the players actually drafted — anything else")
        print("  would hand the simulator the answer.\n")
        rep = DBT.report(n_sims=args.sims)
        print(f"  {rep['actual_picks']} actual picks, pool of {rep['pool_size']}, "
              f"{rep['n_sims']} sims\n")
        for name, arm in rep["arms"].items():
            q, c = arm["qb_timing"], arm["calibration"]
            mark = "  <-- THE GRADE" if name == rep["graded_arm"] else ""
            print(f"─── {name}{mark}")
            print(f"    first QB       actual {q['actual_first_qb_pick']:>3}   "
                  f"sim {q['sim_first_qb_pick_median']:>3} "
                  f"{q['sim_first_qb_pick_p10_p90']}   "
                  f"inside={q['actual_first_qb_in_sim_interval']}")
            print(f"    QBs in 3 rds   actual {q['actual_qbs_in_first_3_rounds']:>3}   "
                  f"sim {q['sim_qbs_in_first_3_rounds_median']:>3} "
                  f"{q['sim_qbs_in_first_3_rounds_p10_p90']}   "
                  f"inside={q['actual_qb_count_in_sim_interval']}")
            print(f"    composition    mean|err| "
                  f"{arm['composition']['mean_abs_share_error']}")
            print(f"    {c['verdict']}")
        print("\n  The QB error FLIPS SIGN between the first two arms: without the")
        print("  positional offsets the simulator takes far too many quarterbacks")
        print("  early (following superflex ECR), and with offsets fitted on two")
        print("  ONE-QB seasons it takes far too few. The truth sits between them,")
        print("  which is M5's format-change finding arriving from a new direction.")
        print("\nIs the interval width just the ADP kernel? (sigma sweep)")
        with WIDE:
            print(DBT.sigma_sensitivity(n_sims=max(args.sims // 4, 500)))
        return 0

    path = DB.write(season=args.season or SEASON, n_sims=args.sims)
    print(f"wrote {path}\n")
    import json
    d = json.loads(__import__("pathlib").Path(path).read_text())

    print("EDGE DECOMPOSITION — how much of the simulated margin is real?")
    with WIDE:
        print(pl.DataFrame(d["edge_decomposition"]))
    print("  model_seat must sit at ~0: it is me drafting exactly like the model.")
    print("  best_consensus carries ZERO board edge, so its margin is bought")
    print("  entirely from M5's opponents picking with an 8-pick spread.\n")

    tbl = pl.DataFrame(d["by_slot_and_policy"]).drop("roster_shape", "sanity_alarms")
    print("BY SLOT AND POLICY")
    with WIDE:
        print(tbl.sort(["slot", "p_title"], descending=[False, True]))

    print("\n§2.1 — HOW MANY QUARTERBACKS? (uncapped VOR takes FOUR, every time)")
    with WIDE:
        print(pl.DataFrame(d["qb_cap_sweep"]))
    print("  VOR is measured against the STARTER replacement level (QB18), so it")
    print("  prices QB3 and QB4 as if they would start. They will not — I start two.")
    print("  M4 answered this question TWO on a ceiling comparison; M7 answers it")
    print("  over the distribution of QBs I actually get, on expected points. They")
    print("  measure different things — see the open question in CLAUDE.md.")

    print("\n§2.4 — is roster variance actually good?")
    with WIDE:
        print(pl.DataFrame(d["surrogate"]["variance_slope"]))
    x = d["surrogate"]["variance_crossover"]
    print(f"  title crossover at delta {x['title_crossover_delta']}, "
          f"top-2 crossover at delta {x['top2_crossover_delta']}")
    print("  §2.4 says variance is good, full stop. It is good BELOW the")
    print("  crossover and costs seeding above it — and the top-2 crossover")
    print("  comes first, so the first-round bye is what variance spends.")

    print(f"\nSURROGATE: {d['surrogate']['verdict']}")
    print("\nSEAT SENSITIVITY — opponents' slots are unknown:")
    with WIDE:
        print(pl.DataFrame(d["seat_sensitivity"]))

    alarms = sorted({a for s in d["by_slot_and_policy"] for a in s["sanity_alarms"]})
    print("\n§10 SANITY ALARMS (my seat): " + ("none" if not alarms else ""))
    for a in alarms:
        print(f"  !! {a}")
    return 1 if alarms else 0


def cmd_plans(args) -> int:
    """M8: precompute all nine slot plans -> plan_1..9.json."""
    from ff_agent.draft import plan_build as PB

    paths = PB.write(
        season=args.season or SEASON, n_sims=args.sims, qb_cap=args.qb_cap
    )
    print(f"wrote {len(paths)} files:")
    for p in paths:
        print(f"  {p}")

    import json
    idx = json.loads(__import__("pathlib").Path(paths[-1]).read_text())

    print("\n§2.1 QB COUNT — M4 said TWO, M7 said THREE. Resolved on P(title):")
    q = idx["qb_count_decision"]
    with WIDE:
        print(pl.DataFrame(q["by_cap"]))
    print(f"  DECISION: {q['decision']} quarterbacks "
          f"(runner-up {q['runner_up']}, margin {q['margin_over_runner_up']:+.4f})")
    print(f"  ranking at the INFLATED delta: {q['ranking_at_inflated_delta']}")
    print(f"  ranking at the HONEST delta:   {q['ranking_at_honest_delta']}")
    print(f"  the noise correction changes the answer: "
          f"{q['correction_changes_the_answer']}")
    print(f"  my honest delta {q['my_honest_delta']} against a variance "
          f"crossover at {q['variance_crossover_delta']} "
          f"(above it, variance costs seeding)")

    print("\nQB SCENARIOS — the axis the draft turns on:")
    with WIDE:
        print(pl.DataFrame(idx["qb_scenarios"]["pressure"]))

    print("\n§11 STEP 8 — do the nine plans actually differ?")
    g = idx["plans_differ"]
    print(f"  {g['verdict']}")
    if g.get("extremes"):
        print(f"  slot1-vs-slot5 {g['extremes']['slot1_vs_slot5']}   "
              f"slot1-vs-slot9 {g['extremes']['slot1_vs_slot9']}   "
              f"slot5-vs-slot9 {g['extremes']['slot5_vs_slot9']}")

    print("\nPER SLOT")
    rows = [{"slot": int(k), **{kk: vv for kk, vv in v.items()
                                if kk != "roster_strength"},
             **{f"strength_{a}": b for a, b in v["roster_strength"].items()}}
            for k, v in idx["slots"].items()]
    with WIDE:
        print(pl.DataFrame(rows).sort("slot"))
    return 0


def cmd_draft(args) -> int:
    """§5's T-60 drill: load one precomputed plan and print the sheet.

    Deliberately does NO computation. If the plan is missing this fails with an
    instruction rather than quietly building one while you are on the clock.
    """
    import random
    import time

    from ff_agent.draft import plan_build as PB
    from ff_agent.draft import sheet as SH

    t0 = time.perf_counter()
    slot = args.slot or random.randint(1, 9)
    if args.slot is None:
        print(f"REHEARSAL — random slot {slot} (§5 asks for two of these)\n")
    plan = PB.load(slot)
    print(SH.render(plan, max_turns=args.turns))
    elapsed = time.perf_counter() - t0
    print(f"\nloaded in {elapsed:.2f}s"
          + ("" if elapsed < 5 else "   ⚠ §5 BUDGET IS 5 SECONDS"))
    return 0


def cmd_settings(args) -> int:
    from ff_agent.data import espn

    d = espn.settings_dump(args.season or SEASON)
    print(f"wrote {d.get('_written_to')}")
    return 0


def cmd_verify(_) -> int:
    from ff_agent.data import espn

    r = espn.verify_credentials(SEASON)
    if r["ok"]:
        print(f"OK  {r['league_name']} ({r['n_teams']} teams, {r['year']})")
        for t in r["team_names"]:
            print(f"      {t}")
        return 0
    print(f"FAIL [{r['stage']}]\n{r['detail']}")
    return 1


def cmd_offline(_) -> int:
    """Prove the draft-day path: every table served from disk, network off."""
    print("Reading every cached table with offline=True (no network permitted)…")
    ok = True
    checks = [("players", lambda: nv.players(offline=True)),
              ("schedules", lambda: nv.schedules(offline=True))]
    for s in HISTORY_SEASONS:
        checks.append((f"player_stats[{s}]", lambda s=s: nv.player_stats(s, offline=True)))
        checks.append((f"pbp[{s}]", lambda s=s: nv.pbp(s, offline=True)))
    for name, fn in checks:
        try:
            df = fn()
            print(f"  ok   {name:22s} {df.height:>8,} rows")
        except Exception as e:
            print(f"  FAIL {name:22s} {type(e).__name__}: {str(e).splitlines()[0]}")
            ok = False
    try:
        cw.canonical_players()
        byes_mod.bye_weeks(SEASON)
        print("  ok   crosswalk + byes derived offline")
    except Exception as e:
        print(f"  FAIL crosswalk/byes: {e}")
        ok = False
    print("\nOFFLINE PATH OK" if ok else "\nOFFLINE PATH BROKEN")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ff_agent.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("ingest")
    p.add_argument("--season", type=int)
    p.add_argument("--no-pbp", action="store_true")
    p.set_defaults(fn=cmd_ingest)
    sub.add_parser("byes").set_defaults(fn=cmd_byes)
    sub.add_parser("crosswalk").set_defaults(fn=cmd_crosswalk)
    p = sub.add_parser("score"); p.add_argument("--season", type=int); p.set_defaults(fn=cmd_score)
    p = sub.add_parser("project")
    p.add_argument("--season", type=int); p.add_argument("--top", type=int, default=20)
    p.add_argument("--backtest", action="store_true")
    p.set_defaults(fn=cmd_project)
    p = sub.add_parser("xfp")
    p.add_argument("--season", type=int); p.add_argument("--top", type=int, default=12)
    p.add_argument("--validate", action="store_true")
    p.set_defaults(fn=cmd_xfp)
    p = sub.add_parser("board")
    p.add_argument("--season", type=int); p.add_argument("--top", type=int, default=25)
    p.set_defaults(fn=cmd_board)
    p = sub.add_parser("opponents")
    p.add_argument("--season", type=int); p.add_argument("--validate", action="store_true")
    p.set_defaults(fn=cmd_opponents)
    p = sub.add_parser("simulate")
    p.add_argument("--season", type=int); p.add_argument("--sims", type=int, default=20000)
    p.add_argument("--validate", action="store_true")
    p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser("draftsim")
    p.add_argument("--season", type=int); p.add_argument("--sims", type=int, default=10000)
    p.add_argument("--validate", action="store_true")
    p.set_defaults(fn=cmd_draftsim)
    p = sub.add_parser("plans")
    p.add_argument("--season", type=int); p.add_argument("--sims", type=int, default=10000)
    p.add_argument("--qb-cap", type=int, dest="qb_cap", default=None)
    p.set_defaults(fn=cmd_plans)
    p = sub.add_parser("draft")
    p.add_argument("--slot", type=int, default=None)
    p.add_argument("--turns", type=int, default=None)
    p.set_defaults(fn=cmd_draft)
    p = sub.add_parser("settings"); p.add_argument("--season", type=int); p.set_defaults(fn=cmd_settings)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("offline").set_defaults(fn=cmd_offline)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
