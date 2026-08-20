"""Milestone 1 command line.

    uv run python -m ff_agent.cli status      # cache inventory + credential check
    uv run python -m ff_agent.cli ingest      # pull/refresh nflverse 2016-2025
    uv run python -m ff_agent.cli byes        # bye table + §2.1 free-bye teams
    uv run python -m ff_agent.cli crosswalk   # resolve the ESPN pool, assert, report
    uv run python -m ff_agent.cli score       # M2 GATE — scores vs ESPN
    uv run python -m ff_agent.cli project     # M3 — build projections
    uv run python -m ff_agent.cli project --backtest   # M3 GATE — beats consensus?
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
    p = sub.add_parser("settings"); p.add_argument("--season", type=int); p.set_defaults(fn=cmd_settings)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("offline").set_defaults(fn=cmd_offline)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
