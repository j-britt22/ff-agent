"""Live ESPN facts that §1 marked CONFIRM, now verified.

These are cheap regression guards: if a commissioner changes a setting mid-season,
the season simulator's objective function silently becomes wrong (§8). Better to
fail a test.
"""
import polars as pl
import pytest

from ff_agent import config as c


pytestmark = pytest.mark.espn


@pytest.fixture(scope="module")
def settings():
    from ff_agent.data import espn
    return espn.settings_dump(c.SEASON, write=False)


def test_nine_teams_fourteen_week_regular_season(settings):
    assert settings["team_count"] == c.N_TEAMS == 9
    assert settings["reg_season_count"] == 14


def test_six_playoff_teams_three_one_week_rounds(settings):
    # §2.4's whole objective function rests on this shape.
    assert settings["playoff_team_count"] == c.PLAYOFF_TEAMS == 6
    assert settings["playoff_matchup_period_length"] == 1
    assert max(int(k) for k in settings["matchup_periods"]) == 17
    assert c.PLAYOFF_WEEKS == (15, 16, 17)


def test_full_redraft_not_a_keeper_league(settings):
    assert settings["keeper_count"] == 0
    assert c.IS_KEEPER_LEAGUE is False


def test_rolling_priority_waivers_not_faab(settings):
    # §9.3's entire model assumes priority is indivisible, not a budget.
    assert settings["faab"] is False


def test_roster_slots_match_spec(settings):
    slots = settings["position_slot_counts"]
    assert slots["QB"] == 2 and slots["RB"] == 2 and slots["WR"] == 2
    assert slots["TE"] == 1 and slots["D/ST"] == 1 and slots["K"] == 1
    assert slots["RB/WR/TE"] == 1          # the FLEX
    assert slots["BE"] == c.BENCH_SLOTS == 7
    assert slots["IR"] == c.IR_SLOTS == 1
    # every IDP / exotic slot must be zero, per §1
    for z in ("TQB", "RB/WR", "WR/TE", "OP", "DT", "DE", "LB", "DL",
              "CB", "S", "DB", "DP", "P", "HC"):
        assert slots[z] == 0, f"{z} is not zero — §1 assumed no IDP"


def _rule(settings, abbr):
    for x in settings["scoring_format"]:
        if x["abbr"] == abbr:
            return x["points"]
    return 0.0


def test_the_two_unusual_scoring_rules_are_real(settings):
    """§3.3 and §3.4 — the entire edge. If these are not set, the premise dies."""
    assert _rule(settings, "SKD") == -1.0   # sack taken
    assert _rule(settings, "RA") == 0.05    # per rushing attempt


def test_six_point_passing_tds_and_half_ppr(settings):
    assert _rule(settings, "PTD") == 6.0
    assert _rule(settings, "REC") == 0.5
    assert _rule(settings, "PY") == 0.04
    assert _rule(settings, "INTT") == -2.0


def test_distance_weighted_field_goals(settings):
    assert _rule(settings, "FG0") == 3.0
    assert _rule(settings, "FG40") == 4.0
    assert _rule(settings, "FG50") == 5.0
    assert _rule(settings, "FG60") == 6.0
    assert _rule(settings, "FGM") == -1.0


def test_dst_middle_buckets_really_do_score_zero(settings):
    """§1 listed 18-21 AND 22-27 both at 0, which looked like a transcription
    slip because ESPN's default for 22-27 is -1. ESPN omits zero-valued rules
    entirely, and both are absent — so §1 was right."""
    labels = {x["label"] for x in settings["scoring_format"]}
    assert "18-21 points allowed" not in labels
    assert "22-27 points allowed" not in labels
    assert "300-349 total yards allowed" not in labels
    # the non-zero neighbours must still be present and correct
    assert _rule(settings, "PA14") == 1.0
    assert _rule(settings, "PA28") == -1.0
    assert _rule(settings, "YA399") == -1.0


def test_my_schedule_matches_the_spec():
    """12 games, byes in weeks 5 and 14, four double-up opponents (§2.1, §2.3).

    Keyed on MANAGER, not on team name. Team names are editable and one of them
    already changed: §1 records "A Chane Reaction" for Jeff Boyd and that team is
    now called "TBD". The structural facts §2.1 and §2.3 depend on — how many
    games, which weeks are free, who I face twice — are properties of the
    schedule, not of what somebody called their team this morning.

    A rename is reported, because a stale §1 is worth knowing about; it is not a
    failure, because nothing downstream depends on the string any more.
    """
    from ff_agent.data.espn import get_league, team_names_by_manager

    lg = get_league(c.SEASON)
    me = [t for t in lg.teams
          if c.normalize_team_name(t.team_name) == c.MY_TEAM_NAME][0]
    sched = list(getattr(me, "schedule", []) or [])
    assert len(sched) == 14

    live = team_names_by_manager(c.SEASON)
    mgr_of = dict(zip(live["team"].to_list(), live["manager"].to_list()))
    expected_mgr = {
        w: __import__("ff_agent.opponents.history", fromlist=["x"])
        .canonical_manager(o["manager"])
        for o in c.OPPONENTS for w in o["weeks"]
    }

    byes, renames = [], []
    for wk, opp in enumerate(sched, start=1):
        name = c.normalize_team_name(getattr(opp, "team_name", None))
        if name == c.MY_TEAM_NAME:      # ESPN encodes a bye as playing yourself
            byes.append(wk)
            continue
        assert mgr_of.get(name) == expected_mgr[wk], (
            f"week {wk}: ESPN team {name!r} is managed by "
            f"{mgr_of.get(name)!r}, spec expects {expected_mgr[wk]!r}"
        )
        spec_name = next(
            c.normalize_team_name(o["team"]) for o in c.OPPONENTS if wk in o["weeks"]
        )
        if name != spec_name:
            renames.append(f"week {wk}: §1 says {spec_name!r}, ESPN now says {name!r}")

    assert byes == sorted(c.MY_BYE_WEEKS) == [5, 14]
    if renames:
        print("\n§1 team names are stale (harmless — everything keys on "
              "manager):\n  " + "\n  ".join(renames))


def test_team_names_are_resolved_from_the_league_not_hardcoded():
    """Names change; managers do not. Pins the fix for the "TBD" rename."""
    from ff_agent.data.espn import team_names_by_manager
    from ff_agent.draft import build as DB

    live = team_names_by_manager(c.SEASON)
    for mgr, team in zip(live["manager"].to_list(), live["team"].to_list()):
        if mgr:
            assert DB.team_name_for(mgr) == c.normalize_team_name(team), mgr


def test_the_milestone_1_gate_draftable_pool():
    """§0.2, on the population that matters on draft day."""
    from ff_agent.data import crosswalk as cw, espn
    pool = espn.draftable_players(c.SEASON)
    players = pool.filter(pl.col("position") != "D/ST")
    r = cw.resolve_players(players)
    cw.assert_all_resolved(r, f"draftable_pool_{c.SEASON}")
    assert r.filter(pl.col("match_method") == "unresolved").height == 0


def test_the_milestone_1_gate_last_season_rosters():
    from ff_agent.data import crosswalk as cw, espn
    ros = espn.rosters(c.LAST_SEASON)
    players = ros.filter(pl.col("position") != "D/ST")
    r = cw.resolve_players(players)
    cw.assert_all_resolved(r, f"rosters_{c.LAST_SEASON}")


def test_all_32_dst_present_and_mappable():
    from ff_agent.data import crosswalk as cw, espn
    pool = espn.draftable_players(c.SEASON)
    dst = pool.filter(pl.col("position") == "D/ST")
    assert dst.height == 32
    valid = set(cw.dst_crosswalk()["team"].to_list())
    bad = [t for t in dst["team"].to_list() if cw.normalize_team(t) not in valid]
    assert not bad, f"unmappable D/ST teams: {bad}"


@pytest.mark.parametrize("year", [2023, 2024, 2025])
def test_historical_drafts_fully_resolve(year):
    """§7.5's opponent model reads these. An unresolved pick is a hole in it."""
    from ff_agent.data import crosswalk as cw, espn
    d = espn.draft_results(year)
    dst = d.filter(pl.col("espn_id").str.starts_with("-"))
    players = d.filter(~pl.col("espn_id").str.starts_with("-")).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("position"),
        pl.lit(None, dtype=pl.Utf8).alias("team"))
    r = cw.resolve_players(players)
    cw.assert_all_resolved(r, f"draft_{year}")
    assert cw.resolve_dst(dst).filter(
        pl.col("match_method") == "unresolved").height == 0


def test_league_history_depth_is_three_seasons():
    """Gates Milestone 5. Team COUNT changed year to year (8/10/8 -> 9), so
    positional-run dynamics do not transfer cleanly between seasons."""
    from ff_agent.data import espn
    sizes = {}
    for yr in (2023, 2024, 2025):
        sizes[yr] = espn.draft_results(yr)["fantasy_team"].n_unique()
    assert sizes == {2023: 8, 2024: 10, 2025: 8}
    assert c.N_TEAMS == 9, "2026 is a 9-team league — a size this league has not run before"


def test_draftable_pool_resolves_to_the_right_people():
    """Companion to the §0.2 gate: resolving is not enough, it must resolve to
    the correct player. Catches nflverse espn_id errors."""
    from ff_agent.data import crosswalk as cw, espn
    pool = espn.draftable_players(c.SEASON).filter(pl.col("position") != "D/ST")
    r = cw.resolve_players(pool)
    cw.assert_resolutions_plausible(r, f"draftable_pool_{c.SEASON}", c.SEASON)
