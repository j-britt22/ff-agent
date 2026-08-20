"""Scoring engine — the derivations that a generic implementation gets wrong.

§7.1 names sacks taken, rushing attempts and the two D/ST bucket tables as the
places this silently breaks. Each derivation below was established empirically
against ESPN's own recorded scores, and the figure in each docstring is the
measurement that justified it.
"""
import polars as pl
import pytest

from ff_agent.config import LAST_SEASON
from ff_agent.scoring import dst as D
from ff_agent.scoring import engine as E
from ff_agent.scoring.rules import load_rules


@pytest.fixture(scope="module")
def rules():
    return load_rules(LAST_SEASON)


# ─── the two rules nobody else scores (§3.3, §3.4) ──────────────────────────
def test_sacks_taken_maps_to_sacks_suffered():
    """NOT `sacks` (no such column) and NOT `def_sacks` (a DEFENDER's sacks)."""
    assert E.PLAYER_STAT_MAP["sacks_suffered"] == "SKD"
    assert "def_sacks" not in E.PLAYER_STAT_MAP
    assert "sacks" not in E.PLAYER_STAT_MAP


def test_rushing_attempts_maps_to_carries():
    """`attempts` is PASS attempts; `carries` is rush attempts."""
    assert E.PLAYER_STAT_MAP["carries"] == "RA"
    assert E.PLAYER_STAT_MAP.get("attempts") != "RA"


def test_a_sacked_qb_loses_points(rules):
    stats = pl.DataFrame({
        "player_id": ["x"], "player_display_name": ["Test QB"], "position": ["QB"],
        "team": ["KC"], "season": [2025], "week": [1], "season_type": ["REG"],
        "passing_yards": [250.0], "passing_tds": [2.0], "passing_interceptions": [1.0],
        "sacks_suffered": [5.0], "passing_2pt_conversions": [0.0],
        "completions": [20.0], "attempts": [30.0],
        "rushing_yards": [20.0], "carries": [4.0], "rushing_tds": [0.0],
        "rushing_2pt_conversions": [0.0],
        "receiving_yards": [0.0], "receptions": [0.0], "receiving_tds": [0.0],
        "receiving_2pt_conversions": [0.0], "fumbles_lost_total": [1.0],
        "fumble_recovery_tds": [0.0], "pat_made": [0.0], "fg_missed": [0.0],
        "fg_blocked": [0.0], "fg_made_0_19": [0.0], "fg_made_20_29": [0.0],
        "fg_made_30_39": [0.0], "fg_made_40_49": [0.0], "fg_made_50_59": [0.0],
        "fg_made_60_": [0.0], "pt_return_tds": [0.0], "special_teams_tds": [0.0],
    })
    out = E.score_players(2025, rules=rules, stats=stats)
    # 250*.04=10, 2 TD=12, INT=-2, 5 sacks=-5, 4 carries=.2, 20 rush yd=2, fumble=-2
    assert out["pts_SKD"][0] == -5.0
    assert out["pts_RA"][0] == pytest.approx(0.2)
    assert out["fantasy_points"][0] == pytest.approx(15.2)


# ─── kicker ─────────────────────────────────────────────────────────────────
def test_blocked_field_goals_count_as_misses(rules):
    """§1's flat -1 "Total FG Missed" includes blocks. nflverse tracks them
    separately: fg_missed alone matches ESPN on 109/113 kicker-weeks, and
    fg_missed + fg_blocked on 113/113."""
    base = dict.fromkeys(
        ["passing_yards", "passing_tds", "passing_interceptions", "sacks_suffered",
         "passing_2pt_conversions", "completions", "attempts", "rushing_yards",
         "carries", "rushing_tds", "rushing_2pt_conversions", "receiving_yards",
         "receptions", "receiving_tds", "receiving_2pt_conversions",
         "fumbles_lost_total", "fumble_recovery_tds", "fg_made_0_19",
         "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59",
         "fg_made_60_", "pt_return_tds", "special_teams_tds"], [0.0])
    stats = pl.DataFrame({
        "player_id": ["k"], "player_display_name": ["Test K"], "position": ["K"],
        "team": ["DET"], "season": [2025], "week": [1], "season_type": ["REG"],
        "pat_made": [3.0], "fg_missed": [0.0], "fg_blocked": [1.0], **base,
    })
    out = E.score_players(2025, rules=rules, stats=stats)
    assert out["pts_FGM"][0] == -1.0
    assert out["fantasy_points"][0] == pytest.approx(2.0)   # 3 PAT - 1 block


def test_field_goal_distance_buckets_collapse_correctly():
    """§1 pays one rate for 0-39; nflverse splits it into three buckets."""
    assert E.PLAYER_STAT_MAP["fg_made_40_49"] == "FG40"
    assert E.PLAYER_STAT_MAP["fg_made_50_59"] == "FG50"
    assert E.PLAYER_STAT_MAP["fg_made_60_"] == "FG60"
    for c in ("fg_made_0_19", "fg_made_20_29", "fg_made_30_39"):
        assert c not in E.PLAYER_STAT_MAP     # summed into FG0 explicitly


# ─── D/ST derivations ───────────────────────────────────────────────────────
def test_dst_touchdowns_span_three_nflverse_columns():
    """def_tds alone misses fumble-recovery and special-teams returns; all three
    together match ESPN on 128/128 team-weeks of 2025."""
    assert E.DST_TD_COLUMNS == ("def_tds", "fumble_recovery_tds", "special_teams_tds")


def test_dst_fumble_recovery_column():
    """fumble_recovery_opp matches ESPN on 134/135; def_fumbles matches neither
    direction (MIN wk3 2025: ESPN 3, recovery_opp 3, def_fumbles 0)."""
    assert E.DST_FUMBLE_RECOVERY_COLUMN == "fumble_recovery_opp"


def test_yards_allowed_adds_negative_sack_yardage():
    """sack_yards_lost is stored NEGATIVE, so it ADDS to give net yards.
    Exact on 119/119 team-weeks of 2025; gross passing+rushing is wrong."""
    d = D.dst_inputs(LAST_SEASON)
    from ff_agent.data import nflverse as nv
    ts = nv.team_stats(LAST_SEASON, offline=True).filter(pl.col("season_type") == "REG")
    row = d.filter((pl.col("team") == "TB") & (pl.col("week") == 3)).to_dicts()[0]
    opp = ts.filter((pl.col("team") == row["opp"]) & (pl.col("week") == 3)).to_dicts()[0]
    assert opp["sack_yards_lost"] <= 0
    assert row["yards_allowed"] == (
        opp["passing_yards"] + opp["sack_yards_lost"] + opp["rushing_yards"]
    )
    assert row["yards_allowed"] == 267        # ESPN's recorded figure


def test_points_allowed_excludes_only_what_the_offense_conceded():
    """A pick-six is the offense's doing, so ESPN does not charge the D/ST.
    MIN wk1 2025: CHI scored 24, MIN's D/ST was charged 18."""
    d = D.dst_inputs(LAST_SEASON)
    row = d.filter((pl.col("team") == "MIN") & (pl.col("week") == 1)).to_dicts()[0]
    assert row["opp_score"] == 24
    assert row["offense_conceded_points"] == 6
    assert row["points_allowed"] == 18


def test_special_teams_scores_still_count_as_allowed():
    """The fantasy D/ST includes special teams, so a kickoff-recovery TD or a
    punter tackled in his own end zone IS charged. PIT wk2 2025: SEA scored 31
    including a kickoff TD, and PIT's D/ST was charged all 31."""
    d = D.dst_inputs(LAST_SEASON)
    row = d.filter((pl.col("team") == "PIT") & (pl.col("week") == 2)).to_dicts()[0]
    assert row["opp_score"] == 31
    assert row["points_allowed"] == 31
    # and the punt-formation safety case
    phi = d.filter((pl.col("team") == "PHI") & (pl.col("week") == 4)).to_dicts()[0]
    assert phi["points_allowed"] == phi["opp_score"] == 25


def test_dst_sacks_come_from_play_by_play():
    """team_stats.def_sacks aggregates fractional PLAYER credits and misses one
    sack in 2025 (LV wk6: six sack plays, 5.0 credited). Counting sack PLAYS
    matches ESPN on 128/128."""
    s = D.dst_sacks(LAST_SEASON)
    lv = s.filter((pl.col("team") == "LV") & (pl.col("week") == 6))
    assert lv["sacks"][0] == 6.0


def test_rams_abbreviation_normalised_for_joins():
    """nflverse spells the Rams "LA"; ESPN and the crosswalk use "LAR"."""
    out = E.score_dst(LAST_SEASON)
    teams = set(out["team"].unique().to_list())
    assert "LAR" in teams and "LA" not in teams
