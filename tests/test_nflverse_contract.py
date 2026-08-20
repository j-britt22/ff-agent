"""The column contract — §3.3 and §3.4 are the whole edge.

sacks taken is `sacks_suffered` (NOT `sacks`; `def_sacks` is a DEFENDER's sacks)
rush attempts is `carries`     (`attempts` is PASS attempts)

If nflverse renames one of these, this fails here rather than silently
mispricing every QB in Milestone 4.
"""
import polars as pl
import pytest

from ff_agent.config import HISTORY_SEASONS
from ff_agent.data import nflverse as nv


@pytest.mark.parametrize("season", HISTORY_SEASONS)
def test_player_stats_contract_holds_every_season(season):
    df = nv.player_stats(season, offline=True)
    missing = sorted(nv.PLAYER_STATS_REQUIRED - set(df.columns))
    assert not missing, f"{season}: missing {missing}"
    assert df.height > 0


@pytest.mark.parametrize("season", HISTORY_SEASONS)
def test_team_stats_contract_holds_every_season(season):
    df = nv.team_stats(season, offline=True)
    assert not sorted(nv.TEAM_STATS_REQUIRED - set(df.columns))


def test_sacks_taken_is_not_confused_with_defensive_sacks():
    df = nv.player_stats(2025, offline=True)
    assert "sacks_suffered" in df.columns
    # A QB should have taken sacks; sacks_suffered must not be all-zero.
    qb = df.filter(pl.col("position") == "QB")
    assert qb["sacks_suffered"].sum() > 0
    # def_sacks is a different stat and must not be used as a substitute.
    assert "def_sacks" in df.columns


def test_carries_is_rush_attempts_not_pass_attempts():
    df = nv.player_stats(2025, offline=True)
    rb = df.filter(pl.col("position") == "RB")
    assert rb["carries"].sum() > 0
    # RBs should have ~no pass attempts; proves `attempts` is the passing column.
    assert rb["attempts"].sum() < rb["carries"].sum() / 100


def test_fg_distance_buckets_exist_for_scoring():
    # §1 pays 5 for a 50-59 FG and 6 for 60+; flat-3 models cannot express this.
    df = nv.player_stats(2025, offline=True)
    for c in ("fg_made_40_49", "fg_made_50_59", "fg_made_60_", "fg_missed"):
        assert c in df.columns
    assert df["fg_made_50_59"].sum() > 0


def test_schedules_cover_every_history_season():
    s = nv.schedules(offline=True)
    have = set(s["season"].unique().to_list())
    assert set(HISTORY_SEASONS) <= have
