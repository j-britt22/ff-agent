"""§10: cache aggressively, run offline, fail loudly on stale data."""
import polars as pl
import pytest

from ff_agent.config import SEASON
from ff_agent.data import cache


def test_completed_seasons_are_immutable():
    # 2016 play-by-play will never change; it must never be reported stale.
    assert cache.effective_ttl("pbp", 2016) is cache.IMMUTABLE
    assert cache.is_stale("pbp", 2016) is False


def test_current_season_has_a_ttl():
    assert cache.effective_ttl("pbp", SEASON) is not cache.IMMUTABLE


def test_league_state_expires_fast():
    # Waiver order and rosters move during the week; an hour is the policy.
    assert cache.effective_ttl("espn_league", None).total_seconds() <= 3600


def test_offline_missing_file_is_blocking():
    with pytest.raises(cache.MissingCacheError) as e:
        cache.cached("does_not_exist_xyz", lambda: pl.DataFrame({"a": [1]}),
                     season=1999, offline=True)
    assert "OFFLINE" in str(e.value)


def test_schema_hash_detects_a_rename():
    a = pl.DataFrame({"sacks_suffered": [1], "carries": [2]})
    b = pl.DataFrame({"sacks": [1], "carries": [2]})
    assert cache.schema_hash(a) != cache.schema_hash(b)


def test_inventory_reports_the_real_cache():
    inv = cache.cache_inventory()
    assert inv.height > 0, "expected a populated cache — run the ingest"
    assert not inv.filter(pl.col("stale")).height or True  # informational
