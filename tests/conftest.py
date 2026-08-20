import pytest
import polars as pl

from ff_agent.config import have_espn_credentials
from ff_agent.data import crosswalk as cw


def pytest_collection_modifyitems(config, items):
    """Skip ESPN-dependent tests with a clear reason when .env is unfilled."""
    if have_espn_credentials():
        return
    skip = pytest.mark.skip(
        reason="no ESPN credentials in .env — see SETUP.md §3. "
               "Milestone 1 cannot CLOSE until these run."
    )
    for item in items:
        if "espn" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def canonical() -> pl.DataFrame:
    return cw.canonical_players()
