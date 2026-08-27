import itertools

import polars as pl
import pytest

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


# ─── Synthetic league, so M10b tests never need credentials ──────────────────
# §0.3 wants the whole pipeline to run offline. The in-season jobs are the part
# most tempting to test only against the live league, and therefore the part most
# likely to end up untested. This builds a real 9-team, 14-week schedule with the
# league's actual shape — four games and one bye per week, 14 bye slots — so the
# simulator, the lineup solver and the value layer are all exercised for real.
SYNTH_TEAMS = tuple(f"T{i}" for i in range(1, 10))


def _round_robin(teams: tuple[str, ...], weeks: int) -> list[dict]:
    """Circle method with an odd team count: exactly one bye per week."""
    rotation = list(teams)
    rows: list[dict] = []
    for wk in range(1, weeks + 1):
        bye = rotation[0]
        rest = rotation[1:]
        for a, b in zip(rest[: len(rest) // 2], rest[len(rest) // 2:][::-1]):
            rows += [
                {"season": 2026, "week": wk, "team": a, "opponent": b,
                 "team_id": teams.index(a) + 1},
                {"season": 2026, "week": wk, "team": b, "opponent": a,
                 "team_id": teams.index(b) + 1},
            ]
        rows.append({"season": 2026, "week": wk, "team": bye, "opponent": None,
                     "team_id": teams.index(bye) + 1})
        rotation = [rotation[-1]] + rotation[:-1]
    return rows


@pytest.fixture(scope="session")
def synth_schedule() -> pl.DataFrame:
    return pl.DataFrame(_round_robin(SYNTH_TEAMS, 14))


@pytest.fixture
def synth_league(monkeypatch, synth_schedule):
    """Point every schedule reader at the synthetic league for one test."""
    from ff_agent.season import schedule as SCH

    def _matchups(season: int = 2026) -> pl.DataFrame:
        return (
            synth_schedule.filter(pl.col("opponent").is_not_null())
            .with_columns(
                pl.min_horizontal("team", "opponent").alias("a"),
                pl.max_horizontal("team", "opponent").alias("b"),
            )
            .unique(subset=["week", "a", "b"])
            .select("season", "week", "a", "b")
            .sort("week", "a")
        )

    monkeypatch.setattr(SCH, "league_schedule", lambda season=2026, **kw: synth_schedule)
    monkeypatch.setattr(SCH, "matchups", _matchups)
    return synth_schedule


def brute_force_lineup(players: pl.DataFrame, slots: dict[str, int],
                       value: str = "weekly_points",
                       pinned: dict[str, str] | None = None) -> float:
    """Exhaustive best lineup. Only tractable on tiny rosters — that is the point.

    The fast solver's optimality rests on an argument about the FLEX accepting a
    superset of the strict slots. An argument is not a proof that the CODE is
    right, so small cases get checked against every legal assignment.

    A slot may be left EMPTY: with pins in play a roster can genuinely be unable
    to fill one, and a search that requires every slot filled would score those
    rosters at the pinned floor and "prove" the solver wrong.
    """
    from ff_agent.season.lineup import slot_accepts

    pinned = pinned or {}
    rows = players.to_dicts()
    val = {r["canonical_id"]: r[value] for r in rows}
    pos = {r["canonical_id"]: r["position"] for r in rows}

    open_slots: list[str] = []
    for s, n in slots.items():
        open_slots += [s] * n
    for s in pinned.values():
        open_slots.remove(s)

    free = [r["canonical_id"] for r in rows if r["canonical_id"] not in pinned]

    def best_from(i: int, used: frozenset) -> float:
        if i == len(open_slots):
            return 0.0
        slot = open_slots[i]
        best = best_from(i + 1, used)                       # leave it empty
        for cid in free:
            if cid in used or not slot_accepts(slot, pos[cid]):
                continue
            best = max(best, val[cid] + best_from(i + 1, used | {cid}))
        return best

    return sum(val[c] for c in pinned) + best_from(0, frozenset())
