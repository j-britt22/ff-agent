"""§1 / §2 structural facts. Cheap, but they catch a bad edit to config."""
from ff_agent import config as c


def test_twelve_game_weeks_not_fourteen():
    # §2.2: my season ends in week 13; weeks 5 and 14 are byes.
    assert len(c.MY_GAME_WEEKS) == 12
    assert 5 not in c.MY_GAME_WEEKS and 14 not in c.MY_GAME_WEEKS


def test_schedule_covers_exactly_twelve_games():
    assert sum(len(o["weeks"]) for o in c.OPPONENTS) == 12
    weeks = sorted(w for o in c.OPPONENTS for w in o["weeks"])
    assert weeks == list(c.MY_GAME_WEEKS)


def test_four_double_up_opponents_eight_games():
    # §2.3: two-thirds of the schedule is four managers.
    assert len(c.DOUBLE_UP_MANAGERS) == 4
    assert sum(len(o["weeks"]) for o in c.OPPONENTS if o["weight"] == 2) == 8


def test_starter_slots_sum_to_ten():
    assert sum(c.STARTER_SLOTS.values()) == 10
    assert c.ROSTER_TOTAL == 10 + c.BENCH_SLOTS


def test_history_window_starts_at_ngs_boundary():
    assert c.HISTORY_START == 2016
    assert len(c.HISTORY_SEASONS) == 10
    assert c.SEASON not in c.HISTORY_SEASONS
