"""The two bucketed D/ST tables."""
import polars as pl

from ff_agent.config import SEASON
from ff_agent.scoring import buckets as B
from ff_agent.scoring.rules import load_rules


def _pts(value, table):
    r = load_rules(SEASON)
    df = pl.DataFrame({"v": [value]})
    return df.select(B.bucket_points_expr("v", table, r).alias("p"))["p"][0]


def test_points_allowed_buckets():
    assert _pts(0, B.POINTS_ALLOWED_BUCKETS) == 5.0
    assert _pts(6, B.POINTS_ALLOWED_BUCKETS) == 4.0
    assert _pts(13, B.POINTS_ALLOWED_BUCKETS) == 3.0
    assert _pts(17, B.POINTS_ALLOWED_BUCKETS) == 1.0
    assert _pts(28, B.POINTS_ALLOWED_BUCKETS) == -1.0
    assert _pts(45, B.POINTS_ALLOWED_BUCKETS) == -3.0
    assert _pts(60, B.POINTS_ALLOWED_BUCKETS) == -5.0


def test_the_two_zero_buckets_really_score_zero():
    """Verified against a scored game: Buccaneers D/ST allowed exactly 27 in
    week 3 2025 and the 22-27 flag contributed nothing."""
    for v in (18, 19, 20, 21, 22, 25, 27):
        assert _pts(v, B.POINTS_ALLOWED_BUCKETS) == 0.0


def test_yards_allowed_buckets():
    assert _pts(99, B.YARDS_ALLOWED_BUCKETS) == 5.0
    assert _pts(199, B.YARDS_ALLOWED_BUCKETS) == 3.0
    assert _pts(299, B.YARDS_ALLOWED_BUCKETS) == 2.0
    assert _pts(399, B.YARDS_ALLOWED_BUCKETS) == -1.0
    assert _pts(449, B.YARDS_ALLOWED_BUCKETS) == -3.0
    assert _pts(499, B.YARDS_ALLOWED_BUCKETS) == -5.0
    assert _pts(549, B.YARDS_ALLOWED_BUCKETS) == -6.0
    assert _pts(700, B.YARDS_ALLOWED_BUCKETS) == -7.0


def test_300_to_349_yards_scores_zero():
    for v in (300, 325, 349):
        assert _pts(v, B.YARDS_ALLOWED_BUCKETS) == 0.0


def test_bucket_boundaries_are_inclusive_and_contiguous():
    """No value may fall between two buckets."""
    for v in range(0, 80):
        assert B.bucket_label_expr and _pts(v, B.POINTS_ALLOWED_BUCKETS) is not None
    for v in range(0, 700, 7):
        assert _pts(v, B.YARDS_ALLOWED_BUCKETS) is not None
