"""Kicker streaming (§3.5, ADD-§E).

**ESPN's own projection cannot price this league's kicker scoring.** M3 found
ESPN merges 50-59 and 60+ into a single ``50Plus`` bucket, while §1 pays **5 and
6** respectively — so the projection collapses exactly the distinction ADD-§E
says is worth money here. That is why a home-grown kicker number is required
rather than optional, and it is the second of the two corrections ``ros.py``
applies to the ESPN anchor.

§3.5's rule: favour accurate big legs on offences that stall in field-goal range.
The scoring is distance-weighted with a flat **-1 per miss at any distance**, so
a long-range specialist who misses is not punished more than a short-range one —
which makes range strictly valuable and accuracy separately valuable.

ADD-§H's negative finding is respected: 4th-down aggressiveness has **no
measurable effect** on kicker scoring, so it is not an input here.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ff_agent.scoring.rules import load_rules

BUCKETS = (("FG0", 0, 39), ("FG40", 40, 49), ("FG50", 50, 59), ("FG60", 60, 99))


@dataclass
class KickerWeek:
    name: str
    team: str
    opponent: str
    expected_attempts: dict[str, float]
    expected_pats: float
    miss_rate: float
    total: float

    def summary(self) -> dict:
        return {
            "kicker": self.name, "team": self.team, "vs": self.opponent,
            "attempts": {k: round(v, 2) for k, v in self.expected_attempts.items()},
            "pats": round(self.expected_pats, 2),
            "total": round(self.total, 2),
        }


def score_week(
    name: str,
    team: str,
    opponent: str,
    expected_attempts: dict[str, float],
    expected_pats: float = 2.2,
    miss_rate: float = 0.15,
    rules: dict[str, float] | None = None,
) -> KickerWeek:
    """Expected points under §1, bucket by bucket.

    ``expected_attempts`` is keyed by the bucket names in ``BUCKETS``. Keeping
    the 60+ bucket separate is the entire point — collapsing it into 50+, as
    ESPN's own projection does, throws away a point per long make in a league
    that pays for it.
    """
    r = rules or load_rules()
    made_pts = 0.0
    attempts = 0.0
    for key, _lo, _hi in BUCKETS:
        n = float(expected_attempts.get(key, 0.0))
        attempts += n
        made_pts += n * (1 - miss_rate) * float(r.get(key, 0.0))
    miss_pts = attempts * miss_rate * float(r.get("FGM", -1.0))
    pat_pts = expected_pats * float(r.get("PAT", 1.0))
    return KickerWeek(
        name=name, team=team, opponent=opponent,
        expected_attempts=dict(expected_attempts), expected_pats=expected_pats,
        miss_rate=miss_rate, total=made_pts + miss_pts + pat_pts,
    )


def long_range_premium(rules: dict[str, float] | None = None) -> float:
    """What ESPN's merged bucket costs us, per expected 60+ make.

    One point. Small, real, and invisible to every other source — which is
    exactly the shape of most of this project's edges.
    """
    r = rules or load_rules()
    return float(r.get("FG60", 6.0)) - float(r.get("FG50", 5.0))


def rank(weeks: list[KickerWeek]) -> pl.DataFrame:
    if not weeks:
        return pl.DataFrame(schema={"kicker": pl.Utf8, "total": pl.Float64})
    return pl.DataFrame([w.summary() for w in weeks]).sort("total", descending=True)
