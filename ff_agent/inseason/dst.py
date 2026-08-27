"""D/ST streaming (§3.5, ADD-§F).

**This league scores YARDS allowed, and almost no public model projects them.**
§1's bucket runs from +5 under 100 yards to **-7 at 550+**, with continuous
yards-allowed switched off. Every public D/ST model projects points allowed plus
turnovers, so the single largest term in this league's defensive scoring is one
nobody else computes.

ADD-§F names the trap this creates, and it is worth encoding literally: a defence
facing a **high-volume, methodical offence that stalls in the red zone** allows
few points and 400+ yards, and posts a **negative** score here while looking
perfectly fine in every other league's box score. The mirror image — a fast,
three-and-out-prone opponent — is the ideal stream, because it concedes few plays
and therefore few yards.

So the projection is of opponent **total yards**, driven by pace (seconds per
play), plays per game and yards per play — not by opponent scoring. Streaming
inputs in ADD-§F's order: opponent implied total, opponent QB sack rate taken,
pressure rate allowed, EPA per play allowed, home/road as a tiebreaker.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ff_agent.scoring.buckets import POINTS_ALLOWED_BUCKETS, YARDS_ALLOWED_BUCKETS
from ff_agent.scoring.rules import load_rules

SECONDS_PER_GAME = 3600.0
"""Used to turn pace into a plays-per-game estimate when plays are not given."""


@dataclass
class StreamOption:
    team: str
    opponent: str
    projected_yards_allowed: float
    projected_points_allowed: float
    yards_bucket_points: float
    points_bucket_points: float
    turnover_points: float
    sack_points: float
    total: float
    warnings: list[str]

    def summary(self) -> dict:
        return {
            "dst": self.team, "vs": self.opponent,
            "proj_yards": round(self.projected_yards_allowed),
            "proj_points": round(self.projected_points_allowed, 1),
            "yards_bucket": round(self.yards_bucket_points, 1),
            "points_bucket": round(self.points_bucket_points, 1),
            "total": round(self.total, 2),
            "warnings": self.warnings,
        }


def _bucket_value(value: float, buckets, rules: dict[str, float]) -> float:
    """Points for a bucketed stat. A bucket with no rule scores ZERO.

    ESPN omits zero-valued rules from its payload entirely, so "absent" and
    "zero" look identical from the outside — M2 verified all three of §1's
    zero buckets are genuinely zero rather than transcription slips.
    """
    for low, high, rule in buckets:
        if (low is None or value >= low) and (high is None or value <= high):
            return 0.0 if rule is None else float(rules.get(rule, 0.0))
    return 0.0


def project_yards(
    opponent_plays_per_game: float,
    opponent_yards_per_play: float,
) -> float:
    """The number nobody else computes."""
    return float(opponent_plays_per_game * opponent_yards_per_play)


def plays_from_pace(seconds_per_play: float, share_of_clock: float = 0.5) -> float:
    """Pace -> plays per game. ADD-§F's driver, made explicit."""
    if seconds_per_play <= 0:
        raise ValueError("seconds_per_play must be positive")
    return SECONDS_PER_GAME * share_of_clock / seconds_per_play


def score_option(
    team: str,
    opponent: str,
    plays_per_game: float,
    yards_per_play: float,
    implied_total: float,
    expected_takeaways: float = 1.3,
    expected_sacks: float = 2.4,
    rules: dict[str, float] | None = None,
) -> StreamOption:
    """One week's expected D/ST points under §1, from opponent VOLUME.

    ``implied_total`` is the opponent's Vegas implied points — ADD-§F's top-ranked
    streaming input. Everything else is volume, which is what this league
    actually pays for.
    """
    r = rules or load_rules()
    yards = project_yards(plays_per_game, yards_per_play)
    ya = _bucket_value(yards, YARDS_ALLOWED_BUCKETS, r)
    pa = _bucket_value(implied_total, POINTS_ALLOWED_BUCKETS, r)
    to_pts = expected_takeaways * (
        (r.get("DEFINT", 2.0) + r.get("DEFFR", 2.0)) / 2
    )
    sack_pts = expected_sacks * r.get("DEFSK", 1.0)

    warnings: list[str] = []
    if yards >= 400 and implied_total <= 20:
        warnings.append(
            f"ADD-§F's trap: {opponent} is projected for {yards:.0f} yards but only "
            f"{implied_total:.0f} points — a methodical offence that stalls. This "
            f"defence looks fine in every other league and scores "
            f"{ya:+.0f} here on yardage alone."
        )
    if yards < 300 and plays_per_game < 62:
        warnings.append(
            f"the ideal stream: {opponent} is low-volume ({plays_per_game:.0f} "
            f"plays) as well as low-scoring"
        )
    return StreamOption(
        team=team, opponent=opponent, projected_yards_allowed=yards,
        projected_points_allowed=implied_total, yards_bucket_points=ya,
        points_bucket_points=pa, turnover_points=to_pts, sack_points=sack_pts,
        total=ya + pa + to_pts + sack_pts, warnings=warnings,
    )


def rank(options: list[StreamOption]) -> pl.DataFrame:
    """Weekly stream board, best first, with the downside case visible.

    §3.5: "always check the downside case". The yards bucket is shown as its own
    column precisely so a defence carried by turnover luck and exposed on volume
    cannot hide inside a single total.
    """
    if not options:
        return pl.DataFrame(schema={"dst": pl.Utf8, "vs": pl.Utf8, "total": pl.Float64})
    return pl.DataFrame([o.summary() for o in options]).sort("total", descending=True)
