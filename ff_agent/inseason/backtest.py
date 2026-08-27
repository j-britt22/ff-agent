"""§11 step 10's gate — replay 2025 and compare against a CONTROL.

    "Replay several weeks of last season; compare optimizer lineups to what was
    actually started and recommended claims to what actually paid off."

**The controls are the point, not decoration.** M7's ``best_consensus`` arm found
a policy with ZERO board edge capturing +29.9 weekly points against the full
model's +32.1 — the board contributed 2.2 and the rest was bought from the
opponents' spread. Without a control the whole 32 gets banked as skill. So every
arm here has one, and if our recommendation cannot beat the naive move, **the
control ships and the finding gets written down.**

Finding F3 is what makes the replay possible at all. I had the historical
free-agent pool down as the thing that would sink this gate, because ESPN does
not retain "who was a free agent in week 6". It retains the pieces:

  * ``League.load_roster_week(week)`` requests ``mRoster`` with
    ``scoringPeriodId`` and rebuilds every team's roster as of that week;
  * ``League.transactions(scoring_period, types={FREEAGENT, WAIVER,
    WAIVER_ERROR})`` returns adds, drops, claims **and failed claims**;
  * ``League.box_scores(week)`` carries ``slot_position``, so what was actually
    STARTED is known.

Week W's pool is therefore ``draftable universe - union(week-W rosters)``, and
``WAIVER_ERROR`` is the league's only observed counterfactual — the sole
evidence available for calibrating P(claim succeeds).

**Caveats, stated before the numbers rather than after them:**

* 2025 was an **eight-team** league; 2026 is nine. Waiver depth scales with
  roster slots, so the replay's pool is SHALLOWER than next season's — it
  understates how rich the wire will be (§3.2). Direction safe, level not.
* 2025 *was* 2-QB and *did* use the post-2024 ruleset, so format and scoring
  match. A better starting position than M5 had.
* There is exactly ONE season of in-season 2-QB history. Wide bands, and a
  stated preference for the control whenever the margin sits inside them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ff_agent.inseason import lineup as LN
from ff_agent.inseason import value as V

REPLAY_SEASON = 2025
REPLAY_WEEKS = tuple(range(4, 14))
"""Weeks 4-13. Before week 4 there is not enough current-season usage to matter
(``ros.MIN_GAMES_FOR_USAGE``), and week 14 is my bye."""

CAVEATS = (
    "2025 was an EIGHT-team league; 2026 is nine. Waiver depth scales with "
    "roster slots, so this replay UNDERSTATES how rich the 2026 wire will be.",
    "2025 was 2-QB and used the post-2024 ruleset, so format and scoring match "
    "— a better starting position than M5 had.",
    "One season of in-season 2-QB history. Expect wide bands, and prefer the "
    "control whenever the margin sits inside them.",
)


@dataclass
class ArmResult:
    """One measured arm, against its control."""
    name: str
    n: int
    ours: float
    control: float
    control_name: str
    notes: list[str] = field(default_factory=list)

    @property
    def edge(self) -> float:
        return round(self.ours - self.control, 3)

    @property
    def verdict(self) -> str:
        if self.n == 0:
            return "UNMEASURED"
        return "BEATS CONTROL" if self.edge > 0 else "SHIP THE CONTROL"

    def summary(self) -> dict:
        return {
            "arm": self.name, "n": self.n,
            "ours": round(self.ours, 3), "control": round(self.control, 3),
            "control_is": self.control_name,
            "edge": self.edge, "verdict": self.verdict, "notes": self.notes,
        }


@dataclass
class GateResult:
    season: int
    weeks: tuple[int, ...]
    arms: list[ArmResult]
    caveats: tuple[str, ...] = CAVEATS

    @property
    def passed(self) -> bool:
        """A gate that can only pass is not a gate.

        Passing means every MEASURED arm beat its control. An unmeasured arm does
        not pass by default — it is reported as unmeasured, which is a different
        thing and reads differently.
        """
        measured = [a for a in self.arms if a.n > 0]
        return bool(measured) and all(a.edge > 0 for a in measured)

    def report(self) -> str:
        lines = [
            f"M10b gate — {self.season} weeks {self.weeks[0]}-{self.weeks[-1]}",
            "=" * 60, "",
            "CAVEATS, before the numbers:",
        ]
        lines += [f"  · {c}" for c in self.caveats]
        lines += ["", f"{'ARM':<18}{'OURS':>9}{'CONTROL':>9}{'EDGE':>9}  VERDICT"]
        for a in self.arms:
            lines.append(
                f"{a.name:<18}{a.ours:>9.2f}{a.control:>9.2f}{a.edge:>9.2f}  "
                f"{a.verdict}"
            )
            lines.append(f"{'':18}control = {a.control_name}")
        lines += ["", f"GATE: {'PASSED' if self.passed else 'NOT PASSED'}"]
        if not self.passed:
            lines.append(
                "  Not a failure to hide. M3b's headline test failed and was "
                "recorded rather than tuned away; if an arm loses to its control, "
                "the control ships and the finding goes in CLAUDE.md."
            )
        return "\n".join(lines)


# ─── Arms ────────────────────────────────────────────────────────────────────
def lineup_arm(weeks: list[dict]) -> ArmResult:
    """Our optimal lineup vs what was actually started, and vs ESPN's own.

    Each week dict carries ``roster`` (with weekly_points and espn_projection),
    ``actual_points`` and ``actual_started``. The ESPN control matters: if we
    cannot beat ESPN's own projections at setting a lineup, we should ship those.
    """
    ours = control = 0.0
    n = 0
    for wk in weeks:
        pts = wk["actual_points"]
        roster = wk["roster"]
        our_start = LN.LU.optimal_lineup(roster)["canonical_id"].to_list()
        espn_start = LN.LU.optimal_lineup(
            roster, value="espn_projection"
        )["canonical_id"].to_list() if "espn_projection" in roster.columns else our_start
        ours += sum(pts.get(p, 0.0) for p in our_start)
        control += sum(pts.get(p, 0.0) for p in espn_start)
        n += 1
    return ArmResult(
        "lineup", n, ours / max(n, 1), control / max(n, 1),
        "ESPN's own projection-optimal lineup",
    )


def waiver_arm(weeks: list[dict]) -> ArmResult:
    """Our top claim vs the most-added player across the league (F4).

    Scored on points added to the STARTING lineup, not raw points — a 20-point
    week from somebody who never started is worth nothing, which is the same rule
    the recommendation was made under.
    """
    ours = control = 0.0
    n = 0
    for wk in weeks:
        if not wk.get("our_claim") or not wk.get("most_added"):
            continue
        ours += wk["ros_gain"].get(wk["our_claim"], 0.0)
        control += wk["ros_gain"].get(wk["most_added"], 0.0)
        n += 1
    return ArmResult(
        "waivers", n, ours / max(n, 1), control / max(n, 1),
        "the most-added player across the league that week (F4)",
        notes=[] if n else [
            "no week carried both a recommendation and a control — the transaction "
            "log is what supplies the control, so this arm needs it."
        ],
    )


def claim_calibration(predicted: list[float], succeeded: list[bool]) -> ArmResult:
    """Predicted P(claim succeeds) against observed, on WAIVER_ERROR rows.

    Calibration, not a contest — so the "control" is a coin flip, which is the
    honest baseline for a probability nobody has measured before.
    """
    if not predicted:
        return ArmResult("claim odds", 0, 0.0, 0.0, "an uninformative 0.5 prior")
    brier = sum((p - float(s)) ** 2 for p, s in zip(predicted, succeeded)) / len(predicted)
    base = sum((0.5 - float(s)) ** 2 for s in succeeded) / len(succeeded)
    # lower Brier is better, so the sign is flipped to keep "edge > 0 is good"
    return ArmResult("claim odds", len(predicted), -brier, -base,
                     "an uninformative 0.5 prior")


def thursday_arm(decisions: list[dict]) -> ArmResult:
    """The arm that justifies the option-value machinery existing at all.

    Our counterfactual call versus naively starting the higher projection. If it
    loses, §5.3's simulation is decoration and the naive rule ships — which is
    exactly what M3b did with xFP.
    """
    ours = control = 0.0
    n = 0
    for d in decisions:
        actual = d["actual_points"]
        ours += actual.get(d["our_pick"], 0.0)
        control += actual.get(d["naive_pick"], 0.0)
        n += 1
    return ArmResult(
        "thursday call", n, ours / max(n, 1), control / max(n, 1),
        "naively starting the higher projection",
        notes=[] if n else [
            "no Thursday decision in the replay window. The arm is unmeasured, "
            "which is not the same as passing."
        ],
    )


def run(
    lineup_weeks: list[dict] | None = None,
    waiver_weeks: list[dict] | None = None,
    thursday_decisions: list[dict] | None = None,
    claim_predictions: tuple[list[float], list[bool]] | None = None,
    season: int = REPLAY_SEASON,
    weeks: tuple[int, ...] = REPLAY_WEEKS,
) -> GateResult:
    """Assemble the gate. Inputs are injected so the arms are testable offline."""
    arms = [
        lineup_arm(lineup_weeks or []),
        waiver_arm(waiver_weeks or []),
        thursday_arm(thursday_decisions or []),
    ]
    if claim_predictions:
        arms.append(claim_calibration(*claim_predictions))
    return GateResult(season=season, weeks=weeks, arms=arms)


# ─── Reading the real 2025 season (F3) ───────────────────────────────────────
def load_season(
    season: int = REPLAY_SEASON,
    weeks: tuple[int, ...] = REPLAY_WEEKS,
    my_team: str | None = None,
) -> dict:
    """Reconstruct the replay inputs from ESPN, week by week, with no lookahead.

    Each week is assembled from what was knowable on the Tuesday: rosters as of
    that week, the transaction log for the control, and box scores for what
    actually happened. Nothing here reads a later week.
    """
    from ff_agent.config import MY_TEAM_NAME
    from ff_agent.data import espn as ESPN
    from ff_agent.inseason import audit as AU

    my_team = my_team or MY_TEAM_NAME
    lineup_weeks, waiver_weeks = [], []
    notes: list[str] = []

    for wk in weeks:
        try:
            box = ESPN.started_lineup(season, wk)
        except Exception as exc:
            notes.append(f"week {wk}: no box scores ({type(exc).__name__})")
            continue

        mine = box.filter(pl.col("fantasy_team") == my_team)
        if mine.is_empty():
            notes.append(
                f"week {wk}: no rows for {my_team!r} — team names are mutable, "
                f"so this may be a rename rather than a bye."
            )
            continue

        actual_points = dict(zip(mine["espn_id"].to_list(),
                                 [p or 0.0 for p in mine["points"].to_list()]))
        started = mine.filter(
            ~pl.col("slot_position").is_in(["BE", "IR"])
        )["espn_id"].to_list()

        roster = mine.select(
            pl.col("espn_id").alias("canonical_id"),
            pl.col("name"),
            pl.col("slot_position").alias("position"),
            pl.col("projected_points").fill_null(0.0).alias("weekly_points"),
            pl.col("projected_points").fill_null(0.0).alias("espn_projection"),
        )
        lineup_weeks.append({
            "week": wk, "roster": roster,
            "actual_points": actual_points, "actual_started": started,
        })

        try:
            txns = ESPN.transactions(season, wk)
            waiver_weeks.append({
                "week": wk,
                "most_added": AU.most_added(txns, wk),
                "our_claim": None,      # filled by a replay that runs the engine
                "ros_gain": {},
            })
        except Exception as exc:
            notes.append(f"week {wk}: no transaction log ({type(exc).__name__})")

    if not lineup_weeks:
        notes.append(
            "no week produced usable data. The replay needs a COMPLETED season "
            "with box scores — 2025 is the only one this league has that is "
            "both 2-QB and on the current ruleset."
        )
    return {
        "lineup_weeks": lineup_weeks,
        "waiver_weeks": waiver_weeks,
        "notes": notes,
    }


def run_live(
    season: int = REPLAY_SEASON,
    weeks: tuple[int, ...] = REPLAY_WEEKS,
    my_team: str | None = None,
) -> GateResult:
    """The gate against real data. Still capable of not passing."""
    data = load_season(season, weeks, my_team)
    result = run(
        lineup_weeks=data["lineup_weeks"],
        waiver_weeks=data["waiver_weeks"],
        season=season, weeks=weeks,
    )
    for arm in result.arms:
        arm.notes.extend(data["notes"][:3])
    return result
