"""Did last week's recommendations pay off?

§10 says to log every recommendation "or the season teaches you nothing", and
logging alone is only half of that — a log nobody reads back teaches nothing
either. This closes the loop weekly rather than once in January, which is what
turns a season of emails into a season of evidence.

It is also §11 step 10's acceptance test, run continuously:

  * did the claim I recommended outscore the CONTROL — the most-added player
    across the league that week (F4)?
  * did the player predicted to clear actually clear?
  * was the start/sit call right, and by how much?

**The control is the point.** M7's ``best_consensus`` arm found a policy with
zero board edge capturing +29.9 weekly points against the full model's +32.1 —
the board contributed 2.2 and the rest was bought from the opponents' spread.
Without a control the whole 32 gets banked as skill. If our recommendations
cannot beat the naive move, the honest output says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ff_agent.config import SEASON
from ff_agent.inseason import log as LOG


@dataclass
class WeekAudit:
    week: int
    claims: pl.DataFrame
    lineup: pl.DataFrame
    clears: pl.DataFrame
    notes: list[str] = field(default_factory=list)

    def scoreboard(self) -> dict:
        def _mean(df: pl.DataFrame, col: str) -> float | None:
            return round(float(df[col].mean()), 2) if df.height and col in df.columns else None
        return {
            "week": self.week,
            "claims_scored": self.claims.height,
            "mean_claim_edge_vs_control": _mean(self.claims, "edge_vs_control"),
            "lineup_points_left_on_bench": _mean(self.lineup, "points_missed"),
            "clear_prediction_accuracy": _mean(self.clears, "correct"),
            "notes": self.notes,
        }


def score_claims(
    recommended: pl.DataFrame,
    control: pl.DataFrame,
    actuals: pl.DataFrame,
) -> pl.DataFrame:
    """Recommended claim vs the most-added player, on points ACTUALLY scored.

    Measured against what the player added to my STARTING lineup rather than his
    raw points — a 20-point week from someone who never started is worth nothing,
    which is the same rule the recommendation was made under.
    """
    if recommended.is_empty():
        return pl.DataFrame(schema={
            "week": pl.Int64, "recommended": pl.Utf8, "control": pl.Utf8,
            "edge_vs_control": pl.Float64})
    pts = dict(zip(actuals["canonical_id"].to_list(),
                   actuals["points"].to_list())) if actuals.height else {}
    rows = []
    ctrl = dict(zip(control["week"].to_list(), control["canonical_id"].to_list())) \
        if control.height else {}
    for r in recommended.iter_rows(named=True):
        c_id = ctrl.get(r["week"])
        rows.append({
            "week": r["week"],
            "recommended": r.get("name") or r["canonical_id"],
            "control": c_id,
            "recommended_points": pts.get(r["canonical_id"]),
            "control_points": pts.get(c_id),
            "edge_vs_control": (
                None if pts.get(r["canonical_id"]) is None or pts.get(c_id) is None
                else round(pts[r["canonical_id"]] - pts[c_id], 2)
            ),
        })
    return pl.DataFrame(rows)


def score_lineup(recommended_starters: list[str], actual_started: list[str],
                 points: dict[str, float]) -> pl.DataFrame:
    """What the recommended lineup would have scored versus what was started."""
    ours = sum(points.get(p, 0.0) for p in recommended_starters)
    theirs = sum(points.get(p, 0.0) for p in actual_started)
    return pl.DataFrame([{
        "recommended_points": round(ours, 2),
        "actual_points": round(theirs, 2),
        "points_missed": round(ours - theirs, 2),
    }])


def score_clears(predicted: list[str], actually_cleared: set[str]) -> pl.DataFrame:
    """§9.3 asks the tool to say "this one will clear — grab him Wednesday".
    Wednesday's job is what checks whether the prediction held."""
    if not predicted:
        return pl.DataFrame(schema={"canonical_id": pl.Utf8, "correct": pl.Float64})
    return pl.DataFrame([
        {"canonical_id": p, "correct": 1.0 if p in actually_cleared else 0.0}
        for p in predicted
    ])


def audit_week(
    week: int,
    actuals: pl.DataFrame,
    control: pl.DataFrame | None = None,
    actually_cleared: set[str] | None = None,
    actual_started: list[str] | None = None,
    season: int = SEASON,
) -> WeekAudit:
    """Replay one week of the log against what happened."""
    records = [r for r in LOG.read(season) if r.get("week") == week]
    notes: list[str] = []

    claims = pl.DataFrame([
        {"week": week, "canonical_id": c["add_id"], "name": c.get("add")}
        for r in records if r["kind"] == "waivers"
        for c in r.get("claims", [])[:1]
    ]) if any(r["kind"] == "waivers" for r in records) else pl.DataFrame(
        schema={"week": pl.Int64, "canonical_id": pl.Utf8, "name": pl.Utf8})

    if control is None or control.is_empty():
        notes.append(
            "no control available for this week, so the claim edge is unmeasured. "
            "M7's precedent says most apparent edge is not real — a number with no "
            "control beside it should not be believed."
        )
        control = pl.DataFrame(schema={"week": pl.Int64, "canonical_id": pl.Utf8})

    lineup_rows = [r for r in records if r["kind"] == "lineup"]
    lineup = pl.DataFrame(schema={"points_missed": pl.Float64})
    if lineup_rows and actual_started is not None:
        pts = dict(zip(actuals["canonical_id"].to_list(), actuals["points"].to_list()))
        lineup = score_lineup(lineup_rows[-1].get("starters", []), actual_started, pts)

    predicted = [
        c.get("add_id") for r in records if r["kind"] == "waivers"
        for c in r.get("will_clear", [])
    ]
    clears = score_clears([p for p in predicted if p], actually_cleared or set())

    return WeekAudit(
        week=week,
        claims=score_claims(claims, control, actuals),
        lineup=lineup, clears=clears, notes=notes,
    )
