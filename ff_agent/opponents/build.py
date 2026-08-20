"""opponents.json (§4, §7.5) — what the draft simulator will consume.

Every tendency ships with the evidence behind it. §2.3 weights four managers at
2x and those are not the managers with the most data, so a profile that hides its
own confidence would be worse than no profile.
"""

from __future__ import annotations

import json

import polars as pl

from ff_agent.config import ARTIFACTS_DIR, DOUBLE_UP_MANAGERS, OPPONENTS, SEASON
from ff_agent.opponents import history as H
from ff_agent.opponents import tendencies as TD


def build(season: int = SEASON) -> dict:
    picks = TD.picks_with_consensus()
    cov = H.manager_coverage()
    tilt = TD.positional_tendency(picks)
    reach = TD.reach_profile(picks)
    qb = TD.qb_timing(picks)
    bias = TD.team_bias(picks)

    spec_by_mgr = {H.canonical_manager(o["manager"]): o for o in OPPONENTS}
    dbl = {H.canonical_manager(m) for m in DOUBLE_UP_MANAGERS}

    managers = []
    for mgr in sorted(spec_by_mgr):
        c = cov.filter(pl.col("manager") == mgr)
        spec = spec_by_mgr[mgr]
        t = tilt.filter(pl.col("manager") == mgr)
        r = reach.filter(pl.col("manager") == mgr)
        q = qb.filter(pl.col("manager") == mgr)
        b = bias.filter(pl.col("manager") == mgr).head(3)

        managers.append({
            "manager": mgr,
            "team": spec["team"],
            "weeks_faced": list(spec["weeks"]),
            "faced_twice": mgr in dbl,
            "schedule_weight": spec["weight"],
            "evidence": {
                "n_drafts": int(c["n_drafts"][0]) if c.height else 0,
                "n_picks": int(c["n_picks"][0]) if c.height else 0,
                "n_drafts_two_qb": int(c["n_drafts_two_qb"][0]) if c.height else 0,
                "seasons": c["seasons"][0].to_list() if c.height else [],
                "qb_confidence": c["qb_confidence"][0] if c.height else "none",
            },
            "positional_tilt": [
                {k: row[k] for k in ("phase", "position", "rate", "league_rate",
                                     "tilt_vs_league", "evidence_weight")}
                for row in t.to_dicts()
            ],
            "reach": {
                "mean_reach": float(r["mean_reach"][0]) if r.height else None,
                "raw_mean_reach": float(r["raw_mean_reach"][0]) if r.height else None,
                "reach_sd": float(r["reach_sd"][0]) if r.height and r["reach_sd"][0] else None,
                "league_mean_reach": float(r["league_mean_reach"][0]) if r.height else None,
            },
            "qb_timing_two_qb_only": (
                {k: q[k][0] for k in ("qbs_taken", "first_qb_fraction",
                                      "median_qb_fraction")} if q.height else None
            ),
            "nfl_team_bias": [
                {"team": row["nfl_team"], "picks": int(row["n"]),
                 "bias_vs_league": round(row["bias"], 4)} for row in b.to_dicts()
            ],
        })

    return {
        "season": season,
        "league": "Wildcats League",
        "caveats": {
            "format_change": (
                "2023 and 2024 were ONE-QB leagues; only 2025 was 2-QB. The first "
                "QB went at pick 16, then 35, then 7. QB tendencies rest on a "
                "single 2-QB draft."
            ),
            "team_count": "8, 10, 8 and now 9 — picks are normalised to a fraction of the draft.",
            "shrinkage": (
                f"Tendencies are shrunk toward the league prior with a "
                f"{TD.SHRINKAGE_PICKS:.0f}-pick pseudo-count; the format-matched "
                f"season carries {TD.FORMAT_MATCH_WEIGHT:.0f}x weight."
            ),
            "backbone": "Superflex ADP is the backbone; these are adjustments (§7.5).",
        },
        "managers": managers,
    }


def write(season: int = SEASON) -> str:
    payload = build(season)
    path = ARTIFACTS_DIR / "opponents.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return str(path)
