"""What each manager walked away with.

Team strength is the **optimal starting lineup solved week by week**, not a sum
of the roster — §9.2 is explicit that the FLEX has to be assigned rather than
filled, and M7's ``surrogate.lineup_stats`` already does exactly that. Reusing it
means a third quarterback is worth what a third quarterback is worth, and it
means byes price themselves rather than being adjusted for afterwards.

Every team is measured against **its own play weeks**, which is §2.1
generalised. My fantasy byes are weeks 5 and 14, so a player whose NFL bye lands
there costs me nothing; every other team has its own pair of free weeks, and in
a 9-team league over 14 weeks there are 14 bye slots to go round. The same code
therefore answers "how much did byes cost each team" for all nine, and the
answer differs between them for a reason that has nothing to do with talent.

Two things this deliberately does NOT do:

* **It does not rank teams by total VOR.** M7 settled that: VOR is measured
  against the STARTER replacement, so it prices bench players as though they
  would start. A team that drafted four quarterbacks would lead a VOR table and
  lose every week.
* **It does not pretend to resolve close calls.** M6 found the projections, not
  the simulator, are the binding constraint. This is reliable on roster SHAPE —
  who has holes, who is exposed to byes, who is thin at a starting slot — and
  weak on distinguishing two teams three points apart.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ff_agent.config import (
    MY_GAME_WEEKS, POSITION_MAXIMA, REGULAR_SEASON_WEEKS, SEASON, STARTER_SLOTS,
)
from ff_agent.draft import policy as POL
from ff_agent.draft import pool as PL
from ff_agent.draft import surrogate as SUR
from ff_agent.postdraft.results import DraftRecord


def play_weeks_by_manager(
    record: DraftRecord, season: int = SEASON
) -> tuple[dict[str, tuple[int, ...]], dict[str, str], list[str]]:
    """Each manager's fantasy game weeks and team name, from the live schedule.

    Falls back to my own weeks for everyone if the schedule cannot be read —
    §0.3 requires this to work offline, and a bye-cost figure that is slightly
    wrong for eight teams beats refusing to produce a report at all. The
    substitution is returned as a note, never swallowed.
    """
    notes: list[str] = []
    try:
        from ff_agent.draft import build as DDB
        from ff_agent.season import schedule as SCH

        sched = SCH.league_schedule(season)
        weeks: dict[str, tuple[int, ...]] = {}
        names: dict[str, str] = {}
        # polars hands `group_by` the key as a TUPLE, so unpacking it matters:
        # keying on ('Personality Hires',) silently matched nothing and quietly
        # gave all nine teams my own bye weeks.
        played = sched.filter(pl.col("opponent").is_not_null())
        by_team = {
            str(t[0]): tuple(sorted(g["week"].to_list()))
            for t, g in played.group_by("team")
        }
        # team_id -> the SCHEDULE's own spelling. Names are mutable and the two
        # ESPN tables cache independently, so a team renamed between fetches has
        # one spelling in the draft export and another in the schedule. On draft
        # night that happened live: "leah's team" became "Chase-ing Dubs" minutes
        # after she drafted Ja'Marr Chase, and the ESPN-sourced report died with
        # a KeyError inside the season simulator. team_id is the join that holds.
        id_to_name = {
            int(r["team_id"]): str(r["team"])
            for r in sched.select("team_id", "team").unique().to_dicts()
            if r["team_id"] is not None
        }
        id_of = _manager_team_ids(record)

        for m in record.managers:
            name = None
            if m in by_team:
                name = m
            elif id_of.get(m) in id_to_name:
                name = id_to_name[id_of[m]]
                notes.append(
                    f"'{m}' resolved to schedule team '{name}' by team_id — the "
                    "two ESPN tables disagree on its name"
                )
            else:
                name = DDB.team_name_for(m, season)
            names[m] = name
            if name in by_team:
                weeks[m] = by_team[name]
            else:
                weeks[m] = tuple(MY_GAME_WEEKS)
                notes.append(
                    f"no schedule row for {m} -> '{name}'; using my own play weeks, "
                    f"so that team's bye cost is approximate"
                )
        return weeks, names, notes
    except Exception as exc:                     # offline, or no credentials
        notes.append(
            f"league schedule unavailable ({type(exc).__name__}), so every team is "
            f"measured against MY play weeks {sorted(MY_GAME_WEEKS)}. Bye cost is "
            f"right for me and approximate for everyone else."
        )
        return (
            {m: tuple(MY_GAME_WEEKS) for m in record.managers},
            {m: m for m in record.managers},
            notes,
        )


def _manager_team_ids(record: DraftRecord) -> dict[str, int]:
    """``manager`` -> ESPN ``team_id``, when the source carried one."""
    if "team_id" not in record.picks.columns:
        return {}
    out = {}
    for r in record.picks.select("manager", "team_id").unique().to_dicts():
        if r["team_id"] is not None:
            out[r["manager"]] = int(r["team_id"])
    return out


def _playoff_sos(record: DraftRecord, season: int) -> dict[str, float] | None:
    """Mean weeks 15-17 schedule strength of a roster (§2.4), or None offline."""
    try:
        from ff_agent.projections import board_inputs as BI

        sos = BI.build(season).select("canonical_id", "playoff_sos")
    except Exception:
        return None
    j = record.picks.join(sos, on="canonical_id", how="left")
    out = j.group_by("manager").agg(pl.col("playoff_sos").mean().alias("sos"))
    return {r["manager"]: r["sos"] for r in out.to_dicts()}


def rate(
    record: DraftRecord,
    season: int = SEASON,
    rosters: dict[str, list[int]] | None = None,
    projected: dict[str, int] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """One row per manager: strength, shape, bye exposure, and what it cost them.

    Strength is measured on ``rosters`` — the real picks on a finished draft, and
    the picks extended with projected ones while it is still running. Roster
    SHAPE (the counts, the holes, the §10 alarms) always reports what the manager
    has actually done, because that is a fact and the padding is a model.
    """
    pool = record.pool
    weeks, team_names, notes = play_weeks_by_manager(record, season)
    sos = _playoff_sos(record, season)
    bye = pool.bye_week
    free_bye_cols = pool.df["free_bye_week_5_or_14"].to_numpy()

    rows = []
    for m in record.managers:
        real = np.array(record.roster_indices(m), dtype=np.int64)
        if real.size == 0:
            continue
        idx = np.array((rosters or {}).get(m, real.tolist()), dtype=np.int64)
        wk = tuple(weeks.get(m, MY_GAME_WEEKS))
        val = SUR.lineup_stats(pool, idx[None, :], play_weeks=wk)
        counts = {p: 0 for p in PL.POSITIONS}
        for i in real:
            counts[PL.POSITIONS[pool.pos_idx[i]]] += 1

        my_free = set(w for w in REGULAR_SEASON_WEEKS if w not in wk)
        free = int(sum(1 for i in real if int(bye[i]) in my_free))

        rows.append({
            "manager": m,
            "team": team_names.get(m, m),
            "picks": int(real.size),
            "projected_picks": int((projected or {}).get(m, 0)),
            "weekly_points": round(float(val.mean[0]), 2),
            "weekly_points_sd": round(float(val.mean_sd[0]), 2),
            "bye_cost": round(float(val.bye_cost[0]), 2),
            "fantasy_bye_weeks": sorted(my_free),
            "free_bye_players": free,
            "playoff_sos": (
                round(float(sos[m]), 3) if sos and sos.get(m) is not None else None
            ),
            "total_vor": round(float(np.nansum(pool.vor[real])), 1),
            **{f"n_{p}": counts[p] for p in PL.POSITIONS},
            "starter_holes": _starter_holes(counts),
            "alarms": _alarms(counts, record.rounds),
        })

    df = pl.DataFrame(rows).sort("weekly_points", descending=True)
    return df.with_columns(
        pl.col("weekly_points").rank("ordinal", descending=True).cast(pl.Int32)
        .alias("strength_rank")
    ), notes


def _starter_holes(counts: dict[str, int]) -> list[str]:
    """Starting slots this roster cannot fill (§1's ten starters).

    The FLEX is checked last and against the leftovers, for the reason §9.2
    gives: it accepts a superset of what the strict slots do, so filling it
    first gets the answer wrong.
    """
    holes = []
    for pos, need in POL.REQUIRED_STARTERS.items():
        if counts.get(pos, 0) < need:
            holes.append(f"{pos} {counts.get(pos, 0)}/{need}")
    spare = sum(
        counts.get(p, 0) - POL.REQUIRED_STARTERS.get(p, 0) for p in POL.FLEX_POOL
    )
    if spare < STARTER_SLOTS.get("FLEX", 0):
        holes.append("FLEX")
    return holes


def _alarms(counts: dict[str, int], rounds: int) -> list[str]:
    """§10's sanity alarms, applied to a finished roster.

    M7 split these into mine and the league's for a reason — "only 1 QB after
    round 11" is an alarm about my roster and a fact about a rival's. Here every
    team is reported, and nothing is inferred about what a rival should have
    done.
    """
    out = []
    for pos, cap in POSITION_MAXIMA.items():
        if counts.get(pos, 0) > cap:
            out.append(f"{counts[pos]} {pos} exceeds ESPN's maximum of {cap}")
    if counts.get("QB", 0) <= 1:
        out.append(f"only {counts.get('QB', 0)} QB in a 2-QB league")
    if counts.get("K", 0) == 0:
        out.append("no kicker")
    if counts.get("DST", 0) == 0:
        out.append("no D/ST")
    return out


def draft_efficiency(
    ratings: pl.DataFrame, grades: pl.DataFrame
) -> pl.DataFrame:
    """Join what a manager built to how they built it."""
    from ff_agent.postdraft.grade import GRADE_POINTS

    g = (
        grades.with_columns(pl.col("grade").replace_strict(GRADE_POINTS).alias("gp"))
        .group_by("manager")
        .agg(
            pl.col("gp").mean().round(2).alias("draft_gpa"),
            pl.col("regret_weekly_points").sum().round(1).alias("total_regret"),
            pl.col("value_vs_board").mean().round(1).alias("mean_value_vs_board"),
            pl.col("value_vs_ecr").mean().round(1).alias("mean_value_vs_ecr"),
            (pl.col("grade").is_in(["A+", "A"])).sum().alias("a_picks"),
            (pl.col("grade").is_in(["D", "F"])).sum().alias("poor_picks"),
        )
    )
    return ratings.join(g, on="manager", how="left").sort(
        "weekly_points", descending=True
    )
