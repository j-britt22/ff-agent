"""Live state + rest-of-season numbers -> a Digest, per job.

This is the layer the CLI was missing: every engine was built and tested, and
nothing assembled real league state, ran an engine over it, and rendered the
result. Kept separate from ``jobs.py`` (which owns the guardrails and the send)
and from the engines (which own the maths), so each stays testable alone.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from ff_agent.config import MY_TEAM_NAME, SEASON
from ff_agent.inseason import clock as CK
from ff_agent.inseason import freeagents as FA
from ff_agent.inseason import lineup as LN
from ff_agent.inseason import playoffs as PO
from ff_agent.inseason import state as ST
from ff_agent.inseason import trades as TR
from ff_agent.inseason import value as V
from ff_agent.inseason import waivers as WV
from ff_agent.inseason.notify.base import Digest


def remaining_play_weeks(
    season: int, from_week: int, schedule: pl.DataFrame | None = None
) -> dict[str, tuple[int, ...]]:
    """Each fantasy team's REMAINING game weeks.

    Per-team rather than global because §2.1 and §2.2 are properties of MY
    schedule: my weeks 5 and 14 are not in anybody else's denominator, and
    pricing a rival's bye against my calendar is how M10a gave all nine teams
    my own byes.
    """
    from ff_agent.season import schedule as SCH

    sched = SCH.league_schedule(season) if schedule is None else schedule
    played = sched.filter(
        pl.col("opponent").is_not_null() & (pl.col("week") >= from_week)
    )
    return {
        str(t[0]): tuple(sorted(g["week"].to_list()))
        for t, g in played.group_by("team")
    }


def team_means(
    state: ST.LeagueState,
    ros: pl.DataFrame,
    play_weeks: dict[str, tuple[int, ...]],
) -> dict[str, float]:
    """Every team's expected weekly points — the simulator's input."""
    out: dict[str, float] = {}
    for team in state.rosters["fantasy_team"].unique().to_list():
        if not team:
            continue
        roster = ST.align(FA.playable(
            ST.with_values(state.roster_of(team), ros, team)))
        weeks = play_weeks.get(team) or tuple(range(state.week, 15))
        if not weeks:
            continue
        out[team] = round(V.roster_value(roster, weeks).mean, 2)
    return out


def _title_line(state: ST.LeagueState, means: dict[str, float]) -> str | None:
    """P(title) with the weeks played stated beside it.

    "18% to win it all" means a different thing with two weeks played than with
    ten, so M6's ceiling and the sample size both travel with the number.
    """
    if state.my_team not in means:
        return None
    try:
        odds = V.title_odds(means, state.my_team, completed=state.completed,
                            season=state.season, n_sims=6000)
    except Exception:
        return None
    return (
        f"P(playoffs) {odds.p_playoffs:.0%} · P(top-2 seed) {odds.p_top2:.0%} · "
        f"P(title) {odds.p_title:.0%} — {len(odds.completed_weeks)} weeks played. "
        f"M6 measured a PERFECT simulator at +0.52 Spearman on this task, so read "
        f"the probabilities and ignore any implied finishing order."
    )


# ─── Jobs ────────────────────────────────────────────────────────────────────
def waivers_digest(
    state: ST.LeagueState, ros: pl.DataFrame, n_sims: int = 6000
) -> tuple[Digest, dict]:
    """§9.3's ordered claim list."""
    play_weeks = remaining_play_weeks(state.season, state.week)
    mine = ST.align(FA.playable(ST.with_values(state.my_roster, ros, "my roster")))
    pool = ST.align(FA.rank_by_upside(
        FA.playable(ST.with_values(state.free_agents, ros, "free agents"))))
    my_weeks = play_weeks.get(state.my_team) or tuple(range(state.week, 15))

    my_id = None
    ids = state.rosters.filter(pl.col("fantasy_team") == state.my_team)["team_id"]
    if ids.len():
        my_id = ids[0]
    rivals = {
        int(tid): ST.align(state.rosters.filter(pl.col("team_id") == tid))
        for tid in state.rosters["team_id"].unique().to_list()
        if tid is not None and tid != my_id
    }

    means = team_means(state, ros, play_weeks)
    result = WV.build(
        my_roster=mine, free_agents=pool,
        play_weeks=my_weeks, week=state.week,
        waiver_order=state.waiver_order, rival_rosters=rivals, my_team_id=my_id,
        team_means=means, my_team=state.my_team, completed=state.completed,
        n_sims=n_sims,
    )

    sections = []
    if result.claims:
        lines = []
        for i, c in enumerate(result.claims, 1):
            d = f", {c.d_title:+.2%} title" if c.d_title is not None else ""
            lines.append(
                f"{i}. CLAIM {c.add_name} ({c.position}, {c.team})"
                + (f" — drop {c.drop_name}" if c.drop_name else "")
                + f" · +{c.weekly_delta:.2f} pts/wk{d} · {c.p_success:.0%} to win it"
            )
            lines += [f"     {r}" for r in c.reasons]
        sections.append(("Claim list — submit in this order", lines))
    if result.free_agent_grabs:
        sections.append((
            "Will clear — grab Wednesday at zero priority cost",
            [f"{c.add_name} ({c.position}) · +{c.weekly_delta:.2f} pts/wk"
             for c in result.free_agent_grabs],
        ))

    notes = list(result.notes) + list(state.notes)
    tl = _title_line(state, means)
    if tl:
        notes.insert(0, tl)
    if result.my_waiver_rank:
        notes.append(f"my waiver priority: #{result.my_waiver_rank} of nine.")

    subject = (
        f"{len(result.claims)} claim{'s' if len(result.claims) != 1 else ''}"
        + (f" — top is {result.claims[0].add_name}" if result.claims else " — nothing worth priority")
    )
    return Digest(
        job="waivers", subject=subject, week=state.week,
        headline=f"Option value of holding priority: {result.option_value:.2f} pts/wk.",
        sections=sections, alarms=result.alarms, notes=notes,
    ), {"claims": [c.add_id for c in result.claims],
        "grabs": [c.add_id for c in result.free_agent_grabs]}


def lineup_digest(
    state: ST.LeagueState,
    ros: pl.DataFrame,
    now: dt.datetime | None = None,
    injuries: pl.DataFrame | None = None,
    kickoffs: pl.DataFrame | None = None,
) -> tuple[Digest, dict]:
    """§5.3's lineup sequence for whatever checkpoint is due."""
    now = now or CK.now_et()
    kk = kickoffs if kickoffs is not None else CK.kickoff_table(state.season)
    mine = FA.playable(ST.with_values(state.my_roster, ros, "my roster"))

    plan = LN.build(mine, state.week, kk, now=now, injuries=injuries)
    sections = []
    actionable = plan.actionable()
    if actionable:
        sections.append((
            "Decisions",
            [f"{d.call} {d.name} ({d.position}, {d.slot}) — {d.reason()}"
             for d in actionable],
        ))
    if plan.starters.height:
        sections.append((
            "Starting lineup",
            [f"{r['slot']:5s} {r.get('name') or r['canonical_id']}"
             f" · {r.get('weekly_points') or 0:.1f} pts"
             + (f" · {r['p_out']:.0%} to sit" if (r.get("p_out") or 0) > 0.15 else "")
             for r in plan.starters.sort("slot").iter_rows(named=True)],
        ))
    toss = [d for d in plan.decisions if d.call == "toss-up"]
    if toss:
        sections.append(("Toss-ups", [f"{d.name}: {d.reason()}" for d in toss]))

    return Digest(
        job="lineup", subject=f"lineup — {len(actionable)} decision(s)",
        week=state.week, urgent=bool(plan.alarms),
        headline=f"{plan.open_slots} slots open, {len(plan.pins)} locked. "
                 f"Projected {plan.expected_points:.1f}.",
        sections=sections, alarms=plan.alarms,
        notes=list(plan.notes) + list(state.notes), deadline=
        plan.next_lock.at if plan.next_lock else None,
    ), {"decisions": [(d.canonical_id, d.call) for d in plan.decisions],
        "starters": sorted(plan.starters["canonical_id"].to_list())
        if plan.starters.height else []}


def trades_digest(
    state: ST.LeagueState, ros: pl.DataFrame, n_sims: int = 6000
) -> tuple[Digest, dict]:
    """§9.4's two-sided search, plus the profile of all eight rivals."""
    play_weeks = remaining_play_weeks(state.season, state.week)
    rosters = {
        t: ST.align(FA.playable(ST.with_values(state.roster_of(t), ros, t)))
        for t in state.managers
    }
    managers = {
        r["fantasy_team"]: r["manager"]
        for r in state.rosters.select("fantasy_team", "manager").unique().to_dicts()
        if r.get("manager")
    }
    means = team_means(state, ros, play_weeks)
    props, notes = TR.build(
        my_roster=rosters.get(state.my_team, pl.DataFrame()),
        rosters=rosters, play_weeks_by_team=play_weeks, managers=managers,
        my_team=state.my_team, team_means=means, completed=state.completed,
        n_sims=n_sims,
    )

    sections = []
    for p in props:
        d = f" · {p.d_title:+.2%} title" if p.d_title is not None else ""
        lines = [
            f"give {', '.join(p.i_give)} → get {', '.join(p.i_get)}",
            f"  me +{p.my_weekly_delta:.2f} pts/wk{d} · them +{p.their_weekly_delta:.2f} "
            f"pts/wk under THEIR numbers",
        ]
        lines += [f"  wedge: {w}" for w in p.my_wedges]
        lines += [f"  {n}" for n in p.notes]
        lines.append(f"  send: {p.message()}")
        sections.append((f"Trade with {p.partner}", lines))

    profile = TR.league_profile(rosters, play_weeks, managers)
    sections.append((
        "Every roster, ranked",
        [f"{r['team']}: {r['weekly_points']:.1f} pts/wk · holes {r['holes']}"
         + (" · DOUBLE-UP" if r["double_up"] else "")
         for r in profile.iter_rows(named=True)],
    ))
    return Digest(
        job="trades", subject=f"{len(props)} trade candidate(s)", week=state.week,
        sections=sections, notes=notes + list(state.notes),
    ), {"trades": [(p.partner, tuple(p.i_give), tuple(p.i_get)) for p in props]}


def week14_digest(state: ST.LeagueState, ros: pl.DataFrame) -> tuple[Digest, dict]:
    """§2.2's free week: no lineup, every drop free, weeks 15-17 only."""
    mine = ST.align(FA.playable(
        ST.with_values(state.my_roster, ros, "my roster")))
    pool = ST.align(FA.rank_by_upside(
        FA.playable(ST.with_values(state.free_agents, ros, "free agents")), keep=40))
    plan = PO.week14(mine, pool)
    view = PO.view(mine)
    return Digest(
        job="week14", subject="free week — churn for the bracket", week=14,
        headline="No lineup to set and no game to lose. Every drop is free.",
        sections=[
            ("Drop first (lowest weeks 15-17 value)",
             plan.drops["name"].to_list()),
            ("Target (pure playoff upside)", plan.targets["name"].to_list()),
            ("Playoff view", [
                f"roster is worth {view.weekly_points:.1f} pts/wk in weeks 15-17",
                *([f"weather-exposed: {', '.join(view.weather_exposed)}"]
                  if view.weather_exposed else []),
            ]),
        ],
        notes=list(plan.notes) + list(state.notes),
    ), {"drops": plan.drops["canonical_id"].to_list(),
        "targets": plan.targets["canonical_id"].to_list()}
