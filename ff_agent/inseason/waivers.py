"""§9.3 — the ordered claim list.

Rolling priority, not FAAB. **Priority is a depleting, indivisible asset**: there
is no bidding small, and every successful claim drops you to last. So the rule is
a comparison, not a ranking:

    claim if  marginal_value(player) > option_value(holding priority)

Six things fall out of writing it that way, and each one is a bug avoided:

1. **"Bench upgrades ≈ 0" needs no heuristic.** Value is measured *through* the
   lineup solver, so a player who never starts contributes exactly zero. §3.2's
   "don't hoard handcuffs" is enforced by arithmetic rather than by a rule
   somebody has to remember.
2. **Add and drop are ONE action, scored jointly.** The dropped player's own
   remaining starts are the cost. Constrained by the IR slot, §1's position
   maxima and starter feasibility — the same machinery M7 needed when an
   unconstrained policy drafted eight quarterbacks and no tight end.
3. **§2.1's free bye is a WEEKLY edge, not just a draft-board term.** My
   remaining play weeks exclude 5 and 14, so a player whose NFL bye lands there
   costs me nothing and costs five other teams a start. It falls out of
   ``play_weeks`` with no special case.
4. **Quarterback claims are special twice over.** §9.3: any startable QB on the
   wire is worth top priority in a 2-QB league. §3.3: this league charges -1 per
   sack and no public ranking prices it, so ``ros.py``'s sack correction is
   already in the number.
5. **P(claim succeeds) is modelled and calibratable.** Every team's priority is
   exposed and their holes are computable. A low probability is NOT a reason to
   skip a claim — a failed claim costs nothing — it is a reason to order the list
   differently.
6. **The ORDERING is the deliverable.** Claims process in my order and priority
   drops only on a success, so a single claim is strictly worse than a ranked
   list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ff_agent.config import (
    FREE_BYE_WEEKS, MY_TEAM_NAME, N_TEAMS, POSITION_MAXIMA,
)
from ff_agent.inseason import state as ST
from ff_agent.inseason import value as V

SCREEN_KEEP = 12
"""Candidates surviving the cheap lineup screen and going to the real simulator.

The screen is a strict prefilter rather than an approximation: a move that never
reaches my starting lineup in any remaining week cannot move my title odds."""

MATERIAL_WEEKLY = 0.20
"""Points per week below which a claim is not worth an email."""

BETTER_TARGET_PER_WEEK = 0.25
"""P(a target better than today's best appears on the wire next week).

A placeholder with a stated provenance: §3.2 argues the wire stays rich all
season (153 roster spots against 500+ relevant players), so this is not small.
M10b-3's gate replaces it with the measured rate from the 2025 transaction log.
Until then it is deliberately GENEROUS — an over-large option value makes the
engine too reluctant to spend priority, which is the safe direction to be wrong
in for a resource that recovers on its own."""


class WaiverError(RuntimeError):
    pass


@dataclass
class Claim:
    add_id: str
    add_name: str
    position: str
    team: str
    drop_id: str | None
    drop_name: str | None
    weekly_delta: float
    p_success: float
    free_bye: bool
    is_qb: bool
    d_title: float | None = None
    will_clear: bool = False
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "add": self.add_name, "position": self.position, "team": self.team,
            "drop": self.drop_name, "weekly_delta": round(self.weekly_delta, 2),
            "p_success": round(self.p_success, 2),
            "d_title": None if self.d_title is None else round(self.d_title, 4),
            "will_clear": self.will_clear, "why": "; ".join(self.reasons),
        }


@dataclass
class ClaimList:
    week: int
    claims: list[Claim]
    free_agent_grabs: list[Claim]
    option_value: float
    my_waiver_rank: int | None
    notes: list[str] = field(default_factory=list)
    alarms: list[str] = field(default_factory=list)

    @property
    def burn_priority_on(self) -> Claim | None:
        """The one worth going to the back of the queue for, or None."""
        return self.claims[0] if self.claims else None

    def summary(self) -> dict:
        return {
            "week": self.week,
            "my_waiver_rank": self.my_waiver_rank,
            "option_value_pts_per_week": round(self.option_value, 2),
            "claims": [c.summary() for c in self.claims],
            "will_clear_grab_wednesday": [c.summary() for c in self.free_agent_grabs],
            "notes": self.notes, "alarms": self.alarms,
        }


# ─── P(claim succeeds) ───────────────────────────────────────────────────────
BASE_INTEREST = 0.12
"""P a rival claims a given useful player absent any positional need. Calibrated
by M10b-3 against the league's WAIVER_ERROR rows — the only observed
counterfactual the league produces."""

HOLE_INTEREST = 0.55
"""...and if the player fills a hole in their STARTING lineup."""


def p_claim_succeeds(
    position: str,
    my_rank: int | None,
    rival_holes: dict[int, list[str]],
    rival_ranks: dict[int, int],
) -> float:
    """Chance this claim survives everyone ahead of me in the queue.

    §9.3: "if you're 7th and three teams ahead share the gap, the claim likely
    fails — which costs nothing." That last clause is the important half. A low
    probability never suppresses a claim; it only changes where the claim sits in
    the order, because a failed claim leaves priority untouched.
    """
    if my_rank is None:
        return 1.0
    p = 1.0
    for team_id, rank in rival_ranks.items():
        if rank >= my_rank:
            continue
        holes = rival_holes.get(team_id, [])
        wants = position in holes or (
            position in ("RB", "WR", "TE") and "FLEX" in holes
        )
        p *= 1.0 - (HOLE_INTEREST if wants else BASE_INTEREST)
    return round(p, 4)


def option_value(
    weeks_remaining: int,
    my_rank: int | None,
    best_available_now: float,
    n_teams: int = N_TEAMS,
) -> float:
    """§9.3's other side: what holding priority is worth.

    ``P(better target appears soon) × what you'd forgo at the back of the line``.
    The second factor is where §9.3's own argument lives — "nine teams makes
    priority cheap; the queue is only nine long and you climb back fast" — and
    writing it out makes that a number rather than an assertion: at rank 1 the
    forgone edge is the gap between winning a future claim and losing it, and
    with nine teams that gap is small and recovers within a couple of weeks.
    """
    if my_rank is None or weeks_remaining <= 0:
        return 0.0
    # Chance of winning a contested claim now, versus from the back of the queue.
    p_now = 1.0 - (my_rank - 1) / n_teams
    p_last = 1.0 / n_teams
    edge = max(0.0, p_now - p_last)
    horizon = min(weeks_remaining, 4)      # priority recovers; four weeks is generous
    return round(BETTER_TARGET_PER_WEEK * horizon * edge * best_available_now, 3)


# ─── Candidate generation ────────────────────────────────────────────────────
def candidate_pairs(
    my_roster: pl.DataFrame,
    free_agents: pl.DataFrame,
    play_weeks: tuple[int, ...],
    max_adds: int = 40,
) -> list[tuple[str, str | None, float]]:
    """Every legal (add, drop) with its weekly lineup delta, best first.

    Legality is not decoration. §1's maxima and starter feasibility are what stop
    the engine proposing the roster M7 measured at fifty points a week worse than
    a constrained one.
    """
    counts: dict[str, int] = {}
    for p in my_roster["position"].to_list():
        counts[p] = counts.get(p, 0) + 1

    fa = free_agents.sort("weekly_points", descending=True, nulls_last=True).head(max_adds)
    out: list[tuple[str, str | None, float]] = []

    for add in fa.iter_rows(named=True):
        pos = add["position"]
        cap = POSITION_MAXIMA.get(pos)
        add_row = pl.DataFrame([add]).select(my_roster.columns)

        drops = ST.droppable(my_roster, position_of_add=pos)
        # Never cut somebody we have no number for. An unpriced player reads as
        # zero and would otherwise always be the "cheapest" thing to drop —
        # which is exactly backwards, since not knowing his value is a reason
        # for caution, not a licence.
        if "priced" in drops.columns:
            drops = drops.filter(pl.col("priced").fill_null(False))
        for drop in drops.iter_rows(named=True):
            after_counts = dict(counts)
            after_counts[drop["position"]] -= 1
            after_counts[pos] = after_counts.get(pos, 0) + 1
            if cap is not None and after_counts[pos] > cap:
                continue
            after = pl.concat([
                my_roster.filter(pl.col("canonical_id") != drop["canonical_id"]),
                add_row,
            ])
            d = V.weekly_delta(my_roster, after, play_weeks)
            if d > 0:
                out.append((add["canonical_id"], drop["canonical_id"], d))

    out.sort(key=lambda t: -t[2])
    # keep only the best drop for each add — the rest are the same move, worse
    seen, best = set(), []
    for a, d, v in out:
        if a in seen:
            continue
        seen.add(a)
        best.append((a, d, v))
    return best


# ─── The list ────────────────────────────────────────────────────────────────
def build(
    my_roster: pl.DataFrame,
    free_agents: pl.DataFrame,
    play_weeks: tuple[int, ...],
    week: int,
    waiver_order: pl.DataFrame | None = None,
    rival_rosters: dict[int, pl.DataFrame] | None = None,
    my_team_id: int | None = None,
    team_means: dict[str, float] | None = None,
    my_team: str = MY_TEAM_NAME,
    completed: pl.DataFrame | None = None,
    n_sims: int = 8000,
    screen_keep: int = SCREEN_KEEP,
) -> ClaimList:
    """The Tuesday deliverable: an ORDERED claim list with reasons."""
    notes: list[str] = []
    alarms: list[str] = []

    pairs = candidate_pairs(my_roster, free_agents, play_weeks)
    if not pairs:
        notes.append(
            "no free agent improves the starting lineup in any remaining week. "
            "§9.3's bench-upgrades-are-worth-nothing, arriving as an empty list "
            "rather than as filler."
        )

    fa_by_id = {r["canonical_id"]: r for r in free_agents.iter_rows(named=True)}
    ros_by_id = {r["canonical_id"]: r for r in my_roster.iter_rows(named=True)}

    # who is ahead of me, and what do they need
    my_rank = None
    rival_ranks: dict[int, int] = {}
    rival_holes: dict[int, list[str]] = {}
    if waiver_order is not None and waiver_order.height:
        for row in waiver_order.iter_rows(named=True):
            tid, rank = row.get("team_id"), row.get("waiver_rank")
            if tid is None or rank is None:
                continue
            if tid == my_team_id:
                my_rank = int(rank)
            else:
                rival_ranks[int(tid)] = int(rank)
        for tid, r in (rival_rosters or {}).items():
            counts: dict[str, int] = {}
            for p in r["position"].to_list():
                counts[p] = counts.get(p, 0) + 1
            rival_holes[int(tid)] = ST.starter_holes(counts)
    else:
        notes.append(
            "waiver priority unavailable — every claim is scored as though it "
            "succeeds. Ordering still holds; the odds beside it do not."
        )

    best_now = pairs[0][2] if pairs else 0.0
    ov = option_value(len(play_weeks), my_rank, best_now)

    claims: list[Claim] = []
    for add_id, drop_id, wd in pairs[:screen_keep]:
        add = fa_by_id[add_id]
        drop = ros_by_id.get(drop_id) if drop_id else None
        pos = add["position"]
        p_ok = p_claim_succeeds(pos, my_rank, rival_holes, rival_ranks)
        free_bye = add.get("bye_week") in FREE_BYE_WEEKS

        reasons = [f"+{wd:.2f} pts/wk to the STARTING lineup"]
        if free_bye:
            reasons.append(
                f"NFL bye in week {add['bye_week']} — one of MY free weeks, so it "
                f"costs me nothing and costs five other teams a start (§2.1)"
            )
        if pos == "QB":
            reasons.append(
                "startable QB in a 2-QB league — §9.3 says this is where priority "
                "gets spent"
            )
        if add.get("sack_correction") and abs(add["sack_correction"]) > 0.5:
            reasons.append(
                f"sack term {add['sack_correction']:+.1f} pts — §3.3's edge, which "
                f"no public ranking prices"
            )
        if p_ok < 0.35:
            reasons.append(
                f"only {p_ok:.0%} to survive the queue — but a failed claim costs "
                f"nothing, so it stays on the list"
            )
        claims.append(Claim(
            add_id=add_id, add_name=add.get("name") or add_id, position=pos,
            team=add.get("team") or "", drop_id=drop_id,
            drop_name=(drop or {}).get("name"), weekly_delta=wd,
            p_success=p_ok, free_bye=bool(free_bye), is_qb=pos == "QB",
            reasons=reasons,
        ))

    # the real objective, on the survivors only
    if team_means and claims:
        base = dict(team_means)
        my_base = base.get(my_team)
        if my_base is None:
            alarms.append(
                f"{my_team!r} is not in the simulated league, so no claim carries a "
                f"title delta. Team names are mutable — resolve through team_id."
            )
        else:
            for c in claims:
                d = V.title_delta(
                    base, my_base + c.weekly_delta, my_team,
                    completed=completed, n_sims=n_sims,
                )
                c.d_title = d["d_title"]

    # §9.3: "claim this one; this one will clear — just grab him Wednesday"
    grabs = [c for c in claims if c.p_success >= 0.85 and c.weekly_delta < ov]
    for c in grabs:
        c.will_clear = True
        c.reasons.append(
            "predicted to clear unclaimed — grab him Wednesday at ZERO priority "
            "cost rather than spending a claim"
        )
    ordered = [c for c in claims if not c.will_clear]

    # order by the real objective when we have it, by the screen when we do not
    ordered.sort(
        key=lambda c: (-(c.d_title if c.d_title is not None else 0.0), -c.weekly_delta)
    )
    ordered = [c for c in ordered if c.weekly_delta >= MATERIAL_WEEKLY]

    if ordered and ordered[0].weekly_delta <= ov:
        notes.append(
            f"nothing on the wire beats the option value of holding priority "
            f"({ov:.2f} pts/wk). §9.3 warns that hoarding #1 into November is "
            f"usually a mistake in a nine-team league, so this should be rare."
        )
    if my_rank == 1 and ordered:
        notes.append(
            "holding #1 priority. Nine teams makes it cheap and it recovers fast "
            "— spend it more freely than FAAB intuition suggests (§9.3)."
        )
    qb_claims = [c for c in ordered if c.is_qb]
    if qb_claims and qb_claims[0] is not (ordered[0] if ordered else None):
        notes.append(
            f"a startable QB ({qb_claims[0].add_name}) is on the wire but is not "
            f"first by title delta. §9.3 says QB is where priority gets spent in a "
            f"2-QB league — worth a look before submitting."
        )
    return ClaimList(
        week=week, claims=ordered, free_agent_grabs=grabs, option_value=ov,
        my_waiver_rank=my_rank, notes=notes, alarms=alarms,
    )
