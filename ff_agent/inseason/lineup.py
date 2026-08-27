"""The lineup SEQUENCE (§5.3 of the design, finding F9).

A lineup is not a decision. **ESPN locks each player at his own kickoff**, so it
is a schedule of irreversible per-slot commitments made under increasing
information — and per F9 that schedule is a different shape every week, with six
to nine distinct lock times and the worst of them in the fantasy playoffs.

**The Thursday decision is not "who is better."** Starting a Thursday player is
option-destroying: once he is in, that slot cannot respond to anything that
happens Friday, Saturday or Sunday morning. So the bar is

    start p  iff  E[p] > E[ best Sunday alternative, chosen under SUNDAY info ]

which is strictly higher, because the Sunday choice gets made knowing things
Thursday does not. It is evaluated by forcing each option and simulating the
rest — the pattern M9 used at the draft table, for the same reason: what decides
a pick is the counterfactual, not the ranking.

**A precise result about the information effect, worth stating because it is
easy to get wrong in either direction.** The design argued that a Thursday player
who *starts* reveals my partial score three days early and sharpens Sunday's
floor-versus-ceiling posture — a real argument FOR committing that the naive
"never lock early" instinct throws away. That is true, but only under a nonlinear
objective: **under expected points the information is worth exactly zero**, because
the expected-points-optimal Sunday lineup does not depend on the Thursday
realisation. It has value only under P(beat this opponent), where the lineup that
maximises the win probability depends on the margin still needed.

So this module computes the option COST exactly, and reports the information
value as identically zero under its stated objective rather than pretending to
have measured it. ``objective="win_prob"`` is where it becomes non-zero and is
what M10b-4's gate is for. Claiming both effects were measured when only one can
be, under the objective actually implemented, would be the kind of quiet
overclaim this project keeps refusing to make.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ff_agent.config import MY_BYE_WEEKS, STARTER_SLOTS
from ff_agent.inseason import availability as AV
from ff_agent.inseason import clock as CK
from ff_agent.inseason import value as V
from ff_agent.season import lineup as LU

N_DRAWS = 4000
"""Availability draws per candidate commitment. The whole decision is a handful
of options over a 15-man roster, so this is milliseconds."""

MATERIAL = 0.25
"""Points per week below which a Thursday call is a coin flip and is reported as
one. Advising a swap worth a tenth of a point is how a digest loses its reader."""


class LineupError(RuntimeError):
    pass


# ─── Lock state ──────────────────────────────────────────────────────────────
LOCKED, OPEN, AT_RISK = "locked", "open", "at_risk"


def attach_locks(
    roster: pl.DataFrame,
    week: int,
    kickoffs: pl.DataFrame,
    now: dt.datetime | None = None,
) -> pl.DataFrame:
    """Add ``kickoff`` and ``locked`` to a roster.

    A player with no game this week is on his NFL bye: he cannot lock and cannot
    start. That is different from being locked, and conflating the two would let
    a bye player be "safely" left in a slot.
    """
    now = now or CK.now_et()
    kk = kickoffs.filter(pl.col("week") == week).select(
        pl.col("team"), pl.col("kickoff"), pl.col("opponent").alias("nfl_opponent")
    )
    out = roster.join(kk, on="team", how="left")
    if out.height != roster.height:
        raise LineupError(
            f"the kickoff join changed the row count ({roster.height} -> "
            f"{out.height}). Two kickoffs for one team-week means a lock time "
            f"applied to the wrong game."
        )
    return out.with_columns(
        pl.col("kickoff").is_not_null().alias("plays_this_week"),
        (pl.col("kickoff").is_not_null() & (pl.col("kickoff") <= now)).alias("locked"),
    )


def pins_from_espn(roster: pl.DataFrame) -> dict[str, str]:
    """What ESPN has already locked, as ``canonical_id -> slot``.

    Read from ``lineup_slot``, because ESPN's lineup is the truth about what is
    spent — not our idea of what the lineup should have been.
    """
    if "lineup_slot" not in roster.columns:
        return {}
    locked = roster.filter(pl.col("locked") & pl.col("lineup_slot").is_not_null())
    out = {}
    for row in locked.iter_rows(named=True):
        slot = str(row["lineup_slot"]).upper()
        if slot in STARTER_SLOTS:
            out[row["canonical_id"]] = slot
    return out


# ─── The counterfactual ──────────────────────────────────────────────────────
def _capacity_after(pins: dict[str, str]) -> tuple[tuple[tuple[int, int], ...], int]:
    """Slot capacity left once pins are placed."""
    spent: dict[str, int] = {}
    for s in pins.values():
        spent[s] = spent.get(s, 0) + 1
    strict = tuple(
        (V.POS_INDEX[p], STARTER_SLOTS[p] - spent.get(p, 0))
        for p in STARTER_SLOTS if p != "FLEX"
    )
    n_flex = STARTER_SLOTS.get("FLEX", 0) - spent.get("FLEX", 0)
    return strict, max(0, n_flex)


def _fill(pts, pos, ok, strict, n_flex):
    """Best available into the remaining capacity. Rows pre-sorted descending."""
    starters = np.zeros(pts.shape, dtype=bool)
    for p, need in strict:
        if need <= 0:
            continue
        sel = ok & (pos == p) & ~starters
        starters |= sel & (np.cumsum(sel, axis=1) <= need)
    if n_flex > 0:
        sel = ok & np.isin(pos, V.FLEX_IDX) & ~starters
        starters |= sel & (np.cumsum(sel, axis=1) <= n_flex)
    return (pts * starters).sum(axis=1)


def commitment_value(
    roster: pl.DataFrame,
    pins: dict[str, str],
    exclude: set[str] | None = None,
    n_draws: int = N_DRAWS,
    seed: int = 17,
) -> float:
    """Expected starting points if I commit ``pins`` now and fill the rest later.

    Availability is drawn for EVERY player, pinned included — which is the whole
    reason a Questionable quarterback is such a bad Thursday commitment. A pinned
    player who does not play scores zero AND wastes the slot; a free player who
    does not play is simply passed over on Sunday. That asymmetry is the option
    cost, and it is why F10's positional split matters here rather than being a
    piece of trivia.
    """
    exclude = exclude or set()
    usable = roster.filter(
        ~pl.col("canonical_id").is_in(list(exclude)) if exclude else pl.lit(True)
    )
    if usable.is_empty():
        return 0.0

    ids = usable["canonical_id"].to_list()
    pts = usable["weekly_points"].fill_null(0.0).to_numpy().astype(float)
    pos = np.array([V.POS_INDEX[p] for p in usable["position"].to_list()])
    p_out = (
        usable["p_out"].fill_null(AV.HEALTHY).to_numpy().astype(float)
        if "p_out" in usable.columns else np.full(len(ids), AV.HEALTHY)
    )
    plays = (
        usable["plays_this_week"].fill_null(False).to_numpy()
        if "plays_this_week" in usable.columns else np.ones(len(ids), dtype=bool)
    )

    rng = np.random.default_rng(seed)
    avail = (rng.random((n_draws, len(ids))) >= p_out) & plays

    pin_idx = {c: i for i, c in enumerate(ids)}
    pinned_cols = [pin_idx[c] for c in pins if c in pin_idx]
    # A pinned player scores what he scores; if he does not play, zero — and the
    # slot stays spent, which is exactly the thing being priced.
    pinned_total = (
        (pts[pinned_cols] * avail[:, pinned_cols]).sum(axis=1)
        if pinned_cols else np.zeros(n_draws)
    )

    free_mask = np.ones(len(ids), dtype=bool)
    free_mask[pinned_cols] = False
    if not free_mask.any():
        return float(pinned_total.mean())

    fpts = np.tile(pts[free_mask], (n_draws, 1))
    fpos = np.tile(pos[free_mask], (n_draws, 1))
    fok = avail[:, free_mask]
    # Sort by EFFECTIVE points so unavailable players fall to the back and the
    # cumsum "best available" rule still means what it says.
    eff = np.where(fok, fpts, -1.0)
    order = np.argsort(-eff, axis=1)
    fpts = np.take_along_axis(fpts, order, axis=1)
    fpos = np.take_along_axis(fpos, order, axis=1)
    fok = np.take_along_axis(fok, order, axis=1)

    strict, n_flex = _capacity_after(pins)
    return float((pinned_total + _fill(fpts, fpos, fok, strict, n_flex)).mean())


@dataclass
class Decision:
    """One commit-or-wait call, with the counterfactual that decided it."""
    canonical_id: str
    name: str
    position: str
    team: str
    slot: str
    kickoff: dt.datetime
    start_value: float
    bench_value: float
    p_out: float
    forced_bench_reason: str | None = None
    """Set when no slot exists at all — a same-window teammate outranked him
    for the position and FLEX. Not a close call to weigh; there is nothing to
    start him INTO. Kept distinct from a normal BENCH so ``reason()`` can say
    that plainly instead of quoting a delta against a slot that was never on
    offer."""

    @property
    def delta(self) -> float:
        return round(self.start_value - self.bench_value, 3)

    @property
    def call(self) -> str:
        if self.forced_bench_reason:
            return "BENCH"
        if abs(self.delta) < MATERIAL:
            return "toss-up"
        return "START" if self.delta > 0 else "BENCH"

    def reason(self) -> str:
        if self.forced_bench_reason:
            return self.forced_bench_reason
        if self.call == "toss-up":
            return (
                f"within {MATERIAL} pts either way — the option you give up by "
                f"locking {self.name} in is worth about what he is"
            )
        if self.delta > 0:
            return (
                f"{self.name} beats the best Sunday alternative by {self.delta:+.2f} "
                f"pts/wk even after paying the option cost of locking the slot"
            )
        return (
            f"holding {self.slot} open is worth {-self.delta:+.2f} pts/wk more than "
            f"{self.name}"
            + (f" — he is {self.p_out:.0%} to sit" if self.p_out > 0.15 else "")
        )


@dataclass
class LineupPlan:
    week: int
    now: dt.datetime
    starters: pl.DataFrame
    bench: pl.DataFrame
    pins: dict[str, str]
    decisions: list[Decision]
    next_lock: CK.Kickoff | None
    expected_points: float
    alarms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    espn_locked: dict[str, str] = field(default_factory=dict)
    """Slots ESPN has ALREADY locked — kickoff has passed. Distinct from
    ``pins``, which also carries what we are recommending be committed. Reporting
    the two as one number told a Wednesday reader "1 locked" when nothing had
    kicked off; what was really meant was "1 we suggest you commit"."""

    @property
    def open_slots(self) -> int:
        return sum(STARTER_SLOTS.values()) - len(self.espn_locked)

    @property
    def recommended_commits(self) -> int:
        return len(self.pins) - len(self.espn_locked)

    @property
    def runway(self) -> dt.timedelta | None:
        return None if self.next_lock is None else self.next_lock.at - self.now

    def actionable(self) -> list[Decision]:
        return [d for d in self.decisions if d.call != "toss-up"]

    def summary(self) -> dict:
        return {
            "week": self.week,
            "open_slots": self.open_slots,
            "locked_slots": len(self.espn_locked),
            "recommended_commits": self.recommended_commits,
            "expected_points": round(self.expected_points, 2),
            "next_lock": None if not self.next_lock else self.next_lock.at.isoformat(),
            "minutes_to_next_lock": (
                None if self.runway is None else int(self.runway.total_seconds() // 60)
            ),
            "actionable": len(self.actionable()),
            "alarms": self.alarms,
        }


def this_week_value(roster: pl.DataFrame) -> pl.DataFrame:
    """Swap the rest-of-season rate for THIS WEEK's published projection.

    The two are different questions and the monitor asks both. "Who is worth
    more for the rest of the season" (waivers, trades) is answered by ESPN's
    season projection spread over the remaining games. "Who do I start on
    Sunday" is answered by ESPN's projection FOR SUNDAY, which carries this
    week's opponent, snap projection and injury designation — the very things
    that make it wrong as a season number and right as a weekly one.

    ``ros.from_espn`` keeps both, so this is a column swap rather than a second
    fetch. Falls back to the season rate for anyone ESPN has not published a
    weekly number for, which is most of the free-agent tail.
    """
    if "espn_projection" not in roster.columns:
        return roster
    return roster.with_columns(
        pl.coalesce(pl.col("espn_projection"), pl.col("weekly_points"))
        .alias("weekly_points")
    )


def build(
    roster: pl.DataFrame,
    week: int,
    kickoffs: pl.DataFrame,
    now: dt.datetime | None = None,
    injuries: pl.DataFrame | None = None,
    n_draws: int = N_DRAWS,
    seed: int = 17,
) -> LineupPlan:
    """The full lineup call for one checkpoint.

    ``roster`` needs canonical_id, name, position, team, weekly_points, and
    optionally lineup_slot (what ESPN currently has). Byes and locks come from
    ``kickoffs``; availability from ``injuries``.
    """
    now = now or CK.now_et()
    CK.assert_not_my_bye(week)

    r = attach_locks(this_week_value(roster), week, kickoffs, now)
    r = AV.attach(r, injuries)
    pins = pins_from_espn(r)

    alarms: list[str] = []
    notes: list[str] = []
    if injuries is None:
        notes.append(
            "no injury report available — every player treated as healthy. "
            "Availability drives the whole commit-or-wait call, so this call is "
            "weaker than it looks."
        )

    benched = r.filter(~pl.col("plays_this_week"))
    if benched.height:
        notes.append(
            f"{benched.height} player(s) on an NFL bye this week: "
            + ", ".join(benched["name"].to_list()[:6])
        )

    # Only the SINGLE SOONEST still-open kickoff faces a commit-or-wait choice,
    # and only if a LATER window exists to actually wait for.
    #
    # An earlier version used "everyone before the last game of the week" —
    # which on a normal week is almost the entire active roster, since only
    # Monday-night players are excluded. That pulled every Sunday-1pm player
    # into the same decision as the Thursday game, all evaluated at once. §5.3's
    # question is specifically "lock in NOW versus wait for MORE information" —
    # that only means something for the window coming up next. Once it locks,
    # the NEXT checkpoint asks the same question about whatever is next after
    # that, with better information than we have right now.
    open_players = r.filter(pl.col("plays_this_week") & ~pl.col("locked"))
    if open_players.is_empty():
        notes.append("every slot is already locked — nothing to decide.")
        early = open_players.head(0)
    else:
        next_kickoff = open_players["kickoff"].min()
        last_kick = open_players["kickoff"].max()
        early = (
            open_players.filter(pl.col("kickoff") == next_kickoff)
            if next_kickoff < last_kick else open_players.head(0)
        )

    # Candidates in the SAME window can compete for the SAME slot — e.g. three
    # Sunday-1pm running backs for two RB spots and one FLEX. Processing them
    # against a dict that updates as each is decided is what stops all three
    # independently being told "you'd take RB": once the first two claim it,
    # `_best_slot_for` correctly sees zero RB and offers the third only FLEX,
    # then nothing. Best-value-first, so the strongest case for locking early
    # gets first claim on a contested slot — the same "best available" rule
    # `optimal_lineup` itself uses.
    decisions: list[Decision] = []
    working_pins = dict(pins)
    ordered = early.sort("weekly_points", descending=True, nulls_last=True)
    for row in ordered.iter_rows(named=True):
        cid = row["canonical_id"]
        bench_v = commitment_value(r, working_pins, {cid}, n_draws, seed)
        best_slot = _best_slot_for(r, cid, working_pins)
        if best_slot is None:
            # No RB/FLEX capacity left once higher-value same-window players
            # claimed it. Still reported — silently dropping him from the
            # digest would look like he was never considered at all.
            decisions.append(Decision(
                canonical_id=cid, name=row.get("name") or cid,
                position=row["position"], team=row.get("team") or "",
                slot="BENCH", kickoff=row["kickoff"],
                start_value=bench_v, bench_value=bench_v,
                p_out=float(row.get("p_out") or 0.0),
                forced_bench_reason=(
                    f"no {row['position']}/FLEX slot remains once the higher-"
                    f"value players in this same kickoff window are locked in"
                ),
            ))
            continue
        start_v = commitment_value(
            r, {**working_pins, cid: best_slot}, None, n_draws, seed)
        d = Decision(
            canonical_id=cid, name=row.get("name") or cid,
            position=row["position"], team=row.get("team") or "",
            slot=best_slot, kickoff=row["kickoff"],
            start_value=start_v, bench_value=bench_v,
            p_out=float(row.get("p_out") or 0.0),
        )
        decisions.append(d)
        if d.call == "START":
            working_pins[cid] = best_slot
    decisions.sort(key=lambda d: (d.kickoff, -abs(d.delta)))

    final_pins = working_pins

    playable = r.filter(pl.col("plays_this_week"))
    lu = LU.optimal_lineup(
        playable.with_columns(
            (pl.col("weekly_points") * (1 - pl.col("p_out"))).alias("_ev")
        ),
        value="_ev",
        pinned={k: v for k, v in final_pins.items()
                if k in set(playable["canonical_id"].to_list())},
    )
    starters = lu.drop("_ev") if "_ev" in lu.columns else lu
    bench = r.filter(~pl.col("canonical_id").is_in(starters["canonical_id"].to_list()))

    for row in starters.iter_rows(named=True):
        if (row.get("p_out") or 0) >= 0.5:
            alarms.append(
                f"{row['name']} is starting at {row['slot']} and is "
                f"{row['p_out']:.0%} to sit"
            )
    expected = float(
        (starters["weekly_points"].fill_null(0.0)
         * (1 - starters["p_out"].fill_null(0.0))).sum()
    ) if starters.height else 0.0

    nxt = CK.next_lock(week, set(r["team"].drop_nulls().to_list()),
                       kickoffs=kickoffs, now=now)
    if nxt is not None and nxt.at - now < CK.MIN_RUNWAY:
        alarms.append(
            f"under {CK.MIN_RUNWAY.total_seconds() // 60:.0f} minutes to the next "
            f"lock — a recommendation that lands after its deadline is worse than "
            f"none"
        )
    return LineupPlan(
        week=week, now=now, starters=starters, bench=bench, pins=final_pins,
        espn_locked=dict(pins),
        decisions=decisions, next_lock=nxt, expected_points=expected,
        alarms=alarms, notes=notes,
    )


def _best_slot_for(roster: pl.DataFrame, cid: str, pins: dict[str, str]) -> str | None:
    """Which open slot this player would occupy, if any."""
    row = roster.filter(pl.col("canonical_id") == cid)
    if row.is_empty():
        return None
    pos = row["position"][0]
    strict, n_flex = _capacity_after(pins)
    have = {p: n for (p, n) in strict}
    if pos in STARTER_SLOTS and have.get(V.POS_INDEX[pos], 0) > 0:
        return pos
    if n_flex > 0 and pos in ("RB", "WR", "TE"):
        return "FLEX"
    return None


# ─── The information effect, stated precisely ────────────────────────────────
def information_value(objective: str = "expected_points") -> float:
    """Worth of learning my Thursday score before choosing Sunday's lineup.

    **Exactly zero under expected points**, and that is a result rather than an
    omission: the expected-points-optimal Sunday lineup is the same whatever
    Thursday did, so knowing the outcome changes no decision. It becomes positive
    only under P(beat this opponent), where the best lineup depends on the margin
    still needed — chase ceiling when behind, protect the floor when ahead (§9.2,
    and M7's measured variance crossover at a roster delta of +15.9).

    M10b-4's gate is what measures it under that objective. Until then the
    digest says zero and says why, rather than implying an effect it has not
    computed.
    """
    if objective == "expected_points":
        return 0.0
    raise NotImplementedError(
        "the win-probability objective is M10b-4's gate. Implementing it means "
        "choosing the Sunday lineup that maximises P(total > opponent), which is "
        "not a greedy best-available problem — it depends on the variance of "
        "each candidate, not only the mean."
    )
