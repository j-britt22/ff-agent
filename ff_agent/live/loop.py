"""The draft-day loop.

§4 budgets one second per decision, so the shape is: the human types what
happened, this prints what to do. The command handling is a pure function so the
whole thing is testable without a terminal — ``handle`` takes a state and a
string and returns lines plus a mutation, and the REPL is a thin shell over it.

**§0.1 is absolute here, and this is the module where it would be violated.**
Nothing in ``ff_agent/live/`` writes to ESPN. Every pick is typed by a person and
every recommendation ends with the human clicking. ``test_live.py`` asserts the
package cannot reach a write-capable path.
"""

from __future__ import annotations

from dataclasses import dataclass

from ff_agent.draft import pool as PL
from ff_agent.live import advise as AD
from ff_agent.live.entry import Resolver
from ff_agent.live.state import DraftState

HELP = """
  <name>        record the pick on the clock       (e.g. "bijan", "SF" for D/ST)
  undo          take back the last pick
  me            my roster so far
  board [pos]   best available, optionally by position
  who           who is on the clock
  log           where this session is being written
  help / q      this / quit
""".rstrip()


@dataclass
class Reply:
    lines: list[str]
    recorded: int | None = None
    quit: bool = False
    needs_advice: bool = False


def _fmt_advice(a: AD.Advice, state: DraftState) -> list[str]:
    out = [
        "",
        f"── YOUR PICK {a.pick} (round {a.round_no}) "
        f"── {a.n_sims} sims in {a.elapsed}s ── scenario: {a.scenario}",
        f"   {a.headline()}",
        "",
    ]
    for o in a.options:
        out.append(o.line())
    if a.slipping_away:
        out.append("")
        out.append("  gone if you wait:")
        for c in a.slipping_away[:4]:
            back = c["survival_to_next_turn"]
            out.append(
                f"      {c['name'][:22]:22s} {c['position']:3s} vor {c['vor']:6.1f}"
                f"  back {'—' if back is None else f'{back:.0%}'}"
            )
    for b in a.branches:
        out.append(
            f"  IF {b['condition']} (p={b['p_condition']:.0%}): "
            f"take {b['if_true_take']} — otherwise {b['if_false_take']}"
        )
    if a.scenario_evidence.get("note"):
        out.append(f"  [{a.scenario_evidence['note']}]")
    for w in a.alarms:
        out.append(f"  !! §10 ALARM: {w}")
    if a.degraded:
        out.append(f"  !! {a.degraded}")
    out += ["", "  You click. This never submits anything. (§0.1)"]
    return out


def handle(state: DraftState, resolver: Resolver, raw: str) -> Reply:
    """One typed line -> what to print, and at most one recorded pick."""
    cmd = raw.strip()
    low = cmd.lower()

    if low in {"q", "quit", "exit"}:
        return Reply(["bye — nothing was submitted to ESPN (§0.1)"], quit=True)
    if low in {"help", "?", "h"}:
        return Reply(HELP.splitlines())
    if low == "who":
        if state.complete:
            return Reply(["the draft is complete"])
        who = state.managers[state.on_the_clock]
        mine = "  <- YOU" if state.my_turn else ""
        until = state.picks_until_my_turn()
        tail = "" if until is None else f" · your next pick in {until}"
        return Reply([f"pick {state.next_pick}: {who}{mine}{tail}"])
    if low == "me":
        names = state.pool.df["name"].to_list()
        pos = state.pool.df["position"].to_list()
        rows = [f"  {pos[i]:4s} {names[i]}" for i in state.my_roster()]
        return Reply([state.summary()] + (rows or ["  (nothing yet)"])
                     + [f"  !! {w}" for w in state.alarms()])
    if low.startswith("board"):
        parts = low.split()
        want = parts[1].upper() if len(parts) > 1 else None
        return Reply(_board(state, want))
    if low == "undo":
        try:
            i = state.undo()
        except ValueError as e:
            return Reply([str(e)])
        return Reply([f"undid pick {len(state.history) + 1}: "
                      f"{resolver.describe(i)}"], needs_advice=state.my_turn)

    if state.complete:
        return Reply(["the draft is complete — nothing left to record"])

    m = resolver.resolve(cmd, available=set(range(state.pool.n)) - state.taken())
    if m.resolved:
        state.record(m.index)
        who = state.managers[int(state.order[len(state.history) - 1])]
        return Reply(
            [f"  {len(state.history):>3}. {who}: {resolver.describe(m.index)}"],
            recorded=m.index,
            needs_advice=state.my_turn,
        )
    if m.ambiguous:
        opts = [f"  {n + 1}) {resolver.describe(i)}" for n, i in enumerate(m.candidates)]
        return Reply([f"which one? ({m.reason})"] + opts
                     + ["  type more of the name — I will not guess (§0.2)"])
    return Reply([f"no match for {cmd!r} — {m.reason}"])


def _board(state: DraftState, position: str | None) -> list[str]:
    avail = state.available_mask()
    names = state.pool.df["name"].to_list()
    idx = [i for i in range(state.pool.n) if avail[i]]
    if position:
        if position not in PL.POSITIONS:
            return [f"unknown position {position!r} — one of {list(PL.POSITIONS)}"]
        idx = [i for i in idx if PL.POSITIONS[state.pool.pos_idx[i]] == position]
    out = []
    for i in idx[:12]:
        out.append(
            f"  {PL.POSITIONS[state.pool.pos_idx[i]]:4s} {names[i][:24]:24s} "
            f"vor {state.pool.vor[i]:7.1f}"
        )
    return out or ["  nothing available"]
