"""The draft-day sheet — what §5's T-60 drill prints.

Two jobs, and the second is the one people forget: the terminal sheet is what I
read while on the clock, and the printed sheet is what saves the draft when the
laptop dies. §5 is explicit about both.

Formatting rules follow from being read under a 60-second clock: the concrete
names first with the probability I actually take each one, then the players who
will not come back, then only the branches that change the call. No table that
requires reading a header to interpret.
"""

from __future__ import annotations

RULE = "─" * 78


def _pct(x) -> str:
    return "  — " if x is None else f"{x * 100:3.0f}%"


def turn_block(t: dict, robust: dict | None = None) -> list[str]:
    out = [
        "",
        f"PICK {t['pick']}   (round {t['round']})"
        + (f"   next turn in {t['picks_until_next_turn']} picks"
           if t["picks_until_next_turn"] else "   LAST PICK"),
    ]
    rec = t["recommended"]
    line = f"  take {rec['modal_position']}  ({rec['concentration'] * 100:.0f}% of sims)"
    if robust is not None and not robust.get("robust", True):
        alts = {k.replace("take_if_", ""): v for k, v in robust.items()
                if k.startswith("take_if_")}
        line += "   ⚠ SCENARIO-DEPENDENT: " + ", ".join(
            f"{k}->{v}" for k, v in alts.items()
        )
    out.append(line)

    cands = t["shortlist"]["candidates"]
    if cands:
        out.append("  candidates:")
        for c in cands[:6]:
            flag = " ←TAKE NOW" if c["take_now"] else ""
            bye = " *bye5/14" if c["free_bye"] else ""
            out.append(
                f"    {_pct(c['p_i_take_him_here'])}  {c['name'][:22]:22s} "
                f"{c['position']:3s} vor {c['vor']:6.1f}  "
                f"there {_pct(c['p_available'])}{bye}{flag}"
            )
    slip = t["shortlist"]["slipping_away"]
    if slip:
        out.append("  gone if you wait:")
        for c in slip[:4]:
            out.append(
                f"          {c['name'][:22]:22s} {c['position']:3s} "
                f"vor {c['vor']:6.1f}  there {_pct(c['p_available'])}  "
                f"back {_pct(c['survival_to_next_turn'])}"
            )
    for b in t["branches"]:
        out.append(
            f"  IF {b['condition']} (p={b['p_condition']:.0%}): "
            f"take {b['if_true_take']} — otherwise {b['if_false_take']}"
        )
    return out


def render(plan: dict, max_turns: int | None = None) -> str:
    s = plan["shared"]
    lines = [
        RULE,
        f"FIRST DOWN SYNDROME — DRAFT SLOT {plan['slot']}   ({s['season']})",
        RULE,
        f"  picks: {', '.join(str(p) for p in plan['pick_numbers'])}",
        f"  gaps:  {sorted(set(plan['snake_gaps']))}   policy: {s['policy']}",
        f"  roster strength p10/p50/p90: "
        f"{plan['roster_strength']['p10']} / {plan['roster_strength']['p50']} / "
        f"{plan['roster_strength']['p90']} weekly points",
        "",
        f"  QB COUNT: take {s['qb_count_decision']['decision']}"
        f"  (runner-up {s['qb_count_decision']['runner_up']}, margin "
        f"{s['qb_count_decision']['margin_over_runner_up']:+.4f} P(title))",
    ]
    div = {r["pick"]: r for r in plan.get("scenario_divergence", [])}
    fragile = [r for r in div.values() if not r["robust"]]
    lines.append(
        f"  {len(div) - len(fragile)}/{len(div)} turns give the same answer under "
        "all three QB scenarios."
    )
    if fragile:
        lines.append(
            "  scenario-dependent turns: "
            + ", ".join(str(r["pick"]) for r in fragile)
        )

    turns = plan["per_turn"][: max_turns or len(plan["per_turn"])]
    for t in turns:
        lines += turn_block(t, div.get(t["pick"]))
    lines += ["", RULE,
              "  NEVER auto-submit. Recommend; you click. (§0.1)", RULE]
    return "\n".join(lines)
