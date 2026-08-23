"""The §11 review, for a draft that has already happened.

Ordered by what a human actually wants after a draft: my own picks first, then
the league, then what it all implies — and the limits printed with the numbers
rather than in a footnote nobody reads.
"""

from __future__ import annotations

import polars as pl

from ff_agent.postdraft.build import PostDraft

RULE = "─" * 78


def _h(title: str) -> list[str]:
    return ["", RULE, title, RULE]


def _pick_line(r: dict, show_manager: bool = False) -> str:
    who = f"{r['manager'][:16]:<16} " if show_manager else ""
    val = r["value_vs_board"]
    sign = f"{val:+d}" if val is not None else "  ?"
    return (
        f"  {r['overall']:>3}.{r['round']:>3}  {r['grade']:<2} {who}"
        f"{r['name'][:24]:<24} {r['position']:<3} "
        f"board {r['board_rank']:>3} ({sign})  "
        f"cost {r['regret_weekly_points']:>5.1f}"
    )


def lines(pd: PostDraft) -> list[str]:
    out: list[str] = []
    rec = pd.record
    out += [
        "",
        f"POST-DRAFT ANALYSIS — {rec.season}",
        f"  {rec.summary()}",
    ]
    for n in pd.notes:
        out.append(f"  ! {n}")

    # ── my draft ────────────────────────────────────────────────────────────
    if pd.my_picks.height:
        mine = pd.ratings.filter(pl.col("manager") == pd.my_key)
        out += _h("MY DRAFT")
        if mine.height:
            m = mine.to_dicts()[0]
            out += [
                f"  {m['team']} — {m['weekly_points']:.1f} projected weekly points "
                f"(rank {m['strength_rank']} of {pd.ratings.height})",
                f"  draft GPA {m['draft_gpa']} · "
                f"{m['a_picks']} A-grade picks, {m['poor_picks']} poor · "
                f"bye cost {m['bye_cost']:.1f} pts/week · "
                f"{m['free_bye_players']} free-bye players (§2.1)",
            ]
            if m["starter_holes"]:
                out.append(f"  !! cannot fill: {', '.join(m['starter_holes'])}")
            for a in m["alarms"]:
                out.append(f"  !! §10 {a}")
        out.append("")
        for r in pd.my_picks.to_dicts():
            out.append(_pick_line(r))
            if r["better_pair"] and r["regret_weekly_points"] > 1.0:
                out.append(f"          instead: {r['better_pair']}")

    # ── the league's best and worst ─────────────────────────────────────────
    # Sorted on regret and then on value against the board: once the
    # counterfactual is working, most picks in a round tie at zero regret, and
    # "who got the best player latest" is the question that still separates them.
    out += _h("BEST PICKS OF THE DRAFT")
    for r in pd.picks.sort(
        ["regret_ratio", "value_vs_board"], descending=[False, True]
    ).head(8).to_dicts():
        out.append(_pick_line(r, show_manager=True))

    out += _h("WORST PICKS OF THE DRAFT")
    for r in pd.picks.sort("regret_ratio", descending=True).head(8).to_dicts():
        out.append(_pick_line(r, show_manager=True))
        if r["better_pair"]:
            out.append(f"          instead: {r['better_pair']}")

    # ── market timing, which is a different question from regret ────────────
    # K and D/ST are excluded from BOTH lists. The board gives them a VOR rank
    # (a top defense lands around 35-76 overall) but that rank was never meant to
    # be an ADP — §3.5 says stream both, and this league has taken exactly one
    # each per team, late, in 26/26 and 27/27 cases. Left in, every "biggest
    # reach" in the draft was a defense taken in round 12, with ECR sitting
    # calmly beside it saying the pick was normal. That is a mismatch between two
    # of our own scales, not a decision anybody made.
    skill = pd.picks.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    steals = skill.filter(pl.col("value_vs_board") > 0).sort(
        "value_vs_board", descending=True
    ).head(6)
    if steals.height:
        out += _h("BIGGEST STEALS — fell furthest below where the board had them")
        out.append("  (skill positions only; K and D/ST have no board ADP — §3.5)")
        for r in steals.to_dicts():
            out.append(_pick_line(r, show_manager=True))

    reaches = skill.filter(pl.col("value_vs_board") < 0).sort(
        "value_vs_board"
    ).head(6)
    if reaches.height:
        out += _h("BIGGEST REACHES — taken furthest ahead of the board")
        out.append(
            "  (a reach on OUR board is not automatically an error: M3 measured "
            "our model as\n   materially worse than consensus alone, which is why "
            "the board only weights it 0.12.\n   The ECR column is the "
            "cross-check.)"
        )
        for r in reaches.to_dicts():
            out.append(
                _pick_line(r, show_manager=True)
                + f"  ecr {r['ecr_rank']:>3} ({r['value_vs_ecr']:+d})"
            )

    # ── team ratings ────────────────────────────────────────────────────────
    out += _h("TEAM RATINGS")
    out.append(
        f"  {'#':<2} {'team':<24} {'wk pts':>7} {'±':>5} {'GPA':>5} "
        f"{'bye':>5} {'free':>5}  shape"
    )
    for r in pd.ratings.to_dicts():
        shape = " ".join(
            f"{p}{r[f'n_{p}']}" for p in ("QB", "RB", "WR", "TE", "DST", "K")
            if r[f"n_{p}"]
        )
        out.append(
            f"  {r['strength_rank']:<2} {r['team'][:24]:<24} "
            f"{r['weekly_points']:>7.1f} {r['weekly_points_sd']:>5.1f} "
            f"{(r['draft_gpa'] or 0):>5.2f} {r['bye_cost']:>5.1f} "
            f"{r['free_bye_players']:>5}  {shape}"
        )
        for a in r["alarms"]:
            out.append(f"       !! {a}")

    # ── projections ─────────────────────────────────────────────────────────
    if pd.standings.height:
        out += _h("PROJECTED STANDINGS AND PLAYOFF ODDS")
        out.append(
            f"  {'#':<2} {'team':<24} {'W':>5} {'PCT':>6} {'playoffs':>9} "
            f"{'top-2':>7} {'title':>7}"
        )
        for r in pd.standings.to_dicts():
            out.append(
                f"  {r['projected_finish']:<2} {r['team'][:24]:<24} "
                f"{r['mean_wins']:>5.1f} {r['mean_win_pct']:>6.3f} "
                f"{r['p_playoffs']:>8.1%} {r['p_top2_seed']:>7.1%} "
                f"{r['p_title']:>7.1%}"
            )
        cal = pd.meta["calibration"]
        out += [
            "",
            "  HOW MUCH TO TRUST THIS — measured, not estimated:",
            f"    M6 ran this exact test on 2025 and scored Spearman "
            f"{cal['preseason_projections_spearman']} against the real standings.",
            f"    A PERFECT simulator scores only "
            f"{cal['perfect_simulator_ceiling_spearman']} on it; perfect player "
            f"knowledge {cal['perfect_player_knowledge_spearman']}.",
            "    Weekly noise (sd 25.7) is ~3x the talent spread (sd 8.7), so "
            "12 games cannot resolve",
            "    team quality much further. Trust the PROBABILITIES; the "
            "finishing order is decoration.",
        ]
        ms = pd.meta.get("my_schedule") or {}
        if ms.get("faced_twice"):
            out += ["", "  §2.3 — the four managers I play TWICE (8 of my 12 games):"]
            for d in ms["faced_twice"]:
                out.append(
                    f"    {d['manager'][:20]:<20} {d['weekly_points']:>7.1f} "
                    f"(rank {d['strength_rank']})"
                )
            out.append(
                f"    their mean {ms['faced_twice_mean_strength']} vs league "
                f"{ms['league_mean_strength']}"
            )
            for m in ms.get("unmatched_double_up_managers") or []:
                out.append(f"    !! §1 lists '{m}' but no team in this draft matches")
            out.append(f"    {ms['note']}")

    out += _h("WHAT THIS CANNOT TELL YOU")
    out += [
        "  · Grades are RELATIVE, within each round. The better pair is chosen",
        "    knowing who actually survived, which no manager could. Read",
        "    regret_weekly_points for magnitude, the letter for rank.",
        "  · The counterfactual holds everyone else's picks fixed. Had you taken",
        "    a different player, the draft after you would have moved.",
        "  · §3.2: only 153 roster spots exist against 500+ relevant players, so",
        "    the wire stays rich all season. Nothing here models waivers, and 19%",
        "    of 2025's starter points came from undrafted players.",
        "",
    ]
    return out


def render(pd: PostDraft) -> str:
    return "\n".join(lines(pd))
