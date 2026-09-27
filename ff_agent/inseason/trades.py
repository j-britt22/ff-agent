"""§9.4 — the two-sided trade search, and profiling the other eight rosters.

**The edge is the wedge between two value functions.** Their side is scored under
*their* numbers — public consensus, which is what the other eight managers price
off — and my side under *mine*: the ESPN points anchor plus the corrections
§1's scoring makes real and no public source contains. A trade is proposed only
when it is positive under BOTH, because a trade nobody would accept is not a
recommendation.

And every proposal must **name its wedge**. *"This works because our board
charges -1 per sack and consensus charges zero"* is a reason; *"our model likes
him"* is not. A proposal whose rationale cannot be stated in one sentence is
probably an artefact of the search rather than a disagreement about football.

Three things §9.4 asks for that the structure gives without special cases:

  * **Consolidation.** "Depth is cheap in a nine-team league, so consolidating
    two mid pieces into one stud is usually favourable." That falls out of the
    lineup solver — two bench-quality players contribute zero and one starter
    contributes — rather than needing a rule.
  * **§2.3's double-up penalty.** Four managers are faced twice, so strengthening
    one costs me twice. Running both sides through the season simulator prices
    that automatically, because their higher mean runs through two of my twelve
    games instead of one. Flagged in words as well, because it changes whether
    the offer should be sent at all.
  * **Playoff weighting.** From around week 9 the weeks 15-17 view starts to
    matter more than the regular-season one (§2.4).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import polars as pl

from ff_agent.config import DOUBLE_UP_MANAGERS, MY_TEAM_NAME, POSITION_MAXIMA
from ff_agent.inseason import state as ST
from ff_agent.inseason import value as V

SCREEN_KEEP = 20
"""Candidates surviving the cheap lineup screen and reaching the real simulator.

Roughly 1,800 one-for-ones and 1,500 two-for-ones per opponent is far too many
for a 20,000-season simulation and exactly right for a lineup delta. Same
two-stage shape M7 used, applied where it actually belongs."""

MIN_WEEKLY_GAIN = 0.5
"""Points per week below which a trade is not worth the social cost of asking."""

MAX_PACKAGE = 2
"""1-for-1, 2-for-1 and 1-for-2. Beyond that the search explodes and the
proposals stop being things a human would actually send."""


@dataclass
class Wedge:
    """Why our number and consensus disagree about a player, in words."""
    player: str
    points: float
    reasons: list[str]

    def __str__(self) -> str:
        return f"{self.player} ({self.points:+.1f} pts): " + "; ".join(self.reasons)


@dataclass
class Proposal:
    partner: str
    partner_manager: str | None
    i_give: list[str]
    i_get: list[str]
    my_weekly_delta: float
    their_weekly_delta: float
    my_wedges: list[Wedge]
    double_up: bool
    d_title: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        """Positive for both sides, under each side's own value function."""
        return self.my_weekly_delta > 0 and self.their_weekly_delta > 0

    def message(self) -> str:
        """Copy-pasteable. I send it; §0.1 means nothing here sends it for me."""
        give = ", ".join(self.i_give)
        get = ", ".join(self.i_get)
        return (
            f"Hey — would you do {give} for {get}? "
            f"Looks like it helps us both: you'd be up about "
            f"{self.their_weekly_delta:.1f} a week on your starting lineup."
        )

    def summary(self) -> dict:
        return {
            "partner": self.partner,
            "i_give": self.i_give, "i_get": self.i_get,
            "my_gain_per_week": round(self.my_weekly_delta, 2),
            "their_gain_per_week": round(self.their_weekly_delta, 2),
            "d_title": None if self.d_title is None else round(self.d_title, 4),
            "double_up_opponent": self.double_up,
            "wedge": [str(w) for w in self.my_wedges],
            "notes": self.notes,
            "message": self.message(),
        }


# ─── Profiling the other eight rosters ───────────────────────────────────────
def profile(
    roster: pl.DataFrame,
    play_weeks: tuple[int, ...],
    team: str,
    manager: str | None = None,
) -> dict:
    """One manager's shape: strength, surplus and deficit in STARTING terms.

    Surplus is defined against the starting lineup rather than against a
    positional average — a third good running back on a team starting two plus a
    flex is surplus, and a second quarterback hole in a 2-QB league is acute.
    """
    counts: dict[str, int] = {}
    for p in roster["position"].to_list():
        counts[p] = counts.get(p, 0) + 1
    val = V.roster_value(roster, play_weeks)

    starters = set()
    from ff_agent.season.lineup import optimal_lineup
    lu = optimal_lineup(roster)
    starters = set(lu["canonical_id"].to_list())
    surplus = roster.filter(~pl.col("canonical_id").is_in(list(starters)))

    return {
        "team": team, "manager": manager,
        "weekly_points": round(val.mean, 2),
        "bye_cost": round(val.bye_cost, 2),
        "counts": counts,
        "holes": ST.starter_holes(counts),
        "surplus": surplus.sort("weekly_points", descending=True),
        "double_up": manager in DOUBLE_UP_MANAGERS if manager else False,
    }


def league_profile(
    rosters: dict[str, pl.DataFrame],
    play_weeks_by_team: dict[str, tuple[int, ...]],
    managers: dict[str, str] | None = None,
) -> pl.DataFrame:
    """The Monday report even when no trade is found."""
    rows = []
    for team, r in rosters.items():
        p = profile(r, play_weeks_by_team.get(team, ()), team,
                    (managers or {}).get(team))
        rows.append({
            "team": p["team"], "manager": p["manager"],
            "weekly_points": p["weekly_points"], "bye_cost": p["bye_cost"],
            "holes": ", ".join(p["holes"]) or "-",
            "n_surplus": p["surplus"].height,
            "double_up": p["double_up"],
        })
    if not rows:
        return pl.DataFrame(schema={"team": pl.Utf8, "weekly_points": pl.Float64})
    return pl.DataFrame(rows).sort("weekly_points", descending=True)


# ─── Naming the wedge ────────────────────────────────────────────────────────
def wedges(players: pl.DataFrame, free_bye_weeks: frozenset[int] | set[int]) -> list[Wedge]:
    """Why our number differs from consensus, per player, in plain words.

    Only reasons with a MEASURED basis appear. "Our model likes him" is not a
    wedge; "consensus prices sacks at zero and this league charges one point
    each, and sacks-over-expected persists at 0.434" is.
    """
    out: list[Wedge] = []
    for row in players.iter_rows(named=True):
        reasons: list[str] = []
        anchor = row.get("anchor_points")
        ours = row.get("ros_points")
        sack = row.get("sack_correction") or 0.0
        kick = row.get("kicker_correction") or 0.0

        if abs(sack) > 0.5:
            reasons.append(
                f"§3.3's sack term ({sack:+.1f} pts) — this league charges -1 per "
                f"sack taken and effectively no other league does, so no public "
                f"ranking prices it"
            )
        if abs(kick) > 0.5:
            reasons.append(
                f"§1 pays 6 for a 60+ field goal and 5 for 50-59; ESPN's own "
                f"projection merges both into one bucket ({kick:+.1f} pts)"
            )
        if row.get("bye_week") in free_bye_weeks:
            reasons.append(
                f"NFL bye in week {row['bye_week']}, which is one of MY fantasy "
                f"byes — free to me, a lost start for most of the league (§2.1)"
            )
        if reasons:
            gap = (ours - anchor) if (ours is not None and anchor is not None) else 0.0
            out.append(Wedge(row.get("name") or row["canonical_id"], gap, reasons))
    return out


# ─── The search ──────────────────────────────────────────────────────────────
def _legal(roster: pl.DataFrame, give: tuple, get_rows: list[dict]) -> bool:
    counts: dict[str, int] = {}
    for p in roster["position"].to_list():
        counts[p] = counts.get(p, 0) + 1
    for g in give:
        counts[g["position"]] = counts.get(g["position"], 0) - 1
    for g in get_rows:
        counts[g["position"]] = counts.get(g["position"], 0) + 1
    for pos, n in counts.items():
        cap = POSITION_MAXIMA.get(pos)
        if cap is not None and n > cap:
            return False
    return not ST.starter_holes(counts)


def _their_values(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Re-score a roster under the OTHER manager's value function.

    They price off public consensus, so their weekly value is the anchor without
    our league-specific corrections. This is the mechanism, not a convenience: if
    both sides used our numbers there would be no trade to find, and if both used
    consensus there would be no edge in finding one.
    """
    if col not in df.columns or col == "weekly_points":
        return df
    return df.drop("weekly_points").rename({col: "weekly_points"})


def _apply(roster: pl.DataFrame, give_ids: set[str], get_rows: list[dict]) -> pl.DataFrame:
    kept = roster.filter(~pl.col("canonical_id").is_in(list(give_ids)))
    incoming = pl.DataFrame(get_rows).select(roster.columns)
    return pl.concat([kept, incoming])


def search_partner(
    my_roster: pl.DataFrame,
    their_roster: pl.DataFrame,
    my_play_weeks: tuple[int, ...],
    their_play_weeks: tuple[int, ...],
    partner: str,
    manager: str | None = None,
    their_value_col: str = "anchor_weekly",
    max_package: int = MAX_PACKAGE,
    keep: int = SCREEN_KEEP,
) -> list[Proposal]:
    """Every package with one manager, screened on lineup delta.

    ``their_value_col`` is how THEY see value — the consensus anchor, without our
    league-specific corrections. That is not a modelling convenience; it is the
    mechanism. If both sides used our numbers there would be no trade to find,
    and if both used consensus there would be no edge in finding one.
    """
    mine = my_roster.to_dicts()
    theirs = their_roster.to_dicts()
    props: list[Proposal] = []

    for n_give in range(1, max_package + 1):
        for n_get in range(1, max_package + 1):
            if n_give == n_get == max_package:
                continue                      # 2-for-2 adds little and costs a lot
            for give in itertools.combinations(mine, n_give):
                for get in itertools.combinations(theirs, n_get):
                    give_ids = {g["canonical_id"] for g in give}
                    get_ids = {g["canonical_id"] for g in get}
                    if not _legal(my_roster, give, list(get)):
                        continue
                    if not _legal(their_roster, get, list(give)):
                        continue

                    my_after = _apply(my_roster, give_ids, list(get))
                    my_d = V.weekly_delta(my_roster, my_after, my_play_weeks)
                    if my_d < MIN_WEEKLY_GAIN:
                        continue

                    # their side, under THEIR value function
                    their_after = _apply(their_roster, get_ids, list(give))
                    their_d = V.weekly_delta(
                        _their_values(their_roster, their_value_col),
                        _their_values(their_after, their_value_col),
                        their_play_weeks,
                    )
                    if their_d <= 0:
                        continue

                    incoming = their_roster.filter(
                        pl.col("canonical_id").is_in(list(get_ids)))
                    props.append(Proposal(
                        partner=partner, partner_manager=manager,
                        i_give=[g.get("name") or g["canonical_id"] for g in give],
                        i_get=[g.get("name") or g["canonical_id"] for g in get],
                        my_weekly_delta=my_d, their_weekly_delta=their_d,
                        my_wedges=wedges(incoming, _FREE_BYES),
                        double_up=manager in DOUBLE_UP_MANAGERS if manager else False,
                    ))
    props.sort(key=lambda p: -p.my_weekly_delta)
    return props[:keep]


from ff_agent.config import FREE_BYE_WEEKS as _FREE_BYES


def build(
    my_roster: pl.DataFrame,
    rosters: dict[str, pl.DataFrame],
    play_weeks_by_team: dict[str, tuple[int, ...]],
    managers: dict[str, str] | None = None,
    my_team: str = MY_TEAM_NAME,
    team_means: dict[str, float] | None = None,
    completed: pl.DataFrame | None = None,
    n_sims: int = 8000,
    keep: int = 6,
) -> tuple[list[Proposal], list[str]]:
    """The Monday report: proposals plus the notes that qualify them."""
    notes: list[str] = []
    all_props: list[Proposal] = []
    my_weeks = play_weeks_by_team.get(my_team, ())

    for team, r in rosters.items():
        if team == my_team:
            continue
        all_props += search_partner(
            my_roster, r, my_weeks, play_weeks_by_team.get(team, my_weeks),
            partner=team, manager=(managers or {}).get(team),
        )

    all_props.sort(key=lambda p: -p.my_weekly_delta)
    short = all_props[:keep]

    # the real objective, on the survivors, with BOTH sides moved
    if team_means and short:
        my_base = team_means.get(my_team)
        for p in short:
            if my_base is None:
                break
            also = None
            if p.partner in team_means:
                also = {p.partner: team_means[p.partner] + p.their_weekly_delta}
            d = V.title_delta(
                team_means, my_base + p.my_weekly_delta, my_team,
                also=also, completed=completed, n_sims=n_sims,
            )
            p.d_title = d["d_title"]
            if p.double_up:
                p.notes.append(
                    f"{p.partner_manager} is one of my four DOUBLE-UP opponents "
                    f"(§2.3) — anything this trade gives them costs me twice, in "
                    f"two of my twelve games. The title delta beside this already "
                    f"charges for it."
                )
            if not p.my_wedges:
                p.notes.append(
                    "no league-specific wedge behind this one — it rests on our "
                    "projection disagreeing with consensus for reasons we cannot "
                    "name, which is usually a search artefact rather than a trade."
                )
    if not short:
        notes.append(
            "no package clears the bar on both sides. §9.4's structural bet still "
            "holds — depth is cheap in a nine-team league, so consolidating two "
            "mid pieces into one stud is the shape to keep looking for."
        )
    short.sort(key=lambda p: -(p.d_title if p.d_title is not None else 0.0))
    return short, notes
