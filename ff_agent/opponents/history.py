"""League draft history — three seasons that are not the same game.

The league changed shape underneath itself, and M5 is mostly about not being
fooled by that:

* **2023 and 2024 were ONE-QB leagues; only 2025 was 2-QB.** The first
  quarterback went at pick 16, then 35, then **7**. A 28-pick swing. Anything
  learned about QB timing before 2025 is not thin evidence, it is wrong evidence.
* Team counts ran 8, 10, 8 and now **9** — a size this league has never played.
  Pick numbers are therefore normalised to a FRACTION of the draft.
* Scoring changed too: completions and incompletions were dropped after 2024 (M2).

So 2025 is the only season that resembles 2026, and it is a single 136-pick draft.
"""

from __future__ import annotations

import polars as pl

from ff_agent.data import crosswalk as cw
from ff_agent.data.cache import cached

TWO_QB_SEASONS = frozenset({2025, 2026})
"""Seasons started 2 QBs. Before this, QB draft behaviour is from a different game."""

HISTORY_SEASONS = (2023, 2024, 2025)

MANAGER_ALIASES: dict[str, str] = {
    # FANTASY_SPEC.md §1 spelling -> the owner name ESPN returns
    "R. Sharrett": "Rayne Sharrett",
}
"""Explicit alias map. §0.2's no-name-matching rule is about player IDs, but the
same discipline applies here: managers are matched by a recorded decision, never
by fuzzy similarity."""


def canonical_manager(name: str | None) -> str | None:
    if not name:
        return None
    n = " ".join(str(name).split())
    return MANAGER_ALIASES.get(n, n)


def _owner_names(team) -> list[str]:
    out = []
    for o in (getattr(team, "owners", None) or []):
        if isinstance(o, dict):
            nm = (o.get("firstName", "") + " " + o.get("lastName", "")).strip()
        else:
            nm = str(o)
        if nm:
            out.append(nm)
    return out


def draft_history(seasons: tuple[int, ...] = HISTORY_SEASONS, **kw) -> pl.DataFrame:
    """Every historical pick, with manager, position and normalised timing."""

    def fetch() -> pl.DataFrame:
        from ff_agent.data.espn import get_league

        rows: list[dict] = []
        for yr in seasons:
            lg = get_league(yr)
            n_teams = len(getattr(lg, "teams", []) or [])
            owner_by_team = {}
            for t in lg.teams:
                names = _owner_names(t)
                owner_by_team[t.team_id] = canonical_manager(names[0]) if names else None
            picks = getattr(lg, "draft", None) or []
            total = len(picks)
            for pk in picks:
                team = getattr(pk, "team", None)
                tid = getattr(team, "team_id", None) if team else None
                overall = (pk.round_num - 1) * n_teams + pk.round_pick
                rows.append({
                    "season": yr,
                    "n_teams": n_teams,
                    "total_picks": total,
                    "round": pk.round_num,
                    "round_pick": pk.round_pick,
                    "overall_pick": overall,
                    "manager": owner_by_team.get(tid),
                    "fantasy_team": getattr(team, "team_name", None) if team else None,
                    "espn_id": str(getattr(pk, "playerId", "") or ""),
                    "name": getattr(pk, "playerName", None),
                })
        if not rows:
            raise RuntimeError("no draft history returned")
        return pl.DataFrame(rows)

    df = cached("draft_history", fetch, source="espn", **kw)

    canon = cw.canonical_players().select(
        pl.col("espn_id"), pl.col("position").alias("_pos")
    )
    out = (
        df.join(canon, on="espn_id", how="left")
        .with_columns(
            pl.when(pl.col("espn_id").str.starts_with("-"))
            .then(pl.lit("DST")).otherwise(pl.col("_pos")).alias("position")
        )
        .drop("_pos")
        .with_columns(
            # comparable across 8-, 9- and 10-team drafts
            (pl.col("overall_pick") / pl.col("total_picks")).alias("pick_fraction"),
            pl.col("season").is_in(list(TWO_QB_SEASONS)).alias("two_qb_format"),
        )
        .filter(pl.col("manager").is_not_null())
        .sort(["season", "overall_pick"])
    )
    _warn_non_fantasy_positions(out)
    return out


FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


def _warn_non_fantasy_positions(df: pl.DataFrame) -> pl.DataFrame:
    """Surface picks whose position is nflverse's, not ESPN's fantasy one.

    The position here comes from a raw join to ``canonical_players``, so it is
    the nflverse label. Travis Hunter is a CB in nflverse and a WR in ESPN — the
    ID match is correct (M1 confirmed that) but the POSITION is not, and every
    positional aggregate silently drops him: he still counts toward the phase
    total in ``draft/calibrate.target_rates`` while contributing to no position,
    so the fitted offsets are diluted.

    One pick in 424 today. Reported rather than corrected, because guessing a
    fantasy position from an nflverse one is exactly the fuzzy matching §0.2
    forbids — the fix is an explicit override row when it matters.
    """
    odd = df.filter(
        pl.col("position").is_null() | ~pl.col("position").is_in(list(FANTASY_POSITIONS))
    )
    if odd.height:
        import warnings

        names = odd.select("season", "name", "position").to_dicts()
        warnings.warn(
            f"{odd.height}/{df.height} draft picks carry a non-fantasy position "
            f"(nflverse label, not ESPN's). They are dropped from every "
            f"positional aggregate: {names}",
            stacklevel=3,
        )
    return odd


def manager_coverage(hist: pl.DataFrame | None = None) -> pl.DataFrame:
    """How much evidence exists per manager — and how much of it is relevant.

    ``n_drafts_two_qb`` is the number that matters. §2.3's four double-up
    managers deserve the most weight and several of them have the least data.
    """
    h = draft_history() if hist is None else hist
    from ff_agent.config import DOUBLE_UP_MANAGERS

    dbl = {canonical_manager(m) for m in DOUBLE_UP_MANAGERS}
    return (
        h.group_by("manager")
        .agg(
            pl.col("season").n_unique().alias("n_drafts"),
            pl.len().alias("n_picks"),
            pl.col("season").filter(pl.col("two_qb_format")).n_unique().alias("n_drafts_two_qb"),
            pl.col("two_qb_format").sum().alias("n_picks_two_qb"),
            pl.col("season").unique().sort().alias("seasons"),
        )
        .with_columns(
            pl.col("manager").is_in(list(dbl)).alias("faced_twice"),
            pl.when(pl.col("n_drafts_two_qb") >= 2).then(pl.lit("moderate"))
            .when(pl.col("n_drafts_two_qb") == 1).then(pl.lit("low — one 2-QB draft"))
            .otherwise(pl.lit("none — no 2-QB evidence"))
            .alias("qb_confidence"),
        )
        .sort(["faced_twice", "n_picks"], descending=[True, True])
    )


def format_shift_evidence(hist: pl.DataFrame | None = None) -> pl.DataFrame:
    """Quantifies why pre-2025 QB behaviour must not be trained on."""
    h = draft_history() if hist is None else hist
    return (
        h.filter(pl.col("position") == "QB")
        .group_by("season", "two_qb_format", "n_teams", "total_picks")
        .agg(
            pl.len().alias("qbs_drafted"),
            pl.col("overall_pick").min().alias("first_qb_pick"),
            pl.col("pick_fraction").min().round(3).alias("first_qb_fraction"),
            pl.col("pick_fraction").median().round(3).alias("median_qb_fraction"),
        )
        .sort("season")
    )
