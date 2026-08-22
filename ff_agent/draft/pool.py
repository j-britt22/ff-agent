"""The draftable pool — everything the simulator is allowed to pick.

M4's ``board.json`` covers QB/RB/WR/TE only, because the superflex ECR that
anchors it ranks skill positions and nothing else. That leaves a hole M7 cannot
ignore: **every team drafts exactly one K and one D/ST**, so 18 of 153 picks —
12% of the draft — would not exist. Availability at every one of my turns would
be wrong by a round.

So K and D/ST are added here, and deliberately built differently from the rest:

* **Ordering** comes from ESPN's own draftable pool by ``percent_owned`` — the
  crowd's revealed behaviour, not a projection.
* **Points and uncertainty** come from the M3 rank->points calibration curve,
  which already covers K and DST under §1 scoring from ten seasons of recomputed
  actuals. So a kicker's value is "what the Nth-best kicker has historically
  scored", not a model of that kicker.
* **Timing** comes from THIS LEAGUE's measured K/DST pick fractions, because no
  public ranking prices them and the league's own history is unambiguous: 26 of
  26 kickers and 27 of 27 defenses went one per team, late.

That is weaker than the skill-position projections and it is stated rather than
hidden. It is also proportionate: §3.5 says stream both weekly, which is another
way of saying the differences between them are small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ff_agent.config import N_TEAMS, ROSTER_TOTAL, SEASON
from ff_agent.projections import calibration as CAL

POSITIONS = ("QB", "RB", "WR", "TE", "DST", "K")
"""Fixed order. Every numpy array in the engine indexes positions by this."""

POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}

N_KICKERS = 20
N_DEFENSES = 20
"""How many K / D/ST to carry. Nine teams take one each; the surplus exists so
the pool never runs dry when a simulated manager reaches early."""

GAMES = 17


def total_picks(n_teams: int = N_TEAMS, rounds: int = ROSTER_TOTAL) -> int:
    return n_teams * rounds


@dataclass(frozen=True)
class DraftPool:
    """A pool sorted by VOR descending, plus the numpy views the engine uses.

    The sort is load-bearing: "the j-th best available at position p" is a
    cumulative sum over the availability mask only because rows are already in
    value order.
    """

    df: pl.DataFrame
    consensus_fraction: np.ndarray   # (n,) float32
    pos_idx: np.ndarray              # (n,) int8
    vor: np.ndarray                  # (n,) float32
    points: np.ndarray               # (n,) float32  season points
    points_sd: np.ndarray            # (n,) float32  projection uncertainty
    bye_week: np.ndarray             # (n,) int8     0 = none/unknown
    free_bye: np.ndarray             # (n,) bool
    tier: np.ndarray                 # (n,) int16

    @property
    def n(self) -> int:
        return self.df.height

    def names(self) -> list[str]:
        return self.df["name"].to_list()

    def ids(self) -> list[str]:
        return self.df["canonical_id"].to_list()


def _kdst_timing(hist: pl.DataFrame | None = None) -> dict[str, list[float]]:
    """Measured pick fraction of the r-th kicker / defense taken in this league.

    Returns one list per position, index 0 = the first one off the board. The
    league has three drafts, so these are medians over at most three values —
    stated plainly rather than dressed up.
    """
    from ff_agent.opponents import history as H

    h = H.draft_history() if hist is None else hist
    out: dict[str, list[float]] = {}
    for pos in ("K", "DST"):
        p = (
            h.filter(pl.col("position") == pos)
            .sort(["season", "overall_pick"])
            .with_columns(
                pl.col("overall_pick").rank("ordinal").over("season").cast(pl.Int32)
                .alias("nth")
            )
            .group_by("nth")
            .agg(pl.col("pick_fraction").median().alias("frac"))
            .sort("nth")
        )
        out[pos] = p["frac"].to_list()
    return out


def _extend(fracs: list[float], want: int) -> list[float]:
    """Extend a measured timing curve past the ranks history observed.

    Linear in the last observed slope, clipped just past the end of the draft:
    an undrafted kicker is undrafted, not infinitely late.
    """
    if not fracs:
        return [0.9] * want
    out = list(fracs)
    slope = (out[-1] - out[0]) / max(len(out) - 1, 1) if len(out) > 1 else 0.01
    while len(out) < want:
        out.append(min(out[-1] + max(slope, 0.005), 1.15))
    return out[:want]


def kdst_frame(
    season: int = SEASON, n_teams: int = N_TEAMS, source: str = "espn"
) -> pl.DataFrame:
    """K and D/ST rows shaped like board rows. See the module docstring.

    ``source="espn"`` orders by today's ``percent_owned`` — right for the live
    2026 pool. ``source="prior_season"`` orders by the previous season's
    recomputed §1 points instead, which is what a drafter actually knew at the
    time. Backtests must use the second: ESPN's ownership percentages are
    today's, and reading them into a 2025 draft would be lookahead of the
    laziest kind.
    """
    if source == "prior_season":
        return _kdst_from_prior_season(season, n_teams)
    if source != "espn":
        raise ValueError(f"unknown source {source!r}")
    from ff_agent.data import crosswalk as cw
    from ff_agent.data import espn

    pool = espn.draftable_players(season)
    curve = CAL.smooth_curve(CAL.rank_points_curve(seasons=list(range(2016, season))))
    timing = _kdst_timing()
    tp = total_picks(n_teams)

    frames = []
    for espn_pos, pos, want in (("K", "K", N_KICKERS), ("D/ST", "DST", N_DEFENSES)):
        rows = (
            pool.filter(pl.col("position") == espn_pos)
            .sort("percent_owned", descending=True, nulls_last=True)
            .head(want)
        )
        if rows.is_empty():
            raise RuntimeError(f"no {espn_pos} in the {season} ESPN draftable pool")

        # §0.2 applies here too: these must resolve, not be name-matched.
        if pos == "DST":
            res = cw.resolve_dst(rows)
            bad = res.filter(pl.col("match_method") == "unresolved")
            if bad.height:
                raise cw.CrosswalkError(
                    f"{bad.height} D/ST in the draft pool did not resolve:\n"
                    f"{bad.select('espn_id', 'name')}"
                )
            dup = res.group_by("canonical_id").len().filter(pl.col("len") > 1)
            if dup.height:
                raise cw.CrosswalkError(
                    f"{dup.height} D/ST resolved to a shared canonical id — the "
                    f"ESPN pool listed a defense twice:\n{dup}"
                )
            ids = res["canonical_id"].to_list()
            teams = rows["team"].to_list()
        else:
            res = cw.resolve_players(rows)
            cw.assert_all_resolved(res, f"draft_pool_{pos}_{season}")
            ids = res["canonical_id"].to_list()
            teams = rows["team"].to_list()

        c = curve.filter(pl.col("position") == pos).sort("pos_rank")
        pts = c["expected_points_smooth"].to_list()
        sds = c["sd_points"].to_list()
        repl = float(pts[n_teams - 1]) if len(pts) >= n_teams else float(pts[-1])
        fr = _extend(timing.get(pos, []), want)

        frames.append(pl.DataFrame({
            "canonical_id": ids,
            "name": rows["name"].to_list(),
            "position": [pos] * rows.height,
            "team": teams,
            "blended_points": [float(pts[min(i, len(pts) - 1)]) for i in range(rows.height)],
            "points_sd": [float(sds[min(i, len(sds) - 1)]) for i in range(rows.height)],
            "vor": [float(pts[min(i, len(pts) - 1)]) - repl for i in range(rows.height)],
            "consensus_fraction": [float(fr[i]) for i in range(rows.height)],
            "ecr": [float(fr[i] * tp) for i in range(rows.height)],
            "tier": pl.Series([0] * rows.height, dtype=pl.Int32),
            "value_source": ["rank_calibrated"] * rows.height,
        }))

    # K and D/ST have NFL byes like anyone else. Leaving them at "no bye" would
    # quietly hand every simulated roster a free week at two starting slots.
    from ff_agent.data import byes as BY

    # nflverse spells the Rams "LA"; ESPN and the crosswalk use "LAR". Joining
    # raw abbreviations silently hands the Rams K and D/ST a 17-week season.
    bw = BY.bye_weeks(season).select(
        pl.col("team").map_elements(cw.normalize_team, return_dtype=pl.Utf8).alias("team"),
        pl.col("bye_week").cast(pl.Int32),
        pl.col("free_bye"),
    )
    out = (
        pl.concat(frames)
        .with_columns(
            pl.col("team").map_elements(cw.normalize_team, return_dtype=pl.Utf8).alias("team")
        )
        .join(bw, on="team", how="left")
        .with_columns(
            pl.col("bye_week").fill_null(0).cast(pl.Int32),
            pl.col("free_bye").fill_null(False).alias("free_bye_week_5_or_14"),
        )
        .drop("free_bye")
    )
    missing = out.filter(pl.col("team").is_not_null() & (pl.col("bye_week") == 0))
    if missing.height:
        raise RuntimeError(
            f"{missing.height} K/D-ST on a real NFL team have no bye week — a team "
            f"abbreviation did not join:\n{missing.select('name', 'team', 'position')}"
        )
    return out


def _kdst_from_prior_season(season: int, n_teams: int) -> pl.DataFrame:
    """K and D/ST ordered by LAST season's production under §1 scoring."""
    from ff_agent.data import byes as BY
    from ff_agent.data import crosswalk as cw
    from ff_agent.projections import actuals as A

    prior = season - 1
    curve = CAL.smooth_curve(CAL.rank_points_curve(seasons=list(range(2016, season))))
    timing = _kdst_timing()
    tp = total_picks(n_teams)

    frames = []
    for pos, want, fetch in (
        ("K", N_KICKERS, lambda: A.player_season_actuals(prior).filter(
            pl.col("position") == "K")),
        ("DST", N_DEFENSES, lambda: A.dst_season_actuals(prior)),
    ):
        rows = fetch().sort("points", descending=True, nulls_last=True).head(want)
        if rows.is_empty():
            raise RuntimeError(f"no {pos} production in {prior}")
        if "team" not in rows.columns:
            rows = rows.join(
                cw.canonical_players().select("canonical_id", "team"),
                on="canonical_id", how="left",
            )
        c = curve.filter(pl.col("position") == pos).sort("pos_rank")
        pts = c["expected_points_smooth"].to_list()
        sds = c["sd_points"].to_list()
        repl = float(pts[min(n_teams - 1, len(pts) - 1)])
        fr = _extend(timing.get(pos, []), want)
        names = (
            rows["name"].to_list() if "name" in rows.columns
            else [f"{t} D/ST" for t in rows["team"].to_list()]
        )
        frames.append(pl.DataFrame({
            "canonical_id": rows["canonical_id"].to_list(),
            "name": names,
            "position": [pos] * rows.height,
            "team": rows["team"].to_list(),
            "blended_points": [float(pts[min(i, len(pts) - 1)]) for i in range(rows.height)],
            "points_sd": [float(sds[min(i, len(sds) - 1)]) for i in range(rows.height)],
            "vor": [float(pts[min(i, len(pts) - 1)]) - repl for i in range(rows.height)],
            "consensus_fraction": [float(fr[i]) for i in range(rows.height)],
            "ecr": [float(fr[i] * tp) for i in range(rows.height)],
            "tier": pl.Series([0] * rows.height, dtype=pl.Int32),
            "value_source": ["rank_calibrated_prior_season"] * rows.height,
        }))

    bw = BY.bye_weeks(season).select(
        pl.col("team").map_elements(cw.normalize_team, return_dtype=pl.Utf8).alias("team"),
        pl.col("bye_week").cast(pl.Int32),
        pl.col("free_bye"),
    )
    return (
        pl.concat(frames)
        .with_columns(
            pl.col("team").map_elements(cw.normalize_team, return_dtype=pl.Utf8).alias("team")
        )
        .join(bw, on="team", how="left")
        .with_columns(
            pl.col("bye_week").fill_null(0).cast(pl.Int32),
            pl.col("free_bye").fill_null(False).alias("free_bye_week_5_or_14"),
        )
        .drop("free_bye")
        # maintain_order: an unordered unique makes the whole sim irreproducible
        .unique(subset=["canonical_id"], keep="first", maintain_order=True)
    )


def build_pool(
    season: int = SEASON,
    n_teams: int = N_TEAMS,
    board: pl.DataFrame | None = None,
    top_n: int | None = None,
    kdst_source: str = "espn",
) -> DraftPool:
    """Board (QB/RB/WR/TE) + K/DST, sorted by VOR, as numpy-ready arrays."""
    from ff_agent.board import build as BB

    b = BB.build(season) if board is None else board
    tp = total_picks(n_teams)

    skill = b.select(
        "canonical_id", "name", "position", "team", "blended_points", "points_sd",
        "vor", "ecr", "tier", "bye_week", "free_bye_week_5_or_14",
    ).with_columns(
        pl.col("tier").cast(pl.Int32),
        pl.col("bye_week").cast(pl.Int32),
        (pl.col("ecr") / tp).cast(pl.Float64).alias("consensus_fraction"),
        pl.lit("projected").alias("value_source"),
    )
    if top_n:
        skill = skill.sort("vor", descending=True, nulls_last=True).head(top_n)

    cols = [
        "canonical_id", "name", "position", "team", "blended_points", "points_sd",
        "vor", "ecr", "tier", "bye_week", "free_bye_week_5_or_14",
        "consensus_fraction", "value_source",
    ]
    df = (
        pl.concat([skill.select(cols), kdst_frame(season, n_teams, kdst_source).select(cols)])
        .with_columns(
            pl.col("bye_week").fill_null(0).cast(pl.Int8),
            pl.col("free_bye_week_5_or_14").fill_null(False),
            pl.col("tier").fill_null(0).cast(pl.Int16),
            pl.col("points_sd").fill_null(0.0),
        )
        .sort("vor", descending=True, nulls_last=True)
    )
    return _to_pool(df)


def _to_pool(df: pl.DataFrame) -> DraftPool:
    # §0.2 again, at the last point before the simulator. The engine marks pool
    # INDICES taken, not people, so a duplicated canonical_id lets two fantasy
    # teams draft the same player in the same simulated draft.
    dupes = df.group_by("canonical_id").len().filter(pl.col("len") > 1)
    if dupes.height:
        raise ValueError(
            f"{dupes.height} canonical_id(s) appear more than once in the draft "
            f"pool. The simulator would let two teams draft the same person.\n"
            f"{df.filter(pl.col('canonical_id').is_in(dupes['canonical_id'].implode())).select('canonical_id', 'name', 'team', 'position')}"
        )
    unknown = sorted(set(df["position"].to_list()) - set(POSITIONS))
    if unknown:
        raise ValueError(
            f"pool contains positions the engine has no slot for: {unknown}. "
            "Fantasy position, not nflverse position — Travis Hunter is a CB in "
            "nflverse and a WR in ESPN, and the ESPN answer is the one that drafts."
        )
    return DraftPool(
        df=df,
        consensus_fraction=df["consensus_fraction"].fill_null(1.5).to_numpy().astype(np.float32),
        pos_idx=np.array([POS_INDEX[p] for p in df["position"].to_list()], dtype=np.int8),
        vor=df["vor"].fill_null(-999.0).to_numpy().astype(np.float32),
        points=df["blended_points"].fill_null(0.0).to_numpy().astype(np.float32),
        points_sd=df["points_sd"].fill_null(0.0).to_numpy().astype(np.float32),
        bye_week=df["bye_week"].to_numpy().astype(np.int8),
        free_bye=df["free_bye_week_5_or_14"].to_numpy().astype(bool),
        tier=df["tier"].to_numpy().astype(np.int16),
    )
