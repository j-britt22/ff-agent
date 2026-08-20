"""The Milestone 2 test: recomputed scores must match ESPN exactly (§7.1, §11).

Three layers, deliberately separable, because "the total is off by 1.3" is not a
debuggable statement:

  A. **rules**       score ESPN's OWN stat line with our rules -> ESPN's points.
                     Isolates a wrong rule from a wrong stat.
  B. **end to end**  score the NFLVERSE stat line -> ESPN's points.
                     This is the actual §11 milestone test.
  C. **per category** compare our ``pts_<ABBR>`` against ESPN's
                     ``points_breakdown``, which attributes points category by
                     category, so a failure names the broken rule.

If A passes and B fails, the bug is in the nflverse<->ESPN stat mapping rather
than in the scoring rules.
"""

from __future__ import annotations

import json

import polars as pl

from ff_agent.config import REPORTS_DIR
from ff_agent.data import crosswalk as cw
from ff_agent.data.cache import cached
from ff_agent.scoring.engine import score_dst, score_players
from ff_agent.scoring.rules import load_rules

# ESPN's points_breakdown key -> our rule abbreviation.
# "198" is ESPN's raw stat id for 50-59 yard field goals; espn_api has no name
# for it, so it arrives as a bare numeric key.
ESPN_BREAKDOWN_TO_RULE: dict[str, str] = {
    "passingYards": "PY", "passingTouchdowns": "PTD",
    "passingInterceptions": "INTT", "passingTimesSacked": "SKD",
    "passing2PtConversions": "2PC",
    # retired after 2024, but still needed to validate 2023-24 (see rules.SPEC_SEASONS)
    "passingCompletions": "PC", "passingIncompletions": "INC",
    "rushingYards": "RY", "rushingAttempts": "RA",
    "rushingTouchdowns": "RTD", "rushing2PtConversions": "2PR",
    "receivingYards": "REY", "receivingReceptions": "REC",
    "receivingTouchdowns": "RETD", "receiving2PtConversions": "2PRE",
    "lostFumbles": "FUML", "fumbleRecoveredForTD": "FTD", "fumbleRecoveredForTouchdowns": "FTD",
    "kickoffReturnTouchdowns": "KRTD", "puntReturnTouchdowns": "PRTD",
    "madeExtraPoints": "PAT", "missedFieldGoals": "FGM",
    "madeFieldGoalsFromUnder40": "FG0", "madeFieldGoalsFrom40To49": "FG40",
    "198": "FG50", "madeFieldGoalsFrom50To59": "FG50",
    "madeFieldGoalsFrom60Plus": "FG60",
    "defensiveSacks": "SK", "defensiveInterceptions": "INT",
    "defensiveFumbles": "FR", "defensiveSafeties": "SF",
    "defensiveBlockedKicks": "BLKK",
    "defensiveBlockedKickForTouchdowns": "BLKKRTD",
    "fumbleReturnTouchdowns": "FRTD",
    "interceptionReturnTouchdowns": "INTTD",
    "defensive0PointsAllowed": "PA0", "defensive1To6PointsAllowed": "PA1",
    "defensive7To13PointsAllowed": "PA7", "defensive14To17PointsAllowed": "PA14",
    "defensive28To34PointsAllowed": "PA28", "defensive35To45PointsAllowed": "PA35",
    "defensive46PlusPointsAllowed": "PA46",
    "defensiveLessThan100YardsAllowed": "YA100",
    "defensive100To199YardsAllowed": "YA199",
    "defensive200To299YardsAllowed": "YA299",
    "defensive350To399YardsAllowed": "YA399",
    "defensive400To449YardsAllowed": "YA449",
    "defensive450To499YardsAllowed": "YA499",
    "defensive500To549YardsAllowed": "YA549",
    "defensive550PlusYardsAllowed": "YA550",
}

TOLERANCE = 0.005
"""ESPN reports to 2dp. Anything above half a hundredth is a real disagreement,
not float noise."""

RULE_ABBRS: tuple[str, ...] = tuple(sorted(set(ESPN_BREAKDOWN_TO_RULE.values())))

DST_ONLY_RULES: frozenset[str] = frozenset({
    "SK", "INT", "FR", "SF", "BLKK", "BLKKRTD", "INTTD", "FRTD", "2PRET", "1PSF",
    "PA0", "PA1", "PA7", "PA14", "PA28", "PA35", "PA46",
    "YA100", "YA199", "YA299", "YA399", "YA449", "YA499", "YA549", "YA550",
})
"""Rules that only ever apply to a team defense.

This gate matters for layer A: an OFFENSIVE player's raw stat line also carries a
``defensiveFumbles`` key (meaning his fumble was recovered by the defense), which
must not be paid as the D/ST's +2 fumble-recovery rule."""

PLAYER_ONLY_RULES: frozenset[str] = frozenset({
    "PY", "PTD", "INTT", "SKD", "2PC", "RY", "RA", "RTD", "2PR",
    "REY", "REC", "RETD", "2PRE", "FUML", "FTD",
    "PAT", "FGM", "FG0", "FG40", "FG50", "FG60",
    "PC", "INC",
})

# Our D/ST engine emits one aggregate column per bucketed table and one for all
# defensive/return touchdowns, because every TD flavour pays the same 6. ESPN
# itemises them. These groups make the two comparable.
DST_RULE_GROUPS: dict[str, tuple[str, ...]] = {
    "PA": ("PA0", "PA1", "PA7", "PA14", "PA28", "PA35", "PA46"),
    "YA": ("YA100", "YA199", "YA299", "YA399", "YA449", "YA499", "YA549", "YA550"),
    "DEF_TD": ("INTTD", "FRTD", "KRTD", "PRTD", "BLKKRTD"),
}


def espn_scores(season: int, weeks: range | None = None, **kw) -> pl.DataFrame:
    """Every scored player-week ESPN recorded, with per-rule point attribution.

    The breakdown is flattened into one fixed ``espn_pts_<ABBR>`` column per rule
    rather than kept as a dict. Storing dicts is a trap: polars infers the struct
    schema from a sample of rows, so a category that appears only in later rows
    is silently dropped — which understated every D/ST touchdown here before the
    totals gave it away.
    """
    weeks = weeks or range(1, 15)

    def fetch() -> pl.DataFrame:
        from ff_agent.data.espn import get_league

        lg = get_league(season)
        rows: list[dict] = []
        for wk in weeks:
            for m in lg.box_scores(wk):
                for p in list(m.home_lineup) + list(m.away_lineup):
                    rec = {
                        "week": wk,
                        "espn_id": str(getattr(p, "playerId", "") or ""),
                        "name": getattr(p, "name", None),
                        "position": getattr(p, "position", None),
                        "pro_team": getattr(p, "proTeam", None),
                        "espn_points": float(getattr(p, "points", 0.0) or 0.0),
                        "on_bye": bool(getattr(p, "on_bye_week", False)),
                    }
                    acc = dict.fromkeys(RULE_ABBRS, 0.0)
                    unmapped = 0.0
                    for k, val in (p.points_breakdown or {}).items():
                        if val is None:
                            continue
                        abbr = ESPN_BREAKDOWN_TO_RULE.get(k)
                        if abbr is None:
                            unmapped += float(val)
                        else:
                            acc[abbr] += float(val)
                    rec.update({f"espn_pts_{a}": v for a, v in acc.items()})
                    rec["espn_unmapped_pts"] = round(unmapped, 2)
                    # raw stat line, stored as JSON rather than a dict: polars
                    # infers struct fields from a sample and would drop keys that
                    # only appear in later rows.
                    rec["espn_raw_breakdown"] = json.dumps(
                        {k: v for k, v in (p.breakdown or {}).items() if v is not None}
                    )
                    rows.append(rec)
        if not rows:
            raise RuntimeError(f"No box scores returned for {season}")
        return pl.DataFrame(rows).unique(subset=["week", "espn_id"], keep="first")

    df = cached("espn_box_scores", fetch, season=season, source="espn", **kw)
    missing = [a for a in RULE_ABBRS if f"espn_pts_{a}" not in df.columns]
    if missing and not kw.get("force"):
        # RULE_ABBRS grew since this file was cached; the parquet has no column
        # for the new rules. Refetch rather than silently scoring them as zero.
        return espn_scores(season, weeks, force=True, **{k: v for k, v in kw.items() if k != "force"})
    return df


def layer_a_rules_check(season: int, weeks: range | None = None) -> pl.DataFrame:
    """Layer A: score ESPN's OWN stat line with OUR rules.

    ``espn_raw_breakdown`` is the raw stat line ESPN scored (receptions, yards,
    and for D/ST a 1.0 flag on whichever bucket applied). Multiplying it by our
    loaded rule values must reproduce ESPN's total exactly. This isolates the
    RULES from the DATA: if layer A is clean and layer B is not, the disagreement
    is a stat value, not a scoring rule.

    Runs from the cache, so it needs no network.
    """
    rules = load_rules(season)
    df = espn_scores(season, weeks)

    def score_row(raw: str | None, pos: str | None) -> float:
        if not raw:
            return 0.0
        is_dst = pos == "D/ST"
        total = 0.0
        for k, val in json.loads(raw).items():
            abbr = ESPN_BREAKDOWN_TO_RULE.get(k)
            if abbr is None:
                continue
            if is_dst and abbr in PLAYER_ONLY_RULES:
                continue
            if not is_dst and abbr in DST_ONLY_RULES:
                continue
            total += float(val) * rules.get(abbr, 0.0)
        return float(round(total, 2))

    return df.select(
        "week", "name", "position", "espn_points",
        pl.struct(["espn_raw_breakdown", "position"]).map_elements(
            lambda r: score_row(r["espn_raw_breakdown"], r["position"]),
            return_dtype=pl.Float64,
        ).alias("rules_points"),
    ).with_columns(
        (pl.col("rules_points") - pl.col("espn_points")).round(2).alias("delta")
    )


def _espn_breakdown_sum() -> pl.Expr:
    return pl.sum_horizontal(
        [pl.col(f"espn_pts_{a}") for a in RULE_ABBRS] + [pl.col("espn_unmapped_pts")]
    ).round(2)


def compare_players(season: int) -> pl.DataFrame:
    """Join our scores to ESPN's for every non-D/ST player-week."""
    espn = espn_scores(season).filter(pl.col("position") != "D/ST")

    resolved = cw.resolve_players(
        espn.select("espn_id", "name", "position",
                    pl.col("pro_team").alias("team"))
        .unique(subset=["espn_id"])
    ).select("espn_id", "canonical_id", "match_method")

    ours = score_players(season).filter(pl.col("season_type") == "REG")
    return (
        espn.join(resolved, on="espn_id", how="left")
        .join(ours.rename({"player_id": "canonical_id"}),
              on=["canonical_id", "week"], how="left", suffix="_ours")
        .with_columns(pl.col("fantasy_points").fill_null(0.0).alias("our_points"))
        .with_columns(
            (pl.col("our_points") - pl.col("espn_points")).round(2).alias("delta"),
            (_espn_breakdown_sum() - pl.col("espn_points")).round(2).alias("layerA_delta"),
        )
    )


def compare_dst(season: int) -> pl.DataFrame:
    espn = espn_scores(season).filter(pl.col("position") == "D/ST")
    ours = score_dst(season)
    dstx = cw.dst_crosswalk().select("espn_id", "team")
    joined = (
        espn.join(dstx, on="espn_id", how="left")
        .join(ours, on=["team", "week"], how="left")
        .with_columns(pl.col("fantasy_points").fill_null(0.0).alias("our_points"))
    )
    # collapse ESPN's itemised buckets/TDs to match our aggregate columns
    for group, members in DST_RULE_GROUPS.items():
        joined = joined.with_columns(
            pl.sum_horizontal([pl.col(f"espn_pts_{m}") for m in members])
            .alias(f"espn_pts_{group}")
        )
    return joined.with_columns(
        (pl.col("our_points") - pl.col("espn_points")).round(2).alias("delta"),
        (_espn_breakdown_sum() - pl.col("espn_points")).round(2).alias("layerA_delta"),
    )


def category_deltas(compared: pl.DataFrame) -> pl.DataFrame:
    """Layer C: which RULE disagrees, over every mismatched row."""
    bad = compared.filter(pl.col("delta").abs() > TOLERANCE)
    if bad.is_empty():
        return pl.DataFrame(schema={"rule": pl.Utf8, "week": pl.Int64, "name": pl.Utf8,
                                    "position": pl.Utf8, "ours": pl.Float64,
                                    "espn": pl.Float64, "diff": pl.Float64})
    frames = []
    for col in [c for c in bad.columns if c.startswith("pts_")]:
        abbr = col[len("pts_"):]
        ecol = f"espn_pts_{abbr}"
        if ecol not in bad.columns:
            continue
        frames.append(
            bad.select(
                pl.lit(abbr).alias("rule"), "week", "name", "position",
                pl.col(col).fill_null(0.0).round(2).alias("ours"),
                pl.col(ecol).fill_null(0.0).round(2).alias("espn"),
            ).with_columns((pl.col("ours") - pl.col("espn")).round(2).alias("diff"))
        )
    if not frames:
        return pl.DataFrame(schema={"rule": pl.Utf8, "week": pl.Int64, "name": pl.Utf8,
                                    "position": pl.Utf8, "ours": pl.Float64,
                                    "espn": pl.Float64, "diff": pl.Float64})
    return (pl.concat(frames)
            .filter(pl.col("diff").abs() > TOLERANCE)
            .sort(["rule", "week"]))


def report(season: int, write: bool = True) -> dict:
    """Run all three layers and summarise."""
    players = compare_players(season)
    dst = compare_dst(season)

    def summarise(df: pl.DataFrame, label: str) -> dict:
        n = df.height
        bad = df.filter(pl.col("delta").abs() > TOLERANCE)
        a_bad = df.filter(pl.col("layerA_delta").abs() > TOLERANCE).height
        return {"label": label, "rows": n, "mismatches": bad.height,
                "exact_pct": round(100 * (n - bad.height) / n, 3) if n else 0.0,
                "layerA_bad": a_bad}

    out = {"season": season,
           "players": summarise(players, "players"),
           "dst": summarise(dst, "dst"),
           "player_categories": category_deltas(players),
           "dst_categories": category_deltas(dst)}

    if write:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        players.filter(pl.col("delta").abs() > TOLERANCE).write_csv(
            REPORTS_DIR / f"scoring_mismatch_players_{season}.csv")
        dst.filter(pl.col("delta").abs() > TOLERANCE).write_csv(
            REPORTS_DIR / f"scoring_mismatch_dst_{season}.csv")
    return out
