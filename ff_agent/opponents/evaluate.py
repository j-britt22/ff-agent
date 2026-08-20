"""MILESTONE 5 TEST (§11): beat ADP at predicting this league's own picks.

The baseline is superflex ADP, not national/standard ADP. §7.5 already rules
standard out for a 2-QB league, so beating it would prove nothing — the honest
bar is the best baseline actually available.

Two evaluations, because neither alone is fair:

* **Strict**: fit on 2023-24, predict 2025. No lookahead at all — but both
  training seasons were ONE-QB leagues, so it asks whether manager identity
  transfers across a format change.
* **Per-manager breakdown** of that same strict fit, because §2.3 weights four
  managers at 2x and they are not the ones with the most evidence. A headline
  gain driven entirely by managers you face once would be improving the wrong
  two-thirds of the schedule.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ff_agent.opponents import history as H
from ff_agent.opponents import pickmodel as PM
from ff_agent.opponents import tendencies as TD


def _score_draft(
    season_picks: pl.DataFrame,
    tilt_lookup: dict,
    reach: dict[str, float],
    use_manager_terms: bool = True,
) -> dict:
    """Walk a draft pick by pick, scoring the model against what happened."""
    picks = season_picks.sort("overall_pick")
    pool = picks.select(
        "canonical_id", "position", "consensus_fraction", "name"
    ).unique(subset=["canonical_id"], keep="first")

    taken: set[str] = set()
    recent: list[str] = []
    logps, ranks, hits = [], [], 0

    empty_tilt: dict = {}
    for row in picks.to_dicts():
        avail = pool.filter(~pl.col("canonical_id").is_in(list(taken)) if taken else pl.lit(True))
        if avail.is_empty():
            break
        probs = PM.pick_probabilities(
            avail,
            pick_fraction=float(row["pick_fraction"]),
            manager=row["manager"],
            tilt_lookup=tilt_lookup if use_manager_terms else empty_tilt,
            reach=reach if use_manager_terms else {},
            recent_positions=recent,
        )
        ids = avail["canonical_id"].to_list()
        actual = row["canonical_id"]
        if actual in ids:
            i = ids.index(actual)
            logps.append(float(np.log(max(probs[i], 1e-12))))
            order = np.argsort(-probs)
            ranks.append(int(np.where(order == i)[0][0]) + 1)
            hits += int(order[0] == i)
        taken.add(actual)
        if row["position"]:
            recent.append(row["position"])

    n = len(logps)
    return {
        "n_picks": n,
        "mean_log_prob": round(float(np.mean(logps)), 4) if n else None,
        "median_rank_of_actual": int(np.median(ranks)) if n else None,
        "top1_accuracy": round(hits / n, 4) if n else None,
        "top5_rate": round(float(np.mean([r <= 5 for r in ranks])), 4) if n else None,
    }


def strict_holdout(test_season: int = 2025) -> pl.DataFrame:
    """Fit on every earlier season, predict ``test_season``."""
    picks = TD.picks_with_consensus().filter(pl.col("canonical_id").is_not_null())
    train = picks.filter(pl.col("season") < test_season)
    test = picks.filter(pl.col("season") == test_season)
    tilt, reach = PM.build_lookups(train)
    rows = [
        {"model": "superflex ADP only", **_score_draft(test, {}, {}, use_manager_terms=False)},
        {"model": "ADP + manager terms", **_score_draft(test, tilt, reach)},
    ]
    return pl.DataFrame(rows)


def per_manager_breakdown(test_season: int = 2025) -> pl.DataFrame:
    """Where do the manager terms actually help, and where do they hurt?

    A single headline number hides the thing that matters: §2.3 weights the four
    double-up managers most heavily, and they are not the ones with the most
    evidence. If the gain comes entirely from managers you face once, the model
    is improving the wrong two-thirds of the schedule.
    """
    from ff_agent.config import DOUBLE_UP_MANAGERS

    dbl = {H.canonical_manager(m) for m in DOUBLE_UP_MANAGERS}
    picks = TD.picks_with_consensus().filter(pl.col("canonical_id").is_not_null())
    train = picks.filter(pl.col("season") < test_season)
    test = picks.filter(pl.col("season") == test_season)
    tilt, reach = PM.build_lookups(train)

    rows = []
    for mgr in sorted(test["manager"].unique().to_list()):
        sub = test.filter(pl.col("manager") == mgr)
        if sub.is_empty():
            continue
        base = _score_draft_subset(test, sub, {}, {}, use_manager_terms=False)
        with_m = _score_draft_subset(test, sub, tilt, reach)
        if base["mean_log_prob"] is None or with_m["mean_log_prob"] is None:
            continue
        rows.append({
            "manager": mgr,
            "faced_twice": mgr in dbl,
            "n_picks": with_m["n_picks"],
            "adp_only": base["mean_log_prob"],
            "with_manager": with_m["mean_log_prob"],
            "gain": round(with_m["mean_log_prob"] - base["mean_log_prob"], 4),
        })
    return pl.DataFrame(rows).sort("gain", descending=True)


def _score_draft_subset(
    all_picks: pl.DataFrame,
    subset: pl.DataFrame,
    tilt_lookup: dict,
    reach: dict[str, float],
    use_manager_terms: bool = True,
) -> dict:
    """Walk the full draft for correct availability, but score only ``subset``."""
    want = set(subset["canonical_id"].to_list())
    picks = all_picks.sort("overall_pick")
    pool = picks.select("canonical_id", "position", "consensus_fraction", "name") \
                .unique(subset=["canonical_id"], keep="first")
    taken: set[str] = set()
    recent: list[str] = []
    logps, ranks, hits, n = [], [], 0, 0
    empty: dict = {}
    for row in picks.to_dicts():
        avail = pool.filter(~pl.col("canonical_id").is_in(list(taken)) if taken else pl.lit(True))
        if avail.is_empty():
            break
        actual = row["canonical_id"]
        if actual in want:
            probs = PM.pick_probabilities(
                avail, float(row["pick_fraction"]), row["manager"],
                tilt_lookup if use_manager_terms else empty,
                reach if use_manager_terms else {},
                recent_positions=recent,
            )
            ids = avail["canonical_id"].to_list()
            if actual in ids:
                i = ids.index(actual)
                logps.append(float(np.log(max(probs[i], 1e-12))))
                order = np.argsort(-probs)
                ranks.append(int(np.where(order == i)[0][0]) + 1)
                hits += int(order[0] == i)
                n += 1
        taken.add(actual)
        if row["position"]:
            recent.append(row["position"])
    return {
        "n_picks": n,
        "mean_log_prob": round(float(np.mean(logps)), 4) if n else None,
        "median_rank_of_actual": int(np.median(ranks)) if n else None,
        "top1_accuracy": round(hits / n, 4) if n else None,
    }
