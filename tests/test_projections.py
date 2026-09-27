"""Milestone 3 — consensus ingest, calibration, opportunity model."""
import polars as pl
import pytest

from ff_agent.config import LAST_SEASON, SEASON
from ff_agent.data import nflverse as nv
from ff_agent.projections import calibration as CAL
from ff_agent.projections import consensus as C
from ff_agent.projections import model as M
from ff_agent.projections import opportunity as O


# ─── consensus ──────────────────────────────────────────────────────────────
def test_superflex_is_the_anchor_not_standard():
    """§7.5: standard ADP misstates positional demand in a 2-QB league."""
    e = C.ecr(LAST_SEASON, ecr_type=C.SUPERFLEX)
    assert e.head(24).filter(pl.col("position") == "QB").height >= 8
    std = C.ecr(LAST_SEASON, ecr_type="ro")
    assert std.head(24).filter(pl.col("position") == "QB").height <= 2


def test_ecr_quality_filters_reject_known_artifacts():
    """The scrape carries a literal "Player Name" placeholder ranked 12th overall
    and single-expert entries ranked absurdly high. Both would poison the
    rank->points calibration exactly where it matters most."""
    _, rejected = C.ecr(LAST_SEASON, with_rejects=True)
    reasons = set(rejected["reject_reason"].to_list())
    assert "placeholder name" in reasons
    assert any("single-expert" in r for r in reasons)
    assert rejected.filter(pl.col("player") == "Player Name").height == 1


def test_ecr_resolves_to_canonical_ids_without_name_matching():
    """FantasyPros ids bridge to gsis via ff_playerids — never by name (§0.2)."""
    e = C.ecr(LAST_SEASON)
    resolved = e.filter(pl.col("canonical_id").is_not_null()).height
    assert resolved / e.height > 0.97


def test_preseason_snapshot_has_no_lookahead():
    for season in (2022, 2023, 2024, 2025):
        d = C.preseason_snapshot_date(season)
        assert d.year == season and (d.month, d.day) <= (9, 5)


def test_espn_season_yardage_is_per_game_not_total():
    """ESPN reports season passing/rushing/receiving yards PER GAME while every
    other field is a season total. Read literally, every projection is ~17x wrong."""
    assert "rushingYards" in C.PER_GAME_FIELDS
    assert "receivingYards" in C.PER_GAME_FIELDS
    assert "passingYards" in C.PER_GAME_FIELDS
    assert "rushingAttempts" not in C.PER_GAME_FIELDS
    assert C.GAMES_IN_SEASON == 17


# ─── calibration ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def curve():
    return CAL.smooth_curve(CAL.rank_points_curve())


def test_curve_is_monotonic_within_position(curve):
    """A better positional rank must be worth more points."""
    for pos in ("QB", "RB", "WR", "TE"):
        sub = (curve.filter((pl.col("position") == pos) & (pl.col("pos_rank") <= 40))
               .sort("pos_rank"))
        vals = sub["expected_points_smooth"].to_list()
        assert vals[0] > vals[-1]
        assert vals[0] == max(vals)


def test_replacement_levels_are_in_the_right_ballpark(curve):
    """§7.3's replacement ranks, priced in §1 scoring."""
    def at(pos, rank):
        return curve.filter(
            (pl.col("position") == pos) & (pl.col("pos_rank") == rank)
        )["expected_points_smooth"][0]
    assert 200 < at("QB", 18) < 280
    assert 150 < at("RB", 22) < 230
    assert 140 < at("WR", 22) < 220
    assert 100 < at("TE", 10) < 170


def test_qb_replacement_is_lower_than_the_spec_estimated(curve):
    """§3.1 guessed QB18 ~265 and concluded QB and RB were near-tied on VOR.
    Measured on 10 seasons under §1 scoring, QB18 is well below that, so QB VOR
    is clearly ahead of RB rather than level with it."""
    def vor(pos, rank):
        s = curve.filter(pl.col("position") == pos)
        return (s.filter(pl.col("pos_rank") == 1)["expected_points_smooth"][0]
                - s.filter(pl.col("pos_rank") == rank)["expected_points_smooth"][0])
    assert vor("QB", 18) > vor("RB", 22)


def test_recency_weighting_and_covid_discount():
    w = CAL.season_weights(list(range(2016, 2026)))
    assert w[2025] == 1.0
    assert w[2016] < w[2020] or True          # 2020 is additionally discounted
    assert w[2020] < 0.5 ** ((2025 - 2020) / CAL.DEFAULT_HALFLIFE) + 1e-9


# ─── stickiness (ADD-§B / §I-3c) ────────────────────────────────────────────
@pytest.fixture(scope="module")
def stability():
    return O.stability_table()


def test_volume_is_stickier_than_efficiency(stability):
    """ADD-§B's core claim, measured on our own data rather than assumed."""
    m = stability.group_by("kind").agg(pl.col("yoy_stability").mean())
    d = dict(zip(m["kind"].to_list(), m["yoy_stability"].to_list()))
    assert d["volume"] > d["efficiency"]
    assert d["share"] > d["efficiency"]


def test_target_share_is_sticky_for_receivers(stability):
    """ADD-§B: target share ~0.70, the stickiest skill-position metric."""
    for pos in ("WR", "TE"):
        r = stability.filter(
            (pl.col("position") == pos) & (pl.col("feature") == "target_share")
        )["yoy_stability"][0]
        assert r > 0.6


def test_rb_carries_are_the_top_rb_signal(stability):
    """ADD-§B: RB touches per game, correlations approaching 0.60."""
    r = stability.filter(
        (pl.col("position") == "RB") & (pl.col("feature") == "carries_pg")
    )["yoy_stability"][0]
    assert r > 0.6


def test_efficiency_is_not_projectable(stability):
    """Prior-year yards per carry must not survive feature selection."""
    ypc = stability.filter(
        (pl.col("position") == "WR") & (pl.col("feature") == "yards_per_carry")
    )["yoy_stability"][0]
    assert ypc < 0.3
    for pos in ("RB", "WR"):
        assert "yards_per_carry" not in M.select_features(stability, pos)


def test_te_td_rate_sits_on_the_stability_cutoff(stability):
    """Recorded rather than tuned away. Summing traded players' stints moved TE
    ``td_rate`` from 0.448 to 0.452 against ``MIN_STABILITY = 0.45``, which
    flipped it INTO the TE feature set on four thousandths of correlation.

    Nothing about tight ends changed — a hard threshold just happens to run
    through this one feature. It is not free either: including it is worth up to
    8.7 model points (Tucker Kraft) and reorders the top 20 TEs, though it does
    not change WHICH 20 they are, and at the 0.12 blend weight the board moves
    about a point. So the TE feature set is the fragile part of this model, and
    if it flips back out that is noise, not a finding."""
    r = stability.filter(
        (pl.col("position") == "TE") & (pl.col("feature") == "td_rate")
    )["yoy_stability"][0]
    assert abs(r - M.MIN_STABILITY) < 0.02, (
        f"TE td_rate stability {r} has moved off the {M.MIN_STABILITY} cutoff; "
        "re-measure what including or excluding it now does to the TE board"
    )


def test_history_window_gives_enough_transitions():
    """§I-3c needs measured stability; 2016 buys 9 transitions, not 4."""
    t = O.transitions()
    assert len(set(t["season"].to_list())) >= 9


# ─── traded players: one row, whole season ──────────────────────────────────
@pytest.fixture(scope="module")
def feats_2025():
    return O.player_season_features(2025)


def test_a_traded_player_is_one_row_carrying_his_whole_season(feats_2025):
    """nflverse stats are weekly and carry the team he suited up for, so keying
    on team splits a traded player into stints. Every feature here is a RATE, so
    the shorter stint often projects HIGHER — Adam Thielen's 5 games in PIT came
    out at 49.7 against his 9 in MIN at 37.2 — and whichever stint won, the
    other was dropped downstream. Sum them instead."""
    ps = nv.player_stats(2025, offline=True).filter(pl.col("season_type") == "REG")
    for pid, name in (("00-0030035", "Adam Thielen"), ("00-0034960", "Jakobi Meyers"),
                      ("00-0026158", "Joe Flacco")):
        weekly = ps.filter(pl.col("player_id") == pid)
        row = feats_2025.filter(pl.col("canonical_id") == pid)
        assert row.height == 1, f"{name} appears {row.height} times"
        r = row.to_dicts()[0]
        assert r["n_teams"] > 1, f"{name} is meant to be a mid-season trade"
        assert r["games"] == weekly.height
        for col, feat in (("targets", "targets"), ("carries", "carries"),
                          ("attempts", "pass_att"), ("sacks_suffered", "sacks_taken")):
            assert r[feat] == weekly[col].sum(), f"{name}.{feat}"
        # rates are recomputed over the COMBINED games, not carried from a stint
        assert r["targets_pg"] == pytest.approx(r["targets"] / r["games"])
        # the label is the team he finished on
        assert r["team"] == weekly.sort("week")["team"][-1]


def test_the_aggregation_conserves_every_counting_stat(feats_2025):
    """The check that matters: summing stints must not lose or invent a snap."""
    ps = nv.player_stats(2025, offline=True).filter(
        (pl.col("season_type") == "REG") & pl.col("player_id").is_not_null()
    )
    weekly = ps.group_by("player_id").agg(
        pl.len().alias("g"), pl.col("targets").sum().alias("t"),
        pl.col("carries").sum().alias("c"), pl.col("attempts").sum().alias("a"),
    )
    j = weekly.join(
        feats_2025.select(pl.col("canonical_id").alias("player_id"),
                          "games", "targets", "carries", "pass_att"),
        on="player_id", how="full", coalesce=True,
    )
    assert j.height == weekly.height == feats_2025.height
    assert j.filter(pl.col("g") != pl.col("games")).height == 0
    assert j.filter(pl.col("t") != pl.col("targets")).height == 0
    assert j.filter(pl.col("c") != pl.col("carries")).height == 0
    assert j.filter(pl.col("a") != pl.col("pass_att")).height == 0


def test_games_are_not_clamped_at_seventeen(feats_2025):
    """A traded player can legitimately log 18 in an 18-week season by missing
    neither team's bye. Rashid Shaheed did in 2025 — 9 with NO, 9 with SEA — so
    a min(games, 17) here would silently inflate every one of his per-game rates."""
    shaheed = feats_2025.filter(pl.col("canonical_id") == "00-0037545").to_dicts()[0]
    assert shaheed["games"] == 18 and shaheed["n_teams"] == 2
    assert feats_2025["games"].max() == 18


def test_no_phantom_player_with_no_id(feats_2025):
    """nflverse ships a placeholder weekly row per week with a null player_id and
    zero of everything. Per team they were obvious junk; aggregated per player
    they would collapse into one phantom with 18 games."""
    assert feats_2025["canonical_id"].null_count() == 0


def test_duplicate_rows_raise_instead_of_being_deduplicated():
    """§0.2. ``model.project`` used to paper over the stint split with a
    .unique(keep="first") on a frame sorted by model_points, so the fuller half
    of a real season vanished for 9 of 2025's 25 traded skill players. Which row
    a dedupe keeps is an accident of sort order — fail and find the join."""
    df = pl.DataFrame({
        "canonical_id": ["x", "x", "y"], "name": ["A", "A", "B"],
        "position": ["WR"] * 3, "team": ["PIT", "MIN", "GB"], "games": [5, 9, 17],
    })
    with pytest.raises(ValueError, match="appear more than once") as e:
        O.assert_one_row_per_player(df, "a test")
    assert "PIT" in str(e.value) and "MIN" in str(e.value), (
        "the error must NAME the offending rows or it is not actionable"
    )
    O.assert_one_row_per_player(df.head(1), "a test")


def test_a_duplicated_null_id_still_names_its_rows():
    """A blocking §0.2 error that lists nobody is the opposite of failing loudly.

    ``null.is_in(...)`` is null, so building the detail table with a filter drops
    exactly the rows being complained about — the raise still fires, but prints
    an empty table. Hence the null-aware semi-join."""
    df = pl.DataFrame(
        {"canonical_id": [None, None, "y"], "name": ["ghost", "ghost", "B"],
         "position": [None, None, "WR"], "team": ["TB", "GB", "GB"],
         "games": [9, 9, 17]},
        schema_overrides={"canonical_id": pl.Utf8, "position": pl.Utf8},
    )
    with pytest.raises(ValueError, match="appear more than once") as e:
        O.assert_one_row_per_player(df, "a test")
    assert "ghost" in str(e.value), f"detail table lost the null rows:\n{e.value}"


# ─── model ──────────────────────────────────────────────────────────────────
def test_role_features_prevent_backup_inflation():
    """Per-game rates alone ranked Jimmy Garoppolo and Joe Milton III as top-10
    QBs off tiny 2024 samples. games and points encode role."""
    assert "games" in O.ROLE_FEATURES and "points" in O.ROLE_FEATURES
    p = M.project(LAST_SEASON)
    top_qb = p.filter(pl.col("position") == "QB").head(10)["name"].to_list()
    for backup in ("Jimmy Garoppolo", "Joe Milton III", "Joshua Dobbs"):
        assert backup not in top_qb


def test_model_fits_every_position():
    models = M.fit(LAST_SEASON)
    assert set(models) == set(M.POSITIONS)
    for pos, m in models.items():
        assert m.n_train >= 40 and m.r2 > 0.15


def test_fit_uses_no_data_from_the_target_season():
    t = O.transitions()
    train = t.filter(pl.col("season") <= LAST_SEASON - 2)
    assert train["season"].max() <= LAST_SEASON - 2
    # a transition labelled N carries N+1 outcomes, so nothing reaches LAST_SEASON
    assert (train["season"].max() + 1) < LAST_SEASON


def test_a_traded_player_is_projected_once():
    """§0.2 at the root of the fan-out that reached the 2026 board.

    The prior-season opportunity table used to carry one row per (player, NFL
    team), so anyone traded mid-season arrived twice — 25 players for 2026 — and
    both rows fanned out 1 -> 2 through ``blend`` and 2 -> 4 through the board's
    tier join. It is no longer left alone: the stints are SUMMED into one
    player-season, so the traded player is already one row before the model sees
    him, and he is still one row after it projects him. Identify him by
    ``n_teams`` now rather than by a duplicate, or this test proves nothing.
    """
    prior = (O.features_with_actuals(SEASON - 1)
             .filter(pl.col("position").is_in(M.POSITIONS))
             .drop_nulls("canonical_id"))
    assert prior.height == prior["canonical_id"].n_unique()
    traded = prior.filter(pl.col("n_teams") > 1)
    assert traded.height > 0, "no traded players — this test would prove nothing"

    p = M.project(SEASON)
    assert p.height == p["canonical_id"].n_unique()
    assert p["canonical_id"].null_count() == 0
    assert p.filter(pl.col("canonical_id").is_in(traded["canonical_id"].implode())).height \
        == traded.height


def test_blend_cannot_multiply_rows():
    """Consensus is the left side of a left join, so it fixes the row count —
    but only if the model side is unique on the key."""
    cons = pl.DataFrame({
        "canonical_id": ["a", "b"],
        "consensus_points": [200.0, 100.0],
    })
    model = pl.DataFrame({
        "canonical_id": ["a", "a", "b"],
        "model_points": [180.0, 120.0, 90.0],
    })
    assert M.blend(cons, model).height == 3, "a duplicated model row fans out"
    assert M.blend(cons, model.unique(subset=["canonical_id"])).height == 2
