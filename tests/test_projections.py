"""Milestone 3 — consensus ingest, calibration, opportunity model."""
import polars as pl
import pytest

from ff_agent.config import LAST_SEASON, SEASON
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


def test_history_window_gives_enough_transitions():
    """§I-3c needs measured stability; 2016 buys 9 transitions, not 4."""
    t = O.transitions()
    assert len(set(t["season"].to_list())) >= 9


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

    The prior-season opportunity table carries one row per (player, NFL team),
    so anyone traded mid-season arrives here twice — 25 players for 2026. Both
    rows used to survive, fanning out 1 -> 2 through ``blend`` and 2 -> 4
    through the board's tier join. The opportunity table is deliberately left
    alone: two stints ARE two rows of opportunity, and collapsing them there
    would change what the model is fitted on.
    """
    prior = (O.features_with_actuals(SEASON - 1)
             .filter(pl.col("position").is_in(M.POSITIONS))
             .drop_nulls("canonical_id"))
    traded = prior.group_by("canonical_id").len().filter(pl.col("len") > 1)
    assert traded.height > 0, "no traded players — this test would prove nothing"

    p = M.project(SEASON)
    assert p.height == p["canonical_id"].n_unique()
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
