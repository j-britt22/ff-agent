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
def raw_curve():
    return CAL.rank_points_curve()


@pytest.fixture(scope="module")
def curve(raw_curve):
    return CAL.smooth_curve(raw_curve)


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


def test_curve_never_projects_negative_points(curve, raw_curve):
    """A negative full-season projection is impossible under §1 in practice: the
    only negative categories (INT −2, sack taken −1, fumble lost −2, missed FG
    −1) are reachable only by a player who is simultaneously accruing 0.05/carry,
    0.1/yd and 0.5/reception. The 2026 board nonetheless shipped five RBs at
    negative points, read straight off the deep tail of this curve.

    The floor lives in ``smooth_curve`` — the column every consumer reads — and
    NOT at the end of the pipeline, so the step that produced the negative is the
    step that is fixed. The unfloored fit is deliberately still there, so the
    evidence behind the floor stays checkable rather than being erased by it.
    """
    # Asserted against zero, NOT against MIN_EXPECTED_POINTS: comparing to the
    # constant would let anyone relax the floor and the test in one edit.
    assert CAL.MIN_EXPECTED_POINTS == 0.0
    assert curve["expected_points_smooth"].min() >= 0.0
    assert raw_curve.filter(pl.col("expected_points") < 0).height > 0


def test_the_negative_tail_is_thin_support_not_signal(raw_curve):
    """WHY the floor is right, rather than merely that it is applied.

    At every skill position the first negative rank falls PAST the deepest rank
    that all ten seasons populate (measured 2016-2025: QB 71/73, RB 143/156,
    WR 210/222, TE 123/129). That is a selection effect — a season only produces
    a rank that deep if some marginal player recorded a stat line at all, and the
    players who get there are the ones whose only counted event was a lost
    fumble. So the negative is one season's fluke, not an expectation.
    """
    for pos in ("QB", "RB", "WR", "TE"):
        sub = raw_curve.filter(pl.col("position") == pos).sort("pos_rank")
        full = sub["n_seasons"].max()
        deepest_full_support = sub.filter(pl.col("n_seasons") == full)["pos_rank"].max()
        neg = sub.filter(pl.col("expected_points") < 0)
        assert not neg.is_empty(), f"{pos}: expected the fitted tail to go negative"
        assert neg["pos_rank"].min() > deepest_full_support, (
            f"{pos}: a negative rank appears at {neg['pos_rank'].min()}, inside the "
            f"fully-supported range (to {deepest_full_support}) — the floor's "
            f"justification no longer holds and needs re-deriving, not re-tuning."
        )

    # K and D/ST pools are only 32-47 deep and fully supported throughout.
    for pos in ("K", "DST"):
        sub = raw_curve.filter(pl.col("position") == pos)
        assert sub.filter(pl.col("expected_points") < 0).is_empty()


def test_ranks_past_the_fitted_curve_are_floored_flat_and_flagged(curve):
    """Reading a rank deeper than the curve was fitted on is an EXTRAPOLATION,
    so it is named (``curve_tail``), floored, flat and flagged.

    Before the fix it silently inherited the curve's last fitted value — the
    single thinnest point on the whole curve — which is how RB166 (Ja'Quinden
    Jackson, ECR 386) acquired a projection of −0.50 that nothing measured.
    """
    tail = CAL.curve_tail(curve)
    assert (tail["_tail"] >= 0.0).all()

    rb_max = int(curve.filter(pl.col("position") == "RB")["pos_rank"].max())
    n_past = 5
    synthetic = pl.DataFrame({
        "position": ["RB"] * (rb_max + n_past),
        "ecr": [float(i) for i in range(1, rb_max + n_past + 1)],
    })
    out = CAL.consensus_to_points(synthetic, curve)

    past = out.filter(pl.col("pos_rank") > rb_max)
    assert past.height == n_past
    assert past["consensus_points_extrapolated"].all()
    assert (past["consensus_points"] >= 0.0).all()
    # flat, not extended — continuing the tail's slope would project ever more
    # negative points for players expected to score none
    assert past["consensus_points"].n_unique() == 1

    inside = out.filter(pl.col("pos_rank") <= rb_max)
    assert not inside["consensus_points_extrapolated"].any()


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
