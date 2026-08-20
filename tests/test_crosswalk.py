"""CLAUDE.md §0.2 — the hard gate.

The tests that matter most here are the ones proving the gate FIRES. A crosswalk
test suite that only tests the happy path is exactly how these projects die
silently.
"""
import polars as pl
import pytest

from ff_agent.config import LAST_SEASON, SEASON
from ff_agent.data import crosswalk as cw


# ─── canonical table ─────────────────────────────────────────────────────────
def test_canonical_id_is_never_null(canonical):
    assert canonical.filter(pl.col("canonical_id").is_null()).height == 0


def test_espn_id_is_injective(canonical):
    """No espn_id may map to two players — that is the silent-merge failure."""
    d = (canonical.filter(pl.col("espn_id").is_not_null())
         .group_by("espn_id").len().filter(pl.col("len") > 1))
    assert d.height == 0, f"espn_id collisions: {d}"


def test_espn_id_coverage_on_active_skill_players(canonical):
    live = canonical.filter(
        pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"])
        & pl.col("last_season").cast(pl.Int64, strict=False).is_in([LAST_SEASON, SEASON])
    )
    cov = live.filter(pl.col("espn_id").is_not_null()).height / live.height
    assert cov > 0.95, f"espn_id coverage fell to {cov:.1%} — crosswalk degrading"


def test_status_column_present_for_retired_player_guard(canonical):
    """§6's named failure is a board recommending a retired player."""
    assert "status" in canonical.columns
    assert canonical["status"].drop_nulls().n_unique() > 1


# ─── normalization ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("Marvin Harrison Jr.", "marvin harrison"),
    ("A.J. Brown", "a j brown"),
    ("Amon-Ra St. Brown", "amon ra st brown"),
    ("D'Andre Swift", "dandre swift"),
    ("Kenneth Walker III", "kenneth walker"),
])
def test_name_normalization(raw, expected):
    assert cw.normalize_name(raw) == expected


def test_espn_team_abbreviations_map_to_nflverse():
    assert cw.normalize_team("WSH") == "WAS"   # the one live difference
    assert cw.normalize_team("JAC") == "JAX"
    assert cw.normalize_team("OAK") == "LV"    # historical, for old drafts
    assert cw.normalize_team("KC") == "KC"


def test_dst_crosswalk_covers_all_32_teams():
    d = cw.dst_crosswalk()
    assert d.height == 32
    assert d["canonical_id"].n_unique() == 32
    assert "DST_KC" in d["canonical_id"].to_list()


# ─── resolution machinery (works without ESPN credentials) ───────────────────
@pytest.fixture(scope="module")
def sample_pool(_canonical=None):
    """A synthetic ESPN-shaped pool built from nflverse's own espn_ids.

    Lets the resolver be exercised end-to-end before credentials arrive.
    """
    c = cw.canonical_players()
    live = c.filter(
        pl.col("espn_id").is_not_null()
        & pl.col("position").is_in(["QB", "RB", "WR", "TE", "K"])
        & (pl.col("last_season").cast(pl.Int64, strict=False) >= LAST_SEASON)
    ).head(400)
    return live.select(
        "espn_id",
        pl.col("display_name").alias("name"),
        "position",
        pl.col("team"),
    )


def test_known_pool_resolves_completely(sample_pool):
    r = cw.resolve_players(sample_pool)
    assert r.filter(pl.col("match_method") == "unresolved").height == 0
    cw.assert_all_resolved(r, "sample_pool")


def test_resolution_prefers_direct_espn_id(sample_pool):
    r = cw.resolve_players(sample_pool)
    methods = set(r["match_method"].unique().to_list())
    assert methods == {"espn_id"}, f"expected pure espn_id joins, got {methods}"


# ─── the gate must FIRE ──────────────────────────────────────────────────────
def test_gate_fires_on_an_unresolvable_player():
    bogus = pl.DataFrame({
        "espn_id": ["999999999"],
        "name": ["Nonexistent Playerman"],
        "position": ["WR"],
        "team": ["KC"],
    })
    r = cw.resolve_players(bogus)
    assert r["match_method"].to_list() == ["unresolved"]
    with pytest.raises(cw.CrosswalkError) as e:
        cw.assert_all_resolved(r, "bogus")
    assert "NO canonical id" in str(e.value)


def test_gate_fires_when_two_espn_ids_collide_on_one_player(sample_pool):
    """Two ESPN rows silently merging into one canonical player must be caught."""
    one = sample_pool.head(1)
    dup = one.with_columns(pl.lit("888888888").alias("espn_id"))
    merged = pl.concat([one, dup])
    r = cw.resolve_players(merged)
    # the duplicate resolves via the exact name/pos/team triple to the same id
    r = r.with_columns(pl.lit(r["canonical_id"][0]).alias("canonical_id"))
    with pytest.raises(cw.CrosswalkError) as e:
        cw.assert_all_resolved(r, "collision")
    assert "MORE THAN ONE" in str(e.value)


def test_no_fuzzy_matching_a_typo_must_not_resolve():
    """§6: names do not join reliably. A near-miss must fail, not guess."""
    typo = pl.DataFrame({
        "espn_id": ["777777777"],
        "name": ["Patrik Mahoames"],
        "position": ["QB"],
        "team": ["KC"],
    })
    r = cw.resolve_players(typo)
    assert r["match_method"].to_list() == ["unresolved"]


def test_ambiguous_triple_is_rejected_not_guessed():
    """If (name, position, team) identifies 2+ players, it must not be used."""
    c = cw.canonical_players()
    amb = (c.filter(pl.col("normalized_name").is_not_null() & pl.col("team").is_not_null())
           .group_by(["normalized_name", "position", "team"]).len()
           .filter(pl.col("len") > 1))
    if amb.height == 0:
        pytest.skip("no ambiguous triples in current data")
    row = amb.row(0, named=True)
    probe = pl.DataFrame({
        "espn_id": ["666666666"],
        "name": [row["normalized_name"]],
        "position": [row["position"]],
        "team": [row["team"]],
    })
    r = cw.resolve_players(probe)
    assert r["match_method"].to_list() == ["unresolved"]


# ─── overrides ───────────────────────────────────────────────────────────────
def test_override_file_round_trips(tmp_path, monkeypatch):
    p = tmp_path / "ov.csv"
    monkeypatch.setattr(cw, "OVERRIDE_PATH", p)
    assert cw.load_overrides().height == 0
    cw.add_override("12345", "00-0033873", "Test Player", "rookie missing from nflverse")
    ov = cw.load_overrides()
    assert ov.height == 1 and ov["gsis_id"][0] == "00-0033873"
    with pytest.raises(cw.CrosswalkError):
        cw.add_override("12345", "00-0000001", "Dup", "should refuse")


def test_override_beats_direct_espn_id(tmp_path, monkeypatch, sample_pool):
    """A recorded human decision must win over refreshed data."""
    p = tmp_path / "ov.csv"
    monkeypatch.setattr(cw, "OVERRIDE_PATH", p)
    target = sample_pool.head(1)
    espn_id = target["espn_id"][0]
    cw.add_override(espn_id, "00-9999999", target["name"][0], "manual correction")
    r = cw.resolve_players(target)
    assert r["match_method"].to_list() == ["override"]
    assert r["canonical_id"].to_list() == ["00-9999999"]


# ─── the real Milestone 1 gate (needs credentials) ──────────────────────────
@pytest.mark.espn
def test_espn_draftable_pool_resolves_completely():
    from ff_agent.data import espn
    pool = espn.draftable_players(SEASON)
    players = pool.filter(pl.col("position") != "D/ST")
    r = cw.resolve_players(players)
    cw.assert_all_resolved(r, f"draftable_pool_{SEASON}")


@pytest.mark.espn
def test_espn_last_season_rosters_resolve_completely():
    from ff_agent.data import espn
    ros = espn.rosters(LAST_SEASON)
    players = ros.filter(pl.col("position") != "D/ST")
    r = cw.resolve_players(players)
    cw.assert_all_resolved(r, f"rosters_{LAST_SEASON}")


@pytest.mark.espn
def test_espn_dst_entries_all_map_to_a_real_team():
    from ff_agent.data import espn
    pool = espn.draftable_players(SEASON)
    dst = pool.filter(pl.col("position") == "D/ST")
    valid = set(cw.dst_crosswalk()["team"].to_list())
    bad = [t for t in dst["team"].to_list() if cw.normalize_team(t) not in valid]
    assert not bad, f"D/ST rows with unmappable teams: {bad}"


# ─── players with no nflverse counterpart ────────────────────────────────────
def test_espn_only_player_resolves_but_is_flagged(tmp_path, monkeypatch):
    """An ESPN player with no nflverse record must RESOLVE (so the gate can
    close) while being clearly marked as having no history to project from."""
    p = tmp_path / "ov.csv"
    monkeypatch.setattr(cw, "OVERRIDE_PATH", p)
    cw.mark_no_nflverse_match("4361665", "Christopher Dunn",
                              "verified: zero nflverse candidates")
    probe = pl.DataFrame({
        "espn_id": ["4361665"], "name": ["Christopher Dunn"],
        "position": ["K"], "team": [None],
    })
    r = cw.resolve_players(probe)
    assert r["match_method"].to_list() == ["espn_only"]
    assert r["canonical_id"].to_list() == ["ESPN_4361665"]
    assert r["has_nflverse_data"].to_list() == [False]
    cw.assert_all_resolved(r, "espn_only_probe")


def test_real_players_are_flagged_as_having_nflverse_data(sample_pool):
    r = cw.resolve_players(sample_pool)
    assert r["has_nflverse_data"].all()


def test_no_nflverse_match_sentinel_is_not_a_valid_gsis_id():
    """Guard against the sentinel ever being joined against real data."""
    assert not cw.NO_NFLVERSE_MATCH.startswith("00-")
    c = cw.canonical_players()
    assert c.filter(pl.col("canonical_id") == cw.NO_NFLVERSE_MATCH).height == 0


# ─── D/ST: team entities, not players (§0.2 corollary) ──────────────────────
def test_dst_negative_espn_id_convention():
    """ESPN encodes a defense as -16000 - proTeamId. Real draft history uses it."""
    assert cw.espn_dst_id(25) == "-16025"   # 49ers
    assert cw.espn_dst_id(33) == "-16033"   # Ravens
    d = cw.dst_crosswalk()
    assert d.filter(pl.col("espn_id") == "-16025")["team"][0] == "SF"
    assert d.filter(pl.col("espn_id") == "-16033")["team"][0] == "BAL"


def test_dst_crosswalk_is_injective_both_ways():
    d = cw.dst_crosswalk()
    assert d.height == 32
    assert d["espn_id"].n_unique() == 32
    assert d["canonical_id"].n_unique() == 32
    assert d["team"].n_unique() == 32


def test_dst_resolves_by_negative_id():
    rows = pl.DataFrame({"espn_id": ["-16025", "-16002"],
                         "name": ["49ers D/ST", "Bills D/ST"]})
    r = cw.resolve_dst(rows)
    assert r["canonical_id"].to_list() == ["DST_SF", "DST_BUF"]
    assert set(r["match_method"].to_list()) == {"dst_espn_id"}


def test_dst_resolves_by_team_abbreviation_including_wsh():
    rows = pl.DataFrame({"espn_id": ["x"], "name": ["Commanders D/ST"], "team": ["WSH"]})
    r = cw.resolve_dst(rows)
    assert r["canonical_id"].to_list() == ["DST_WAS"]


def test_unknown_dst_does_not_silently_resolve():
    rows = pl.DataFrame({"espn_id": ["-19999"], "name": ["Fake D/ST"], "team": ["ZZZ"]})
    r = cw.resolve_dst(rows)
    assert r["match_method"].to_list() == ["unresolved"]


def test_is_dst_recognises_both_spellings():
    assert cw.is_dst("D/ST")
    assert cw.is_dst("DST")
    assert cw.is_dst(None, "-16025")
    assert not cw.is_dst("WR", "4429795")


# ─── resolving to the WRONG person (the gate §11 specifies cannot catch this) ─
def test_stale_resolution_is_flagged_as_implausible(tmp_path, monkeypatch):
    """A 2026 draftable player joined onto someone who last played in 1984 is a
    wrong join, not a comeback. This is the live bug we hit: nflverse assigns
    espn_id 4686658 to a DB who retired in 1984, while that id really belongs to
    a 2026 rookie RB who was ~10% rostered."""
    monkeypatch.setattr(cw, "OVERRIDE_PATH", tmp_path / "ov.csv")
    probe = pl.DataFrame({
        "espn_id": ["4686658"], "name": ["Mike Washington Jr."],
        "position": ["RB"], "team": ["LV"],
    })
    r = cw.resolve_players(probe)
    # it DOES resolve — that is exactly why the §11 gate alone is insufficient
    assert r["match_method"].to_list() == ["espn_id"]
    cw.assert_all_resolved(r, "stale_probe")
    # but it must fail the plausibility gate
    bad = cw.implausible_resolutions(r, 2026)
    assert bad.height == 1
    assert bad["last_season"][0] == 1984
    with pytest.raises(cw.CrosswalkError) as e:
        cw.assert_resolutions_plausible(r, "stale_probe", 2026)
    assert "WRONG PERSON" in str(e.value)


def test_override_repairs_the_bad_nflverse_espn_id():
    """With the shipped override in place, the real rookie is resolved."""
    probe = pl.DataFrame({
        "espn_id": ["4686658"], "name": ["Mike Washington Jr."],
        "position": ["RB"], "team": ["LV"],
    })
    r = cw.resolve_players(probe)
    assert r["match_method"].to_list() == ["override"]
    assert r["canonical_id"].to_list() == ["00-0040878"]
    assert cw.implausible_resolutions(r, 2026).height == 0


def test_two_way_players_are_not_falsely_flagged():
    """Travis Hunter is a CB in nflverse and a WR in ESPN — a CORRECT match.
    Position mismatch alone must never trigger the implausibility gate."""
    probe = pl.DataFrame({
        "espn_id": ["4685415"], "name": ["Travis Hunter"],
        "position": ["WR"], "team": ["JAX"],
    })
    r = cw.resolve_players(probe)
    assert r.filter(pl.col("match_method") == "unresolved").height == 0
    assert cw.implausible_resolutions(r, 2026).height == 0
