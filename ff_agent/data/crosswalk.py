"""Canonical player table and the ESPN -> nflverse ID crosswalk.

CLAUDE.md §0.2 is the governing rule:

    Every rostered and drafted player must resolve to exactly one canonical ID.
    Assert it. Fail loudly on any miss.

**There is no fuzzy matching anywhere in this module, at any tier.** Name
normalization exists to make the *diagnostic report* readable, never to make a
join decision. §6: "Names do not join reliably — suffixes, initials, duplicates."
The failure this prevents is silent: a board that recommends a retired player.

Resolution tiers, tried in order, each recorded on the output row:

  1. ``override``       a human wrote it in overrides/player_id_overrides.csv.
                        Wins over everything — a recorded human decision is never
                        overridden by refreshed data.
  2. ``espn_id``        direct join on nflverse ``players.espn_id``. ~99% of live
                        skill players.
  3. ``name_pos_team``  exact (normalized name, position, team) triple, accepted
                        ONLY when it identifies exactly one player. Ambiguity is
                        a failure, not a coin flip.
  4. ``unresolved``     blocking error, written to the unmatched report.

D/ST are not players and never enter the player crosswalk. They resolve
team-abbreviation -> team-abbreviation and get canonical IDs of the form
``DST_KC``.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from ff_agent.config import OVERRIDES_DIR, REPORTS_DIR
from ff_agent.data import nflverse as nv

OVERRIDE_PATH = OVERRIDES_DIR / "player_id_overrides.csv"
OVERRIDE_COLUMNS = ["espn_id", "gsis_id", "player_name", "reason", "added_at"]

NO_NFLVERSE_MATCH = "NO_NFLVERSE_MATCH"
"""Sentinel for the ``gsis_id`` column of an override.

Some ESPN players genuinely have no nflverse counterpart — undrafted free agents
and camp bodies who have never taken an NFL regular-season snap and so were never
issued a gsis_id. They must still RESOLVE (otherwise the §0.2 gate can never
close), but they must never be mistaken for a player we have history on.

Such a player gets a minted ``ESPN_<espn_id>`` canonical id and
``has_nflverse_data = False``. Downstream that flag is a hard stop: a player with
no history must never receive a modelled projection. This is the opposite of
papering over a miss — it records, in a tracked file, a human-verified finding
that no counterpart exists."""

# ─── Teams ───────────────────────────────────────────────────────────────────
CURRENT_TEAMS: tuple[str, ...] = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
)
assert len(CURRENT_TEAMS) == 32

ESPN_TEAM_ALIASES: dict[str, str] = {
    # ESPN spelling            -> nflverse spelling
    "WSH": "WAS", "WFT": "WAS",
    "JAC": "JAX",
    "LA": "LAR",
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    # historical relocations, so old league history still joins
    "OAK": "LV", "SD": "LAC", "STL": "LAR",
}


def normalize_team(abbr: str | None) -> str | None:
    if not abbr:
        return None
    a = abbr.strip().upper()
    return ESPN_TEAM_ALIASES.get(a, a)


DST_ID_OFFSET = -16000
"""ESPN encodes a team defense as a NEGATIVE player id: ``-16000 - proTeamId``.
So the 49ers (proTeamId 25) are ``-16025`` and the Ravens (33) are ``-16033``.
These never appear in any player crosswalk — D/ST are team entities, not people
(CLAUDE.md §0.2 corollary), and every one of them shows up in real draft history.
"""


def espn_dst_id(pro_team_id: int) -> str:
    return str(DST_ID_OFFSET - int(pro_team_id))


def dst_crosswalk() -> pl.DataFrame:
    """The 32 team defenses: ESPN id and abbreviation -> nflverse team abbr.

    Built from espn_api's own PRO_TEAM_MAP so it cannot drift from the source,
    then asserted down to exactly the 32 current teams.
    """
    from espn_api.football.constant import PRO_TEAM_MAP

    rows = []
    for pro_id, espn_abbr in PRO_TEAM_MAP.items():
        if not espn_abbr or espn_abbr == "None" or int(pro_id) == 0:
            continue
        team = normalize_team(espn_abbr)
        if team not in CURRENT_TEAMS:
            continue
        rows.append({
            "espn_id": espn_dst_id(pro_id),
            "pro_team_id": int(pro_id),
            "espn_abbr": espn_abbr,
            "team": team,
            "canonical_id": f"DST_{team}",
            "position": "DST",
        })
    df = pl.DataFrame(rows).unique(subset=["team"], keep="first").sort("team")
    if df.height != 32:
        missing = sorted(set(CURRENT_TEAMS) - set(df["team"].to_list()))
        raise CrosswalkError(
            f"D/ST crosswalk built {df.height} teams, expected 32. Missing: {missing}"
        )
    if df["espn_id"].n_unique() != 32 or df["canonical_id"].n_unique() != 32:
        raise CrosswalkError("D/ST crosswalk is not injective.")
    return df


def resolve_dst(rows: pl.DataFrame) -> pl.DataFrame:
    """Resolve ESPN D/ST entries to DST_<team> canonical ids.

    Accepts either the negative ESPN id (how draft history stores them) or a
    team abbreviation (how the draftable pool stores them).
    """
    x = cw_dst = dst_crosswalk()
    out = rows.with_columns(pl.col("espn_id").cast(pl.Utf8))

    out = out.join(
        cw_dst.select("espn_id", pl.col("canonical_id").alias("_by_id")),
        on="espn_id", how="left",
    )
    if "team" in rows.columns:
        out = out.with_columns(
            pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("_team")
        ).join(
            cw_dst.select(pl.col("team").alias("_team"),
                          pl.col("canonical_id").alias("_by_team")),
            on="_team", how="left",
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("_by_team"))

    out = out.with_columns(
        pl.coalesce("_by_id", "_by_team").alias("canonical_id"),
        pl.when(pl.col("_by_id").is_not_null()).then(pl.lit("dst_espn_id"))
        .when(pl.col("_by_team").is_not_null()).then(pl.lit("dst_team_abbr"))
        .otherwise(pl.lit("unresolved")).alias("match_method"),
        pl.lit(True).alias("has_nflverse_data"),
    )
    drop = [c for c in ("_by_id", "_by_team", "_team") if c in out.columns]
    return out.drop(drop)


def is_dst(position: str | None, espn_id: str | None = None) -> bool:
    """True for a team defense in either ESPN spelling."""
    if position and position.replace("/", "").upper() in {"DST", "DEFST"}:
        return True
    if espn_id:
        try:
            return int(espn_id) <= DST_ID_OFFSET
        except (TypeError, ValueError):
            return False
    return False


# ─── Name normalization — DIAGNOSTIC AND EXACT-TIER USE ONLY ────────────────
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str | None) -> str | None:
    """Lowercase, strip accents/punctuation/suffixes, collapse whitespace.

    Used for the tier-3 EXACT triple match and for readable diagnostics.
    It is never used for approximate or partial matching.
    """
    if not name:
        return None
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", " ").replace("'", "").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    parts = [p for p in s.split() if p and p not in _SUFFIXES]
    return " ".join(parts) or None


_norm_name_expr = (
    pl.col("display_name")
    .map_elements(normalize_name, return_dtype=pl.Utf8)
    .alias("normalized_name")
)


# ─── Canonical player table ─────────────────────────────────────────────────
def canonical_players(refresh: bool = False) -> pl.DataFrame:
    """The master player table. canonical_id == gsis_id (never null in nflverse).

    ``status`` is carried deliberately: it is the direct guard against §6's
    named failure mode, a board recommending a retired player.
    """
    p = nv.players(force=refresh)
    missing_gsis = p.filter(pl.col("gsis_id").is_null()).height
    if missing_gsis:
        raise CrosswalkError(
            f"{missing_gsis} nflverse players have a null gsis_id. canonical_id "
            f"cannot be derived; the assumption that gsis_id is total has broken."
        )
    out = (
        p.select(
            pl.col("gsis_id").alias("canonical_id"),
            "gsis_id",
            pl.col("espn_id").cast(pl.Utf8),
            "display_name",
            "position",
            "position_group",
            "latest_team",
            "status",
            "rookie_season",
            "last_season",
            "years_of_experience",
            "draft_year",
            "birth_date",
        )
        .with_columns(
            _norm_name_expr,
            pl.col("latest_team")
            .map_elements(normalize_team, return_dtype=pl.Utf8)
            .alias("team"),
        )
    )
    dupes = (
        out.filter(pl.col("espn_id").is_not_null())
        .group_by("espn_id").len().filter(pl.col("len") > 1)
    )
    if dupes.height:
        offenders = out.join(dupes.select("espn_id"), on="espn_id", how="semi")
        raise CrosswalkError(
            f"{dupes.height} espn_id(s) map to more than one canonical_id — the "
            f"crosswalk is not injective and joins would silently merge players.\n"
            f"{offenders.select('espn_id', 'canonical_id', 'display_name', 'position')}"
        )
    return out


class CrosswalkError(RuntimeError):
    """A blocking crosswalk failure. Never downgrade this to a warning."""


# ─── Overrides ───────────────────────────────────────────────────────────────
def _empty_overrides() -> pl.DataFrame:
    return pl.DataFrame(schema={c: pl.Utf8 for c in OVERRIDE_COLUMNS})


def load_overrides() -> pl.DataFrame:
    """Human-curated espn_id -> gsis_id decisions. Created empty if absent."""
    if not OVERRIDE_PATH.exists():
        OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _empty_overrides().write_csv(OVERRIDE_PATH)
        return _empty_overrides()
    df = pl.read_csv(OVERRIDE_PATH, schema_overrides={c: pl.Utf8 for c in OVERRIDE_COLUMNS})
    missing = [c for c in OVERRIDE_COLUMNS if c not in df.columns]
    if missing:
        raise CrosswalkError(
            f"{OVERRIDE_PATH} is missing column(s) {missing}. "
            f"Expected header: {','.join(OVERRIDE_COLUMNS)}"
        )
    df = df.filter(pl.col("espn_id").is_not_null() & (pl.col("espn_id").str.strip_chars() != ""))
    dupes = df.group_by("espn_id").len().filter(pl.col("len") > 1)
    if dupes.height:
        raise CrosswalkError(
            f"{OVERRIDE_PATH} lists the same espn_id more than once:\n{dupes}"
        )
    return df.select(OVERRIDE_COLUMNS)


def mark_no_nflverse_match(espn_id: str, player_name: str, reason: str) -> None:
    """Record that an ESPN player has NO nflverse counterpart.

    Only after actually searching. The evidence belongs in ``reason``.
    """
    add_override(espn_id, NO_NFLVERSE_MATCH, player_name, reason)


def add_override(espn_id: str, gsis_id: str, player_name: str, reason: str) -> None:
    """Record a human decision. Appends; never rewrites an existing row."""
    cur = load_overrides()
    if cur.filter(pl.col("espn_id") == str(espn_id)).height:
        raise CrosswalkError(f"espn_id {espn_id} already has an override; edit the CSV by hand.")
    row = pl.DataFrame({
        "espn_id": [str(espn_id)], "gsis_id": [str(gsis_id)],
        "player_name": [player_name], "reason": [reason],
        "added_at": [datetime.now(timezone.utc).isoformat()],
    })
    pl.concat([cur, row]).write_csv(OVERRIDE_PATH)


# ─── Resolution ──────────────────────────────────────────────────────────────
def resolve_players(
    espn_players: pl.DataFrame,
    canonical: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Resolve ESPN player rows to canonical IDs.

    ``espn_players`` needs columns: espn_id, name, position, team (team optional).
    Returns one row per input with ``canonical_id`` and ``match_method``.
    """
    canon = canonical_players() if canonical is None else canonical
    ov = load_overrides()

    src = espn_players.with_columns(
        pl.col("espn_id").cast(pl.Utf8),
        pl.col("name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("_nname"),
        pl.col("team").map_elements(normalize_team, return_dtype=pl.Utf8).alias("_team")
        if "team" in espn_players.columns
        else pl.lit(None, dtype=pl.Utf8).alias("_team"),
    )

    # tier 1 — human override
    src = src.join(
        ov.select(pl.col("espn_id"), pl.col("gsis_id").alias("_ov_id")),
        on="espn_id", how="left",
    )

    # tier 2 — direct espn_id
    src = src.join(
        canon.filter(pl.col("espn_id").is_not_null())
             .select("espn_id", pl.col("canonical_id").alias("_id_espn")),
        on="espn_id", how="left",
    )

    # tier 3 — EXACT (name, position, team), accepted only when unique
    triple = (
        canon.filter(pl.col("normalized_name").is_not_null() & pl.col("team").is_not_null())
        .group_by(["normalized_name", "position", "team"])
        .agg(pl.col("canonical_id").alias("ids"))
        .with_columns(pl.col("ids").list.len().alias("n"))
    )
    unique_triple = (
        triple.filter(pl.col("n") == 1)
        .select(
            pl.col("normalized_name").alias("_nname"),
            pl.col("position"),
            pl.col("team").alias("_team"),
            pl.col("ids").list.first().alias("_id_triple"),
        )
    )
    src = src.join(unique_triple, on=["_nname", "position", "_team"], how="left")

    is_espn_only = pl.col("_ov_id") == NO_NFLVERSE_MATCH

    resolved = src.with_columns(
        pl.when(is_espn_only)
        .then(pl.concat_str([pl.lit("ESPN_"), pl.col("espn_id")]))
        .otherwise(pl.coalesce("_ov_id", "_id_espn", "_id_triple"))
        .alias("canonical_id"),
        pl.when(is_espn_only).then(pl.lit("espn_only"))
        .when(pl.col("_ov_id").is_not_null()).then(pl.lit("override"))
        .when(pl.col("_id_espn").is_not_null()).then(pl.lit("espn_id"))
        .when(pl.col("_id_triple").is_not_null()).then(pl.lit("name_pos_team"))
        .otherwise(pl.lit("unresolved"))
        .alias("match_method"),
    ).with_columns(
        (pl.col("match_method") != "espn_only").alias("has_nflverse_data")
    ).drop("_ov_id", "_id_espn", "_id_triple", "_nname", "_team")

    return resolved


def assert_all_resolved(resolved: pl.DataFrame, label: str) -> pl.DataFrame:
    """CLAUDE.md §0.2 gate. Writes a fix-it report before raising."""
    unresolved = resolved.filter(pl.col("match_method") == "unresolved")
    collisions = (
        resolved.filter(pl.col("canonical_id").is_not_null())
        .group_by("canonical_id").len().filter(pl.col("len") > 1)
    )

    problems = []
    if unresolved.height:
        problems.append(f"{unresolved.height} player(s) resolved to NO canonical id")
    if collisions.height:
        problems.append(
            f"{collisions.height} canonical id(s) claimed by MORE THAN ONE espn_id "
            f"(two players silently merged)"
        )
    if not problems:
        return resolved

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / f"unmatched_{label}.csv"
    rows = unresolved
    if collisions.height:
        rows = pl.concat(
            [unresolved, resolved.join(collisions.select("canonical_id"), on="canonical_id", how="semi")],
            how="diagonal",
        ).unique()
    rows.write_csv(report)

    raise CrosswalkError(
        f"ID CROSSWALK FAILED for {label}: " + "; ".join(problems) + ".\n"
        f"  Fix-it report: {report}\n"
        f"  Resolve each row by adding espn_id,gsis_id to {OVERRIDE_PATH}\n"
        f"  (helper: ff_agent.data.crosswalk.add_override).\n"
        f"  Do NOT relax this assertion and do NOT add fuzzy matching."
    )


STALE_SEASONS_THRESHOLD = 10
"""A resolved player whose nflverse record ends this many seasons before the
target season is almost certainly a WRONG JOIN, not a comeback."""


def implausible_resolutions(
    resolved: pl.DataFrame,
    season: int,
    canonical: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Catch joins that RESOLVED but resolved to the wrong person.

    §11's gate — "every player resolves to exactly one ID" — cannot catch this
    class of error, and we hit a live one: nflverse lists ``espn_id 4686658``
    against a defensive back who last played in **1984**, but that ESPN id
    actually belongs to a 2026 rookie running back who was ~10% rostered. The
    direct espn_id tier, our highest-confidence tier, therefore produced exactly
    the failure §6 warns about — a board recommending a long-retired player —
    while passing every assertion.

    So: a resolved player whose nflverse career ended more than
    STALE_SEASONS_THRESHOLD seasons ago is flagged. Position mismatch alone is
    NOT used, because genuine two-way and hybrid players exist (Travis Hunter is
    a CB in nflverse and a WR in ESPN, and that match is correct).
    """
    canon = canonical_players() if canonical is None else canonical
    joined = resolved.join(
        canon.select(
            "canonical_id",
            pl.col("position").alias("nflverse_position"),
            pl.col("display_name").alias("nflverse_name"),
            pl.col("last_season").cast(pl.Int64, strict=False),
        ),
        on="canonical_id", how="left",
    )
    cutoff = season - STALE_SEASONS_THRESHOLD
    return (
        joined.filter(pl.col("last_season").is_not_null() & (pl.col("last_season") < cutoff))
        .with_columns(
            (pl.lit("nflverse career ended ") + pl.col("last_season").cast(pl.Utf8)
             + pl.lit(f", more than {STALE_SEASONS_THRESHOLD} seasons before {season}"))
            .alias("implausible_reason")
        )
    )


def assert_resolutions_plausible(
    resolved: pl.DataFrame, label: str, season: int,
    canonical: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Blocking companion to assert_all_resolved(). See implausible_resolutions()."""
    bad = implausible_resolutions(resolved, season, canonical)
    if bad.height:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report = REPORTS_DIR / f"implausible_{label}.csv"
        bad.write_csv(report)
        cols = [c for c in ("espn_id", "name", "position", "canonical_id",
                            "nflverse_name", "nflverse_position", "last_season")
                if c in bad.columns]
        raise CrosswalkError(
            f"{label}: {bad.height} player(s) RESOLVED TO THE WRONG PERSON.\n"
            f"{bad.select(cols)}\n"
            f"  Report: {report}\n"
            f"  These passed the 'exactly one id' gate but joined onto a player\n"
            f"  whose career ended long ago — nflverse's espn_id is wrong for them.\n"
            f"  Fix with an explicit override mapping the ESPN id to the correct\n"
            f"  gsis_id. Do NOT widen the threshold to make this pass."
        )
    return resolved


def resolution_summary(resolved: pl.DataFrame) -> pl.DataFrame:
    return (
        resolved.group_by("match_method").len()
        .rename({"len": "n"})
        .with_columns((pl.col("n") / resolved.height * 100).round(2).alias("pct"))
        .sort("n", descending=True)
    )
