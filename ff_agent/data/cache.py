"""Parquet cache with staleness metadata.

Two jobs, both from §10:
  1. "Cache aggressively" + "must run fully offline" — draft-day WiFi will betray you.
  2. "Fail loudly on stale data" — an optimizer silently running week-3 data in
     week 8 is worse than no optimizer.

Every cached file carries a sidecar ``.meta.json`` recording when it was fetched,
what produced it, and a hash of its schema. The schema hash is what catches
nflverse renaming a column out from under us — which would otherwise surface as a
silently wrong board in Milestone 4.

Refresh policy is keyed on the fact that **a completed season is immutable**.
2016 play-by-play will never change, so it never expires. Only current-season and
season-independent tables have a TTL.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from ff_agent.config import CACHE_DIR, SEASON

# ─── Refresh policy ──────────────────────────────────────────────────────────
IMMUTABLE = None
"""TTL sentinel: a completed season never goes stale."""

DEFAULT_TTL = timedelta(days=1)

TTL_POLICY: dict[str, timedelta] = {
    # season-independent tables that churn
    "players": timedelta(days=1),
    "ff_playerids": timedelta(days=1),
    "teams": timedelta(days=30),
    "depth_charts": timedelta(hours=12),
    "injuries": timedelta(hours=6),
    # league state moves fast around the draft and on waiver day
    "espn_league": timedelta(hours=1),
    "espn_players": timedelta(hours=1),
    "espn_rosters": timedelta(hours=1),
    "espn_draft": timedelta(days=1),
    "espn_settings": timedelta(days=1),
    # In-season reads. Projections move on the injury report, so six hours; the
    # free-agent pool and lineup slots move on every transaction, so one.
    "espn_projections": timedelta(hours=6),
    "espn_current_rosters": timedelta(hours=1),
    "espn_results": timedelta(hours=6),
    # draft_history spans COMPLETED seasons (2023-2025) in one table, so it is
    # cached with season=None and effective_ttl()'s "a completed season never
    # expires" rule can never engage for it. Without an entry here it fell back
    # to DEFAULT_TTL and went stale every 24 hours — which is merely wasteful
    # online, and breaks the §0.3 offline draft-day path, since M7's board,
    # calibration and opponent model all read it. Long TTL rather than
    # IMMUTABLE: a fourth season will eventually be added to it.
    "draft_history": timedelta(days=90),
}


class StaleDataError(RuntimeError):
    """Cached data is older than its refresh policy and we cannot refetch."""


class MissingCacheError(RuntimeError):
    """Offline, and the file we need was never fetched."""


def _offline_default() -> bool:
    return os.getenv("FF_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _paths(name: str, season: int | None) -> tuple[Path, Path]:
    stem = name if season is None else f"{name}_{season}"
    d = CACHE_DIR / name
    return d / f"{stem}.parquet", d / f"{stem}.meta.json"


def schema_hash(df: pl.DataFrame) -> str:
    """Stable hash of column names + dtypes. Changes when nflverse renames."""
    payload = "|".join(f"{c}:{df.schema[c]}" for c in sorted(df.columns))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def effective_ttl(name: str, season: int | None) -> timedelta | None:
    """A completed season is immutable; everything else follows TTL_POLICY."""
    if season is not None and season < SEASON:
        return IMMUTABLE
    return TTL_POLICY.get(name, DEFAULT_TTL)


def read_meta(name: str, season: int | None) -> dict[str, Any] | None:
    _, meta_path = _paths(name, season)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None


def age_of(name: str, season: int | None) -> timedelta | None:
    meta = read_meta(name, season)
    if not meta or "fetched_at" not in meta:
        return None
    fetched = datetime.fromisoformat(meta["fetched_at"])
    return datetime.now(timezone.utc) - fetched


def is_stale(name: str, season: int | None) -> bool:
    ttl = effective_ttl(name, season)
    if ttl is IMMUTABLE:
        return False
    age = age_of(name, season)
    return age is None or age > ttl


def cached(
    name: str,
    fetch: Callable[[], pl.DataFrame],
    *,
    season: int | None = None,
    offline: bool | None = None,
    force: bool = False,
    source: str = "nflverse",
) -> pl.DataFrame:
    """Return a cached table, fetching it if absent or stale.

    In offline mode nothing is fetched: a missing file raises MissingCacheError
    and a stale one raises StaleDataError. Both are blocking, by design — the
    draft-day failure we are protecting against is silently using old data.
    """
    offline = _offline_default() if offline is None else offline
    parquet_path, meta_path = _paths(name, season)
    label = name if season is None else f"{name}[{season}]"

    fresh_enough = parquet_path.exists() and not is_stale(name, season)

    if fresh_enough and not force:
        return pl.read_parquet(parquet_path)

    if offline:
        if not parquet_path.exists():
            raise MissingCacheError(
                f"OFFLINE and {label} was never cached.\n"
                f"  Expected: {parquet_path}\n"
                f"  Fix: run the ingest online before going offline."
            )
        age = age_of(name, season)
        ttl = effective_ttl(name, season)
        raise StaleDataError(
            f"OFFLINE and {label} is stale.\n"
            f"  Age: {age}   Policy: {ttl}\n"
            f"  File: {parquet_path}\n"
            f"  Refusing to serve it. Refresh online, or pass offline=False\n"
            f"  if you have consciously decided stale data is acceptable here."
        )

    df = fetch()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)

    meta = {
        "table": name,
        "season": season,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "rows": df.height,
        "cols": df.width,
        "columns": sorted(df.columns),
        "schema_hash": schema_hash(df),
        "bytes": parquet_path.stat().st_size,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return df


def cache_inventory() -> pl.DataFrame:
    """Every cached table with its age and staleness. Used by data-validator."""
    rows = []
    for meta_path in sorted(CACHE_DIR.rglob("*.meta.json")):
        try:
            m = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        name, season = m.get("table"), m.get("season")
        age = age_of(name, season)
        ttl = effective_ttl(name, season)
        rows.append(
            {
                "table": name,
                "season": season,
                "rows": m.get("rows"),
                "cols": m.get("cols"),
                "mb": round((m.get("bytes") or 0) / 1_048_576, 2),
                "age_hours": round(age.total_seconds() / 3600, 2) if age else None,
                "policy": "immutable" if ttl is IMMUTABLE else str(ttl),
                "stale": is_stale(name, season),
                "schema_hash": m.get("schema_hash"),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "table": pl.Utf8, "season": pl.Int64, "rows": pl.Int64,
                "cols": pl.Int64, "mb": pl.Float64, "age_hours": pl.Float64,
                "policy": pl.Utf8, "stale": pl.Boolean, "schema_hash": pl.Utf8,
            }
        )
    return pl.DataFrame(rows).sort(["table", "season"])
