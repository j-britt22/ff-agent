"""The draft that actually happened, reduced to one canonical frame.

Two sources, and they are not interchangeable:

* **The session log** written by ``cli live`` / ``cli gui``. §6 makes the
  human's manual entry the source of truth during the draft, so the log is the
  authoritative record of a draft this tool sat through. It also carries the
  manager on every pick, which ESPN's export does not spell the same way.
* **ESPN's draft export**, for a draft the tool did not watch, or as an
  after-the-fact cross-check. Slower, needs the network, and needs the §0.2
  crosswalk to turn ``espn_id`` into a canonical id.

§0.2 governs both. Every pick must resolve to exactly one canonical id or this
module raises — an unresolved pick is a blocking error, not a footnote, because
a roster with a hole in it silently understates a team and the standings that
fall out of it are then wrong for a reason nobody can see.

Being ABSENT FROM THE BOARD is a different failure from being unresolvable, and
is handled differently: it is a real thing a manager can do (ESPN's pool is
deeper than 555 players) and it is recorded, flagged, and valued at whatever the
projections know about him — never quietly dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ff_agent.config import (
    LOGS_DIR, N_TEAMS, ROSTER_TOTAL, SEASON, normalize_team_name,
)
from ff_agent.draft import pool as PL


@dataclass
class DraftRecord:
    """Every pick, in order, joined to the board it was made against."""

    picks: pl.DataFrame
    pool: PL.DraftPool
    source: str
    season: int
    n_teams: int
    rounds: int
    managers: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def total_picks(self) -> int:
        return self.n_teams * self.rounds

    @property
    def complete(self) -> bool:
        return self.picks.height >= self.total_picks

    @property
    def n_picks(self) -> int:
        return self.picks.height

    def roster_indices(self, manager: str) -> list[int]:
        return (
            self.picks.filter(pl.col("manager") == manager)
            .sort("overall")["pool_index"].to_list()
        )

    def summary(self) -> str:
        state = "COMPLETE" if self.complete else "IN PROGRESS"
        return (
            f"{self.season} draft · {self.n_picks}/{self.total_picks} picks · "
            f"{self.n_teams} teams x {self.rounds} rounds · {state} · "
            f"source={self.source}"
        )


# ── session log ──────────────────────────────────────────────────────────────
def session_logs(season: int = SEASON) -> list[Path]:
    return sorted(LOGS_DIR.glob(f"draft_{season}_slot*.jsonl"))


def _read_log(path: Path) -> tuple[list[dict], dict]:
    recs = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    session = next((r for r in recs if r.get("kind") == "session"), {})
    return recs, session


def _replay(records: list[dict]) -> list[dict]:
    """Apply picks and undos in order, keyed on each pick's own ``overall``.

    Replaying on ``overall`` rather than counting ``undo`` records is deliberate.
    The GUI logs an undo; the terminal loop does not (``cmd_live`` only logs
    picks), so a terminal session can contain a pick that was later taken back
    with nothing marking it. Both cases look identical here — the next pick
    simply arrives with an ``overall`` that has already been used — and both
    replay correctly. A GAP, by contrast, cannot be explained away and raises.
    """
    hist: list[dict] = []
    for rec in records:
        if rec.get("kind") != "pick":
            continue
        o = int(rec["overall"])
        if o < 1:
            raise ValueError(f"pick with non-positive overall {o}: {rec}")
        del hist[o - 1:]
        if len(hist) != o - 1:
            raise ValueError(
                f"draft log jumps from pick {len(hist)} to pick {o}. Picks are "
                "missing, so every roster after this point would be wrong. Fix "
                "the log or read the draft from ESPN instead."
            )
        hist.append(rec)
    return hist


def pick_log(path: Path) -> tuple[pl.DataFrame, dict, list[str]]:
    recs, session = _read_log(path)
    hist = _replay(recs)
    undone = sum(1 for r in recs if r.get("kind") == "pick") - len(hist)
    notes = []
    if undone:
        notes.append(f"{undone} pick(s) were undone during the draft and are excluded")
    if not hist:
        raise ValueError(f"{path} contains no picks")
    df = pl.DataFrame([
        {
            "overall": int(r["overall"]),
            "manager": r["team"],
            "name": r["player"],
            "canonical_id": r["canonical_id"],
            "position": r["position"],
            "mine": bool(r.get("mine", False)),
            "entry_source": r.get("source", "manual"),
        }
        for r in hist
    ]).sort("overall")
    return df, session, notes


def latest_log(season: int = SEASON) -> Path | None:
    """The log holding the most picks for this season.

    Not simply the newest file: opening the GUI writes a session record
    immediately, so an accidental relaunch after the draft leaves a newer log
    holding nothing. Ties break to the newer file.
    """
    best, best_n = None, 0
    for p in session_logs(season):
        try:
            recs, _ = _read_log(p)
            n = len(_replay(recs))
        except Exception:
            continue
        if n >= best_n and n > 0:
            best, best_n = p, n
    return best


# ── ESPN export ──────────────────────────────────────────────────────────────
def _from_espn(season: int, n_teams: int) -> tuple[pl.DataFrame, list[str]]:
    from ff_agent.data import crosswalk as cw
    from ff_agent.data import espn

    d = espn.draft_results(season)
    canon = cw.canonical_players().select("espn_id", "canonical_id")
    rows, unresolved = [], []
    dst = None
    for r in d.sort(["round", "pick"]).to_dicts():
        espn_id = str(r.get("espn_id") or "")
        cid = None
        if espn_id.startswith("-"):
            # §0.2 corollary: D/ST are team entities stored as -16000 - proTeamId
            if dst is None:
                dst = cw.dst_crosswalk()
            hit = dst.filter(pl.col("espn_id") == espn_id)
            cid = hit["canonical_id"][0] if hit.height == 1 else None
        else:
            hit = canon.filter(pl.col("espn_id") == espn_id)
            cid = hit["canonical_id"][0] if hit.height == 1 else None
        if cid is None:
            unresolved.append(f"{r.get('name')} (espn_id {espn_id or 'missing'})")
            continue
        rows.append({
            "overall": (int(r["round"]) - 1) * n_teams + int(r["pick"]),
            "manager": normalize_team_name(r.get("fantasy_team")),
            # carried because TEAM NAMES CHANGE and the two ESPN tables cache
            # independently: on draft night one manager renamed her team and the
            # export said "Chase-ing Dubs" while the cached schedule still said
            # "leah's team". team_id is the only field they agree on.
            "team_id": r.get("team_id"),
            "name": r.get("name"),
            "canonical_id": cid,
            "position": None,
            "mine": False,
            "entry_source": "espn",
        })
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} drafted player(s) did not resolve to exactly one "
            f"canonical id (§0.2). This is a blocking error — add an override row "
            f"in overrides/player_id_overrides.csv rather than dropping them:\n  "
            + "\n  ".join(unresolved)
        )
    return pl.DataFrame(rows).sort("overall"), []


# ── joining picks to the board ───────────────────────────────────────────────
def attach(
    picks: pl.DataFrame, pool: PL.DraftPool, season: int
) -> tuple[pl.DataFrame, PL.DraftPool, list[str]]:
    """Give every pick a pool index, extending the pool for off-board picks."""
    ids = pool.df["canonical_id"].to_list()
    index_of = {c: i for i, c in enumerate(ids)}
    notes: list[str] = []

    unknown = [
        r for r in picks.to_dicts() if r["canonical_id"] not in index_of
    ]
    if unknown:
        extra = _extend_pool(pool, unknown, season)
        pool = extra[0]
        notes.extend(extra[1])
        ids = pool.df["canonical_id"].to_list()
        index_of = {c: i for i, c in enumerate(ids)}

    pos = pool.df["position"].to_list()
    nfl = pool.df["team"].to_list()
    picks = picks.with_columns(
        pl.col("canonical_id").replace_strict(index_of, return_dtype=pl.Int64)
        .alias("pool_index")
    ).with_columns(
        pl.col("pool_index").replace_strict(dict(enumerate(pos)), return_dtype=pl.Utf8)
        .alias("position"),
        pl.col("pool_index").replace_strict(dict(enumerate(nfl)), return_dtype=pl.Utf8)
        .alias("nfl_team"),
    )

    dupes = picks.group_by("canonical_id").len().filter(pl.col("len") > 1)
    if dupes.height:
        detail = picks.filter(
            pl.col("canonical_id").is_in(dupes["canonical_id"].implode())
        ).select("overall", "manager", "name", "canonical_id").sort("canonical_id")
        raise ValueError(
            f"{dupes.height} player(s) were drafted more than once. Two fantasy "
            f"teams cannot own the same person; the draft record is corrupt "
            f"(§0.2).\n{detail}"
        )
    return picks, pool, notes


def _extend_pool(
    pool: PL.DraftPool, unknown: list[dict], season: int
) -> tuple[PL.DraftPool, list[str]]:
    from ff_agent.projections import board_inputs as BI

    want = pl.DataFrame([
        {"canonical_id": r["canonical_id"], "name": r["name"],
         "espn_position": r.get("position")}
        for r in unknown
    ]).unique(subset=["canonical_id"])

    from ff_agent.board import replacement as REP
    from ff_agent.projections import calibration as CAL

    proj = BI.build(season).select(
        "canonical_id", pl.col("position").alias("proj_position"),
        pl.col("team").alias("proj_team"), "blended_points", "points_sd",
        "bye_week", "free_bye_week_5_or_14",
    )
    # VOR has to be computed, not left null: `_to_pool` fills a null VOR with
    # -999, which would make every off-board pick read as the worst pick in the
    # draft. Replacement level is the same one the board used.
    repl = REP.replacement_table(
        CAL.smooth_curve(CAL.rank_points_curve(seasons=list(range(2016, season))))
    ).select(pl.col("position").alias("proj_position"), "replacement_points")
    j = want.join(proj, on="canonical_id", how="left").join(
        repl, on="proj_position", how="left"
    )
    no_proj = j.filter(pl.col("blended_points").is_null())["name"].to_list()

    rows = j.with_columns(
        pl.coalesce("proj_position", "espn_position").alias("position"),
        pl.col("proj_team").alias("team"),
        pl.col("blended_points").fill_null(0.0),
        pl.col("points_sd").fill_null(0.0),
        pl.col("bye_week").fill_null(0).cast(pl.Int8),
        pl.col("free_bye_week_5_or_14").fill_null(False),
        pl.lit(1.5).alias("consensus_fraction"),
        (pl.col("blended_points").fill_null(0.0)
         - pl.col("replacement_points").fill_null(0.0)).alias("vor"),
        pl.lit(None, dtype=pl.Float64).alias("ecr"),
        pl.lit(0, dtype=pl.Int16).alias("tier"),
        pl.lit("off_board").alias("value_source"),
    ).select(pool.df.columns)

    bad = rows.filter(pl.col("position").is_null())
    if bad.height:
        raise ValueError(
            f"{bad.height} drafted player(s) have no position from either the "
            f"projections or the draft record, so no roster slot can be assigned "
            f"(§0.2): {bad['name'].to_list()}"
        )
    notes = [
        f"{rows.height} pick(s) are OFF THE BOARD — drafted from deeper in "
        f"ESPN's pool than the 555 this tool ranks: "
        f"{', '.join(rows['name'].to_list()[:8])}"
        + (" ..." if rows.height > 8 else "")
    ]
    if no_proj:
        notes.append(
            f"{len(no_proj)} of those have NO projection at all and are valued at "
            f"zero, which UNDERSTATES their team: {', '.join(no_proj[:8])}"
        )
    return PL._to_pool(pl.concat([pool.df, rows])), notes


# ── the entry point ──────────────────────────────────────────────────────────
def load(
    season: int = SEASON,
    source: str = "auto",
    log_path: str | Path | None = None,
    pool: PL.DraftPool | None = None,
    n_teams: int | None = None,
    rounds: int = ROSTER_TOTAL,
) -> DraftRecord:
    """Load the completed (or in-progress) draft from ``source``."""
    from ff_agent.config import ARTIFACTS_DIR

    notes: list[str] = []
    picks: pl.DataFrame | None = None
    managers: list[str] = []
    used = source

    if source in ("auto", "log"):
        path = Path(log_path) if log_path else latest_log(season)
        if path is not None and Path(path).exists():
            picks, session, n = pick_log(Path(path))
            managers = list(session.get("managers") or [])
            notes.extend(n)
            notes.append(f"read from session log {Path(path).name}")
            used = "log"
            others = _overlapping_logs(Path(path), season)
            if others:
                notes.append(
                    f"WARNING: {len(others)} other log(s) for {season} hold picks "
                    f"made DURING this session's window "
                    f"({', '.join(p.name for p in others)}). The draft may have "
                    "been split across sessions, which would make this record "
                    "incomplete — pass --log explicitly or read from ESPN."
                )
        elif source == "log":
            raise FileNotFoundError(
                f"no session log with picks for {season} in {LOGS_DIR}"
            )

    if picks is None:
        picks, n = _from_espn(season, n_teams or N_TEAMS)
        notes.extend(n)
        notes.append(f"read from ESPN's draft export for {season}")
        used = "espn"
        managers = _managers_in_pick_order(picks, n_teams or N_TEAMS)

    if not managers:
        managers = _managers_in_pick_order(picks, n_teams or N_TEAMS)
    n_teams = n_teams or len(managers) or N_TEAMS

    if pool is None:
        pool = PL.load_pool(ARTIFACTS_DIR) or PL.build_pool(season, n_teams=n_teams)

    picks, pool, n = attach(picks, pool, season)
    notes.extend(n)

    picks = picks.with_columns(
        (((pl.col("overall") - 1) // n_teams) + 1).alias("round"),
        (((pl.col("overall") - 1) % n_teams) + 1).alias("pick_in_round"),
    )
    return DraftRecord(
        picks=picks, pool=pool, source=used, season=season, n_teams=n_teams,
        rounds=rounds, managers=managers, notes=notes,
    )


def _overlapping_logs(chosen: Path, season: int) -> list[Path]:
    """Other logs whose picks fall inside the chosen session's window.

    An earlier rehearsal log is not evidence of a split draft — this machine has
    several — so plain "another log has picks" is too noisy to be read. What
    would actually corrupt the record is a SECOND session recording picks while
    this one was live, and that is what this looks for.
    """
    span = _pick_span(chosen)
    if span is None:
        return []
    lo, hi = span
    out = []
    for p in session_logs(season):
        if p == chosen:
            continue
        other = _pick_span(p)
        if other and other[1] >= lo and other[0] <= hi:
            out.append(p)
    return out


def _pick_span(path: Path) -> tuple[str, str] | None:
    try:
        recs, _ = _read_log(path)
    except Exception:
        return None
    ts = [r["ts"] for r in recs if r.get("kind") == "pick" and r.get("ts")]
    return (min(ts), max(ts)) if ts else None


def _safe_n(path: Path) -> int:
    try:
        recs, _ = _read_log(path)
        return len(_replay(recs))
    except Exception:
        return 0


def _managers_in_pick_order(picks: pl.DataFrame, n_teams: int) -> list[str]:
    """Recover the snake order from the first round's picks."""
    first = picks.filter(pl.col("overall") <= n_teams).sort("overall")
    seen: list[str] = []
    for m in first["manager"].to_list():
        if m not in seen:
            seen.append(m)
    return seen
