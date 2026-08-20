"""Scoring rules, loaded from ESPN rather than hardcoded.

§7.1 requires recomputed scores to match ESPN *exactly*. The surest way to fail
that is to hand-transcribe 48 rules. So the rules are LOADED from the league's
own ``scoring_format``, and §1 is demoted from source-of-truth to an assertion
against what was loaded.

That inversion matters twice over:
  * a transcription slip in §1 cannot silently corrupt the engine, and
  * if a commissioner changes a setting mid-season, the assertion fires instead
    of the board quietly becoming wrong.

**Absence means zero.** ESPN omits zero-valued rules entirely, so the D/ST
buckets ``18-21``, ``22-27`` and ``300-349`` do not appear in the payload at all.
That absence is load-bearing — §1 records all three as 0, and a scored game
confirms it (Buccaneers D/ST allowed exactly 27 points in week 3 2025;
``defensive22To27PointsAllowed`` fires in the stat line and contributes nothing).
So the absent rules are asserted as absent, not merely defaulted.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ff_agent.config import ARTIFACTS_DIR, SEASON

# ─── §1, verbatim — the ASSERTION, not the source ───────────────────────────
SPEC_RULES: dict[str, float] = {
    # passing
    "PY": 0.04, "PTD": 6.0, "INTT": -2.0, "SKD": -1.0, "2PC": 2.0,
    # rushing
    "RY": 0.1, "RA": 0.05, "RTD": 6.0, "2PR": 2.0,
    # receiving
    "REY": 0.1, "REC": 0.5, "RETD": 6.0, "2PRE": 2.0,
    # misc
    "FUML": -2.0, "KRTD": 6.0, "PRTD": 6.0, "FTD": 6.0,
    # kicking
    "PAT": 1.0, "FG0": 3.0, "FG40": 4.0, "FG50": 5.0, "FG60": 6.0, "FGM": -1.0,
    # D/ST events
    "SK": 1.0, "INT": 2.0, "FR": 2.0, "SF": 2.0, "BLKK": 2.0,
    "INTTD": 6.0, "FRTD": 6.0, "BLKKRTD": 6.0, "2PRET": 2.0, "1PSF": 1.0,
    # D/ST points allowed
    "PA0": 5.0, "PA1": 4.0, "PA7": 3.0, "PA14": 1.0,
    "PA28": -1.0, "PA35": -3.0, "PA46": -5.0,
    # D/ST yards allowed
    "YA100": 5.0, "YA199": 3.0, "YA299": 2.0,
    "YA399": -1.0, "YA449": -3.0, "YA499": -5.0, "YA549": -6.0, "YA550": -7.0,
}

MUST_BE_ABSENT: frozenset[str] = frozenset({"PA18", "PA22", "YA349"})
"""Buckets §1 records as 0. ESPN omits zero-valued rules, so these must not
appear. If one ever shows up, the league changed and the board is now wrong."""

SPEC_SEASONS: frozenset[int] = frozenset({2025, 2026})
"""Seasons whose ESPN settings match §1 exactly (verified 2026-08-20).

**The league changed its scoring after 2024.** 2023 and 2024 additionally scored
``PC`` (0.25 per completion) and ``INC`` (-0.1 per incompletion); both were
removed for 2025. So only 2025 is a valid exact-match target for §11's Milestone
2 test, and historical fantasy points as ESPN RECORDED them are not comparable
across that boundary. Recomputing from stat lines under current rules — which is
what this engine does, and what §7.2 step 2 requires — is the correct handling."""


class ScoringRulesError(RuntimeError):
    """Loaded rules disagree with §1. Blocking — never score through this."""


def settings_path(season: int = SEASON) -> Path:
    return ARTIFACTS_DIR / f"espn_settings_{season}.json"


def load_rules(season: int = SEASON, verify: bool = True) -> dict[str, float]:
    """Load rules from the cached ESPN settings, cross-checked against §1.

    Falls back to SPEC_RULES only when the settings artifact is absent, so the
    draft-day offline path still works. That fallback is announced, never silent.
    """
    path = settings_path(season)
    if not path.exists():
        if verify:
            raise ScoringRulesError(
                f"No ESPN settings at {path}.\n"
                f"  Run: uv run python -m ff_agent.cli settings --season {season}\n"
                f"  (or call load_rules(verify=False) to fall back to §1 offline)."
            )
        return dict(SPEC_RULES)

    payload = json.loads(path.read_text())
    fmt = payload.get("scoring_format")
    if not fmt:
        raise ScoringRulesError(f"{path} has no scoring_format block.")

    rules = {r["abbr"]: float(r["points"]) for r in fmt}
    if verify and season in SPEC_SEASONS:
        assert_matches_spec(rules)
    return rules


def assert_matches_spec(rules: dict[str, float]) -> None:
    """Blocking three-way check: values, absences, and unexpected extras."""
    problems: list[str] = []

    for abbr, expected in SPEC_RULES.items():
        actual = rules.get(abbr)
        if actual is None:
            problems.append(f"  {abbr}: §1 expects {expected}, ESPN has NO SUCH RULE")
        elif abs(actual - expected) > 1e-9:
            problems.append(f"  {abbr}: §1 expects {expected}, ESPN has {actual}")

    for abbr in sorted(MUST_BE_ABSENT):
        if abbr in rules:
            problems.append(
                f"  {abbr}: §1 records this bucket as 0 (absent), but ESPN now "
                f"scores it {rules[abbr]}"
            )

    extra = {a: p for a, p in rules.items()
             if a not in SPEC_RULES and abs(p) > 1e-9}
    for abbr, pts in sorted(extra.items()):
        problems.append(f"  {abbr}: ESPN scores {pts} for a rule §1 does not list")

    if problems:
        raise ScoringRulesError(
            "Loaded scoring rules disagree with FANTASY_SPEC.md §1:\n"
            + "\n".join(problems)
            + "\n  Either the league settings changed or §1 is wrong. Resolve "
              "before scoring anything — every downstream number depends on this."
        )


def rules_table(rules: dict[str, float] | None = None) -> pl.DataFrame:
    r = load_rules() if rules is None else rules
    return (
        pl.DataFrame({"abbr": list(r), "points": list(r.values())})
        .with_columns(pl.col("abbr").is_in(list(SPEC_RULES)).alias("in_spec"))
        .sort("abbr")
    )
