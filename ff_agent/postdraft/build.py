"""``postdraft.json`` — the draft, graded, rated, and projected forward."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import polars as pl

from ff_agent.config import (
    ARTIFACTS_DIR, MY_TEAM_NAME, SEASON, normalize_team_name,
)
from ff_agent.draft.build import MY_MANAGER
from ff_agent.postdraft import finish as FI
from ff_agent.postdraft import grade as GR
from ff_agent.postdraft import predict as PR
from ff_agent.postdraft import results as RS
from ff_agent.postdraft import teams as TM


@dataclass
class PostDraft:
    record: RS.DraftRecord
    picks: pl.DataFrame
    ratings: pl.DataFrame
    standings: pl.DataFrame
    meta: dict
    notes: list[str] = field(default_factory=list)

    @property
    def my_key(self) -> str | None:
        """Which ``manager`` value is me.

        The log records MANAGERS and ESPN's export records TEAM NAMES, so the
        same person arrives under two different strings depending on the source
        — and §1's team name is already stale for one manager in this league
        ("A Chane Reaction" is "TBD" now), which is exactly why the live code
        resolves through managers rather than names.
        """
        seen = set(self.record.managers)
        for candidate in (MY_MANAGER, normalize_team_name(MY_TEAM_NAME), MY_TEAM_NAME):
            if candidate in seen:
                return candidate
        return None

    @property
    def my_picks(self) -> pl.DataFrame:
        return (
            self.picks.filter(pl.col("manager") == self.my_key).sort("overall")
            if self.my_key else self.picks.head(0)
        )


def build(
    season: int = SEASON,
    source: str = "auto",
    log_path: str | None = None,
    n_sims: int = PR.N_SIMS,
    predict_season: bool = True,
    pad_unfinished: bool = True,
) -> PostDraft:
    record = RS.load(season, source=source, log_path=log_path)
    notes = list(record.notes)

    weeks, _, wnotes = TM.play_weeks_by_manager(record, season)
    notes.extend(wnotes)

    rosters, projected = (
        FI.project_remaining(record, weeks) if pad_unfinished
        else ({m: list(record.roster_indices(m)) for m in record.managers},
              {m: 0 for m in record.managers})
    )
    picks = GR.analyse(record, play_weeks=weeks, rosters=rosters)
    ratings, rnotes = TM.rate(record, season, rosters=rosters, projected=projected)
    notes.extend(rnotes)
    ratings = TM.draft_efficiency(ratings, picks)

    if not record.complete:
        left = record.total_picks - record.n_picks
        notes.insert(0, (
            f"THE DRAFT IS NOT FINISHED — {record.n_picks} of "
            f"{record.total_picks} picks."
            + (
                f" The remaining {left} are PROJECTED (greedy on marginal lineup "
                "value) so that unfilled starting slots do not distort every "
                "team's strength and every counterfactual. Roster shape below "
                "still reports only real picks. Re-run when the draft ends."
                if pad_unfinished else
                " Remaining picks are NOT projected, so unfilled starting slots "
                "depress every team's points and the counterfactuals will "
                "recommend filling them at every pick."
            )
        ))

    # play_weeks_by_manager runs here and again inside rate(), so the same
    # advisory can arrive twice; order is preserved because the first note is
    # the completeness banner and it has to stay first.
    notes = list(dict.fromkeys(notes))

    standings, meta = (
        PR.predict(ratings, season=season, n_sims=n_sims)
        if predict_season else (pl.DataFrame(), {"skipped": True})
    )
    meta["my_schedule"] = (
        PR.my_schedule_view(standings, ratings) if predict_season else None
    )
    return PostDraft(
        record=record, picks=picks, ratings=ratings, standings=standings,
        meta=meta, notes=notes,
    )


def payload(pd: PostDraft) -> dict:
    best = pd.picks.sort("regret_ratio").head(10)
    worst = pd.picks.sort("regret_ratio", descending=True).head(10)
    return {
        "season": pd.record.season,
        "source": pd.record.source,
        "complete": pd.record.complete,
        "picks_recorded": pd.record.n_picks,
        "picks_expected": pd.record.total_picks,
        "my_team": MY_TEAM_NAME,
        "my_manager": pd.my_key,
        "notes": pd.notes,
        "grading": {
            "measure": (
                "Regret in weekly starting-lineup points against the best legal "
                "PAIR of players the manager could have taken at this pick and "
                "their next, holding everyone else's picks fixed."
            ),
            "why_a_pair": (
                "M8: 'Josh Allen has the highest VOR on the board (167.5) and is "
                "the THIRD best choice, because he comes back 97% of the time.' "
                "Single-pick value cannot see that; a pair can."
            ),
            "why_lineup_points_not_vor": (
                "M7: VOR is measured against the STARTER replacement, so it "
                "prices QB3 and QB4 as though they would start. Scoring pairs by "
                "VOR handed round one a wall of Ds for not drafting quarterbacks "
                "nobody could start."
            ),
            "why_the_letters_are_a_curve": (
                "Hindsight regret has no meaningful zero — the better pair is "
                "chosen knowing who actually survived, and on 2025 the MEDIAN "
                "pick fell 8-10 weekly points short of it. Letters rank picks "
                "within their own round; regret_weekly_points carries the "
                "magnitude. Read the points before reacting to a letter."
            ),
            "counterfactual_assumption": (
                "The other managers are assumed to have made the same picks "
                "regardless. Unknowable, and true of any post-mortem."
            ),
            "thresholds": list(GR.GRADE_PERCENTILES),
        },
        "picks": pd.picks.to_dicts(),
        "best_picks": best.to_dicts(),
        "worst_picks": worst.to_dicts(),
        "my_picks": pd.my_picks.to_dicts(),
        "team_ratings": pd.ratings.to_dicts(),
        "projected_standings": pd.standings.to_dicts(),
        "prediction_meta": pd.meta,
    }


def write(pd: PostDraft, path=None) -> str:
    path = path or (ARTIFACTS_DIR / "postdraft.json")
    path.write_text(json.dumps(payload(pd), indent=2, default=str))
    return str(path)
