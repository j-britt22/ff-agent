"""§10: "Log every recommendation with timestamp and inputs, or the season
teaches you nothing."

One JSON object per line, appended, never rewritten. The draft is the first
thing worth logging and the hardest to reconstruct afterwards: what was on the
board, what was recommended, and what I actually did are three different things,
and only the third survives in ESPN.

This is what M10 replays.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ff_agent.config import LOGS_DIR


def session_path(season: int, slot: int, started: float | None = None) -> Path:
    stamp = datetime.fromtimestamp(
        started or time.time(), tz=timezone.utc
    ).strftime("%Y%m%d-%H%M%S")
    return LOGS_DIR / f"draft_{season}_slot{slot}_{stamp}.jsonl"


class SessionLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **payload) -> None:
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "kind": kind,
            **payload,
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def recommendation(self, advice, state) -> None:
        """Everything needed to second-guess this later, including the inputs."""
        self.write(
            "recommendation",
            pick=advice.pick,
            round=advice.round_no,
            scenario=advice.scenario,
            scenario_evidence=advice.scenario_evidence,
            n_sims=advice.n_sims,
            elapsed=advice.elapsed,
            degraded=advice.degraded,
            my_roster=[state.pool.df["name"].to_list()[i] for i in state.my_roster()],
            position_counts=state.position_counts(),
            options=[
                {
                    "name": o.name, "position": o.position, "team": o.team,
                    "vor": o.vor, "roster_value": o.roster_value,
                    "delta_vs_best": o.delta_vs_best,
                    "survival_to_next_turn": o.survival_to_next_turn,
                    "never_passed_on": o.never_passed_on,
                }
                for o in advice.options
            ],
            branches=advice.branches,
            alarms=advice.alarms,
        )

    def pick(self, state, index: int, by: str, source: str) -> None:
        self.write(
            "pick",
            overall=len(state.history),
            team=by,
            mine=by == state.managers[state.my_index],
            player=state.pool.df["name"].to_list()[index],
            canonical_id=state.pool.df["canonical_id"].to_list()[index],
            position=state.pool.df["position"].to_list()[index],
            source=source,
        )

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
