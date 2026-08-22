"""A local web GUI for the live draft.

Why this exists: the terminal loop (``cli live``) is the tested path, but a
draft is read at a glance, not scrolled. This serves the *same* engine over
HTTP — ``state.DraftState`` for truth, ``advise.advise`` for the recommendation,
``entry.Resolver`` for manual entry — so there is one implementation of the
thinking and two ways to look at it.

Deliberately stdlib-only (``http.server``). Draft day is the wrong time to
discover a dependency does not install, and §0.3 requires the whole path work
offline; the ESPN poll is the only thing here that touches the network and it
degrades to "nothing new" when it cannot.

**§0.1**: no write path exists. The poll reads ``league.draft``; everything else
is local. Nothing is ever submitted to ESPN — the human clicks in ESPN's own UI.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ff_agent.draft import pool as PL
from ff_agent.live import advise as AD
from ff_agent.live import log as LG
from ff_agent.live import poll as PO
from ff_agent.live.entry import Resolver
from ff_agent.live.state import DraftState

UI_PATH = Path(__file__).with_name("ui.html")


class DraftSession:
    """Everything one browser tab needs, behind a lock.

    The lock matters: the poll runs on a timer thread while the human may be
    typing a name, and two writers to ``history`` would corrupt the board in
    exactly the way §0.2 warns about.
    """

    def __init__(self, state: DraftState, resolver: Resolver, ctx, season: int,
                 slot: int, qb_cap: int | None, log: LG.SessionLog):
        self.state = state
        self.resolver = resolver
        self.tilt, self.reach, self.scen, self.targets = ctx
        self.season = season
        self.slot = slot
        self.qb_cap = qb_cap
        self.log = log
        self.lock = threading.RLock()
        self._advice: AD.Advice | None = None
        self._advice_for: int | None = None
        self._advice_error = ""
        self.index_map = PO.espn_index_map(state.pool)
        self.poll_enabled = False
        self.poll_status = "off"
        self.poll_conflict = ""
        self._last_poll = 0.0

    # ── advice, memoised per pick so re-rendering is free ────────────────────
    def advice(self) -> AD.Advice | None:
        with self.lock:
            if self.state.complete or not self.state.my_turn:
                return None
            if self._advice_for == self.state.next_pick:
                return self._advice
            try:
                a = AD.advise(
                    self.state, self.tilt, self.reach, self.scen, self.targets,
                    qb_cap=self.qb_cap,
                )
                self._advice, self._advice_error = a, ""
            except Exception as exc:
                self._advice, self._advice_error = None, f"{type(exc).__name__}: {exc}"
            self._advice_for = self.state.next_pick
            if self._advice is not None:
                self.log.recommendation(self._advice, self.state)
            return self._advice

    def _invalidate(self) -> None:
        self._advice = None
        self._advice_for = None

    # ── mutation ─────────────────────────────────────────────────────────────
    def record_query(self, query: str) -> dict:
        """Resolve a typed name and record it. Ambiguity PROMPTS, never guesses."""
        with self.lock:
            if self.state.complete:
                return {"ok": False, "message": "the draft is complete"}
            avail = set(range(self.state.pool.n)) - self.state.taken()
            m = self.resolver.resolve(query, available=avail)
            if m.ambiguous:
                return {
                    "ok": False, "ambiguous": True, "message": f"which one? ({m.reason})",
                    "candidates": [
                        {"index": i, "label": self.resolver.describe(i)}
                        for i in m.candidates
                    ],
                }
            if not m.resolved:
                return {"ok": False, "message": f"no match for {query!r} — {m.reason}"}
            return self.record_index(m.index, source="manual")

    def record_index(self, index: int, source: str = "manual") -> dict:
        with self.lock:
            try:
                who = self.state.managers[int(self.state.order[len(self.state.history)])]
                self.state.record(index)
            except (ValueError, IndexError) as e:
                return {"ok": False, "message": str(e)}
            self._invalidate()
            self.log.pick(self.state, index, who, source)
            return {"ok": True, "message": f"{len(self.state.history)}. {who}: "
                                           f"{self.resolver.describe(index)}"}

    def undo(self) -> dict:
        with self.lock:
            try:
                i = self.state.undo()
            except ValueError as e:
                return {"ok": False, "message": str(e)}
            self._invalidate()
            self.log.write("undo", pick=len(self.state.history) + 1,
                           player=self.resolver.describe(i))
            return {"ok": True, "message": f"undid: {self.resolver.describe(i)}"}

    # ── the ESPN feed (§6: convenience, never truth) ──────────────────────────
    def do_poll(self, force: bool = False) -> None:
        if not self.poll_enabled and not force:
            return
        if not force and time.time() - self._last_poll < 4.0:
            return
        self._last_poll = time.time()
        r = PO.poll_picks(self.season, self.index_map, self.state.pool)
        with self.lock:
            if not r.ok:
                self.poll_status = f"unreachable — {r.error[:70]}"
                return
            mine = self.state.history
            feed = r.indices
            # Only ever APPEND, and only where the feed agrees with what is
            # already recorded. A feed that contradicts the human is reported,
            # not applied — §6 makes manual entry the source of truth.
            common = min(len(mine), len(feed))
            if mine[:common] != feed[:common]:
                bad = next(k for k in range(common) if mine[k] != feed[k])
                self.poll_conflict = (
                    f"ESPN and this board disagree from pick {bad + 1}: "
                    f"ESPN says {self.resolver.describe(feed[bad])}, "
                    f"here it is {self.resolver.describe(mine[bad])}. "
                    f"Not auto-applying — fix with undo."
                )
                self.poll_status = "CONFLICT"
                return
            self.poll_conflict = ""
            added = 0
            for i in feed[len(mine):]:
                if i in self.state.taken():
                    continue
                who = self.state.managers[int(self.state.order[len(self.state.history)])]
                try:
                    self.state.record(i)
                except (ValueError, IndexError):
                    break
                self.log.pick(self.state, i, who, "espn_poll")
                added += 1
            if added:
                self._invalidate()
            extra = f" · {len(r.unresolved)} unplaceable" if r.unresolved else ""
            self.poll_status = (f"live · {r.n_feed_picks} picks on ESPN"
                                f"{' · +' + str(added) if added else ''}{extra}")

    # ── what the browser renders ─────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self.lock:
            st = self.state
            pos = st.pool.df["position"].to_list()
            names = st.pool.df["name"].to_list()
            teams = st.pool.df["team"].to_list()
            byes = st.pool.df["bye_week"].to_list()
            free = st.pool.df["free_bye_week_5_or_14"].to_list()

            a = self.advice()
            until = st.picks_until_my_turn()
            board = []
            avail = st.available_mask()
            for i in range(st.pool.n):
                if not avail[i]:
                    continue
                board.append({
                    "index": i, "name": names[i], "position": pos[i],
                    "team": teams[i], "vor": round(float(st.pool.vor[i]), 1),
                    "tier": int(st.pool.df["tier"].to_list()[i]),
                    "bye": byes[i], "free_bye": bool(free[i]),
                })
            board.sort(key=lambda r: -r["vor"])

            recent = []
            tail = st.history[-12:]
            first = len(st.history) - len(tail) + 1   # 1-based pick of tail[0]
            for k, idx in enumerate(tail, start=first):
                team = int(st.order[k - 1])
                recent.append({
                    "pick": k,
                    "manager": st.managers[team],
                    "name": names[idx], "position": pos[idx],
                    "mine": team == st.my_index,
                })

            out = {
                "slot": self.slot,
                "season": self.season,
                "summary": st.summary(),
                "pick": min(st.next_pick, st.total_picks),
                "total_picks": st.total_picks,
                "round": (st.next_pick - 1) // st.n_teams + 1,
                "complete": st.complete,
                "my_turn": bool(st.my_turn),
                "on_the_clock": None if st.complete else st.managers[st.on_the_clock],
                "picks_until_my_turn": until,
                "managers": st.managers,
                "my_index": st.my_index,
                "roster": [
                    {"name": names[i], "position": pos[i], "team": teams[i],
                     "bye": byes[i], "free_bye": bool(free[i])}
                    for i in st.my_roster()
                ],
                "counts": st.position_counts(),
                "alarms": st.alarms(),
                "board": board[:60],
                "recent": list(reversed(recent)),
                "poll": {"enabled": self.poll_enabled, "status": self.poll_status,
                         "conflict": self.poll_conflict},
                "advice_error": self._advice_error,
                "log_path": str(self.log.path),
            }
            if a is not None:
                out["advice"] = {
                    "headline": a.headline(),
                    "scenario": a.scenario,
                    "scenario_note": a.scenario_evidence.get("note", ""),
                    "n_sims": a.n_sims,
                    "elapsed": a.elapsed,
                    "degraded": a.degraded,
                    "options": [
                        {"index": o.index, "name": o.name, "position": o.position,
                         "team": o.team, "vor": round(o.vor, 1),
                         "delta": round(o.delta_vs_best, 2),
                         "survival": o.survival_to_next_turn,
                         "never_passed_on": o.never_passed_on,
                         "free_bye": o.free_bye}
                        for o in a.options
                    ],
                    "slipping_away": a.slipping_away[:5],
                    "branches": a.branches,
                }
            return out


def make_handler(session: DraftSession):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):          # keep the console clean for the human
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8")
            if self.path.startswith("/api/state"):
                session.do_poll()
                return self._json(session.snapshot())
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"ok": False, "message": "bad JSON"}, 400)

            if self.path == "/api/pick":
                if "index" in body:
                    r = session.record_index(int(body["index"]))
                else:
                    r = session.record_query(str(body.get("query", "")))
                return self._json(r)
            if self.path == "/api/undo":
                return self._json(session.undo())
            if self.path == "/api/poll":
                session.poll_enabled = bool(body.get("enabled", True))
                session.poll_status = "starting…" if session.poll_enabled else "off"
                if session.poll_enabled:
                    session.do_poll(force=True)
                return self._json({"ok": True, "enabled": session.poll_enabled,
                                   "status": session.poll_status})
            self._json({"ok": False, "message": "not found"}, 404)

    return Handler


def serve(session: DraftSession, port: int = 8777) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(session))
    return httpd
