# CLAUDE.md — First Down Syndrome

**Read this every session before doing anything.** It exists so you never again have to be
told that this is a 9-team, 2-QB, half-PPR league with 6-point passing TDs, −1 per sack taken,
0.05 per carry, and free bye weeks in 5 and 14.

Full specs: `FANTASY_SPEC.md` (§4–§12 not reproduced here) and `SPEC_ADDENDUM_METRICS.md`.
Section numbers below refer to `FANTASY_SPEC.md` unless prefixed with `ADD-`.

---

## 0. NON-NEGOTIABLE RULES

### 0.1 Never auto-submit
**Never submit a lineup, waiver claim, trade, or draft pick to ESPN. Ever.**
Recommend; the human confirms and clicks. This holds even if the recommendation is obvious,
even if time is short, even if a previous confirmation seems to cover it. Write-capable ESPN
endpoints are out of scope for this project — if code to hit one appears, it is a bug.

The draft is manual entry by design (§6): ESPN's live draft feed is flaky, so the human types
the pick. Polling the feed is a convenience, never the source of truth.

### 0.2 The ID crosswalk is a hard gate
nflverse uses `gsis_id`; ESPN uses its own IDs. **Names do not join reliably** — suffixes,
initials, duplicate names, rookies missing from community crosswalks.

> **Every rostered and drafted player must resolve to exactly one canonical ID.
> Assert it. Fail loudly on any miss. An unmatched ID is a blocking error, not a warning.**

Never paper over a miss with fuzzy name matching, a `try/except`, a silent `dropna()`, or a
default. Unresolved players go into an explicit, human-reviewed override file and nowhere else.
More projects die here than anywhere else, and they die silently — the symptom is a board that
recommends a retired player, discovered on draft day.

Corollary: D/ST are team entities, not players, and need their own ESPN-ID → nflverse-team-abbr
path. Kickers are players but sit outside most skill-position crosswalk tables.

### 0.3 Other standing guardrails (§10, condensed)
- **Log every recommendation** with timestamp and inputs, or the season teaches you nothing.
- **Fail loudly on stale data.** Week-3 data silently running in week 8 is worse than no tool.
- Re-verify ESPN cookies each session; they expire. Clear message, not a stack trace.
- Cache aggressively. nflverse weekly, odds hourly at most. **Must run fully offline** on draft day.
- Sanity alarms — each means something is broken: a K before round 13 · only 1 QB rostered after
  round 11 · any lineup being set for week 5 or week 14.

---

## 0.4 SUBAGENT REGISTRY — delegate, don't flood main context

Defined in `.claude/agents/`. Use them rather than doing this work inline:

| Agent | Delegate to it when |
|---|---|
| `metrics-researcher` | Evaluating whether a candidate input deserves a place in the projection model. **Before adding any new feature.** Defaults to recommending against. |
| `news-scout` | Injury / depth-chart / role-change news. Weekly waiver prep, Sunday ~11:15am ET inactive checks. |
| `opponent-scout` | Profiling a league manager's draft tendencies or roster needs, from ESPN league history. Before the draft and before any trade offer. |
| `data-validator` | **After every pipeline change and before the draft.** Hunts silent corruption: ID joins, stale caches, scoring mismatches, bye-week flags. |

---

## 0.5 PROJECT DECISIONS (settled 2026-08-19)

| Decision | Choice |
|---|---|
| Target season | **2026**; history window **2016–2025** (10 seasons); "last season" = 2025 |
| Python | `uv`, pinned **3.12** (system Python is 3.9.6 / EOL — do not use it) |
| nflverse client | `nflreadpy` (polars) |
| **Crosswalk source** | nflverse **`players`** table — carries `gsis_id` *and* `espn_id` in one row. **Not** DynastyProcess `ff_playerids`, which §6 assumed. Verified 2026-08-19: 98.8% `espn_id` coverage on active skill players, **zero** duplicate `espn_id`s |
| Rookie detection | `rookie_season` or `years_of_experience`, **never `draft_year`** — `draft_year` is blank for the entire 2026 class and only marks actually-drafted players in prior years |
| Cache | Parquet per table per season, queried via DuckDB; each file carries a fetch timestamp |
| Crosswalk test set | ESPN full draftable pool **and** 2025 final ESPN rosters — zero unmatched in both |
| Odds / weather | Deferred out of Milestone 1; they are §9 weekly signals, not crosswalk inputs |

### Notes on the history window
- **2016 is the Next Gen Stats boundary** (verified: `ngs_*` covers exactly 2016–2025), so it is a
  principled start, not an arbitrary one. It buys 9 year-over-year transitions instead of 4 —
  which is what ADD-§I-3c ("weight inputs by measured stability on *your* data") actually needs,
  and what ADD-§D age curves need to be more than noise.
- **Era drift is real.** Do not pool 2016 and 2025 blindly when fitting ADD-§A xFP bucket rates —
  the passing environment moved. Use the full window for *stability measurement* and age curves;
  weight recent seasons (or carry a season term) for xFP expected-outcome rates.
- **2020 is contaminated** (no preseason, empty stadiums, COVID absences). Keep it, but flag it.
- `ff_opportunity` is **not** in the `nflverse-data` repo; it lives in `nflverse/ffopportunity`.
  Source separately or drop it — it is a cross-check for M3b, not a dependency.

### MILESTONE STATUS

**M1 (data layer + ID crosswalk) — COMPLETE, gate closed 2026-08-20.**
Zero unmatched across all five populations: 2026 draftable pool (995 players +
32 D/ST), 2025 rosters (127+10), and the 2023/2024/2025 drafts.

**M2 (scoring engine) — COMPLETE 2026-08-20.** 140 tests pass.
- **Layer A (the hard gate): 100% on 2023, 2024 AND 2025** — our rules applied to
  ESPN's own stat line reproduce ESPN's total exactly, across *two* rulesets.
  A single failure here means a scoring rule is wrong.
- **Layer B (end to end, nflverse -> points): 6048/6052 = 99.93%.**
  2023 is perfect (1821/1821); 2024 and 2025 each have 2 rows where nflverse and
  ESPN disagree about a *stat value*, not a rule. All four are pinned by name in
  `tests/test_scoring_gate.py` so a NEW disagreement fails the suite.
- **Layer C** attributes any mismatch to a specific rule rather than a bare total.

**M3 (projection model) — COMPLETE 2026-08-20.** 162 tests pass.
Gate: blended projections beat raw consensus on rank correlation, **walk-forward
(weight fitted on prior seasons only): 4/4 seasons, +0.0034 mean Spearman**.
The gain is real but SMALL, and it is a plateau not a peak — every weight in
0.05-0.20 wins all five backtested seasons.

- Consensus 0.761 · model alone 0.687 · blend 0.767 (mean Spearman, 2021-2025).
- **The model alone is materially WORSE than consensus.** §7.2 step 4's "blend,
  don't replace" is not a stylistic preference — consensus encodes offseason news
  no historical model can see. Default weight is **0.12** to our model.
- The pre-registered 0.35 merely tied. The model deserves less weight than
  intuition suggested, which is itself the finding.
- Per-position weight fitting was WORSE (5/12) than a single global weight (4/4):
  four weights fitted on three seasons overfits. Use one global weight.

**M3b (xFP/xTD) — COMPLETE 2026-08-20. The headline test FAILED, deliberately
recorded rather than tuned away.**

ADD-§I-3b asks whether xFP predicts next-season points better than prior-season
points does. **It does not, at any position** — QB −0.029, RB −0.035, WR −0.009,
TE −0.029. So **xFP does NOT enter the point projection.** It survives as a
diagnostic only. ADD-§H already lists five things not to build; this is the sixth,
and it was found by testing rather than assuming.

But three sub-findings are worth more than the headline would have been:

1. **xFP IS more stable** — the other half of ADD-§A's claim holds at every
   position (xFP→next xFP beats points→next points: QB .527/.479, WR .752/.700,
   TE .732/.704, RB .707/.698). ADD-§A conflates "more stable" with "more
   predictive"; on this data only the first is true. Stripping luck from last
   year also strips demonstrated skill, and some of that skill persists.
2. **xTD beats actual TD for WR and TE, but NOT for RB and QB.** The asymmetry is
   the finding: RB touchdowns come from goal-line ROLE, a durable team
   assignment, not luck. Use the TD-regression fade for receivers only.
3. **Sacks-over-expected is a TRAIT; TD-over-expected is luck.** SOE carries
   forward at **0.434**, TDOE at **0.104** — four times the persistence. In a
   −1/sack league that is the durable, unpriced edge §3.3 predicted, and it is
   the one piece of xFP that should reach the board.

**M4 (VOR + tiers + bye adjustment → `board.json`) — COMPLETE 2026-08-20.**
192 tests pass. `uv run python -m ff_agent.cli board` writes `artifacts/board.json`
and prints the §11 review: divergences with attributed reasons, bye lift, the
QB-count decision, and §10's sanity alarms.

**M5 (opponent model → `opponents.json`) — COMPLETE 2026-08-20.** 206 tests pass.
Gate passes: fit on 2023–24, predict 2025, against **superflex** ADP (standard
would be a straw man — §7.5 rules it out). mean log-prob −4.942 → −4.890,
top-1 accuracy **3.1% → 6.3%**, top-5 21.9% → 27.3%.

**M6 (season simulator → `season_sim.json`) — COMPLETE 2026-08-20.** 221 tests pass.
Mechanics verified; the §11 standings test does **not** pass cleanly, and the
diagnosis is recorded rather than tuned away.

**M7 (Monte Carlo draft simulator → `draft_sim.json`) — COMPLETE 2026-08-21.**
256 tests pass. Gate: replay the 2025 draft (8 teams, real order, my real slot 7,
opponent model fit on **2023–24 only**, pool of 290 against 136 actual picks).
**Graded arm CALIBRATED — 78% coverage against an 80% target, on intervals 35
picks wide (26% of the draft).** Runs at ~11s per slot at 10,000 sims.

**M8 (nine slot plans → `plan_1..9.json`) — COMPLETE 2026-08-21.** 274 tests pass.

**M9 (§0.2 structural guards) — COMPLETE 2026-08-22.** 290 tests pass.
Guard against fan-out at the tier_stability join and in tier_stability() itself.
Merged with the tier_stability reproducibility refactor (03d217a).

**M4 (LA/LAR bye and SOS) — FIXED 2026-08-22.** +7 regression tests for the
Rams-trap fix: team vocabulary canonicalisation at the nflverse boundary and
assertions on joined team fields.
`uv run python -m ff_agent.cli plans` precomputes; `draft --slot N` is a **file
read** (§5's T−60 budget is 5 seconds and it is asserted, not hoped for).

**M9 (live draft loop → `cli live --slot N`) — COMPLETE 2026-08-22.** 326 tests pass.
Gate (§11 step 9): replay the real 2025 draft through the loop, every pick typed
by NAME through the manual-entry path. **136/136 resolved, 17 turns advised,
p95 latency 0.371s against §4's one-second budget, final roster legal.**
Cold start **0.7s with the network off**.

**Live draft GUI (`cli gui --slot N`) — ADDED 2026-08-22.** 333 tests pass.
The same engine as `cli live` served over stdlib `http.server`: `DraftState` is
still the truth, `advise.advise` still makes the call. Ready in **2.1s**, advice
in ~0.8s. Clicking a recommendation or a board row records that pick; ambiguity
still prompts and never guesses.
- **The ESPN poll is a convenience, never truth (§6).** It only ever APPENDS
  picks it already agrees with; a feed contradicting a typed pick is reported as
  a conflict and left unapplied. All 555 pool entries resolve from a draft-feed
  id — defenses via `dst_crosswalk`'s `canonical_id`, a direct join.
- **§0.1 needed a SECOND guard.** The Python scan covers new modules via
  `rglob`, but cannot see `ui.html` — and the page is the only thing here making
  requests from a browser, where the ESPN cookies live. A test now asserts every
  `fetch()` is a same-origin `/api/` path and no absolute URL appears at all.
- Stdlib and system fonts only: §0.3 needs this to work offline, and a webfont
  would fall back mid-draft.

**Runnable against ANY ESPN league — ADDED 2026-08-22.** 348 tests pass.
`ff_agent/live/profile.py` reads the league's shape from ESPN instead of
trusting §1's constants, so the tool can be handed to somebody in a different
league (`SETUP_ANY_LEAGUE.md`).
- **The draft slot is detectable too, not just team shape.**
  `draftSettings.pickOrder` — a list of `team_id`s in round-1 order — sits in
  the same `mSettings` payload the position limits come from, whenever the
  commissioner has set it. `cli gui --auto` reads it and needs no `--slot` at
  all when it's there; asks for one, loudly, when it isn't. This is how
  `draft_date_time` got resolved (§0.5 STILL OPEN, above) — `draftSettings.date`
  was sitting right next to it. Both were available HOURS before §5's "~1hr
  before" framing, on this league at least — that framing describes one
  commissioner's habit, not a platform guarantee, so `--slot` stays as the
  override for whenever it isn't set that early.
- **Detection independently reproduces EVERY §1 constant** — team count, all
  seven starter slots, bench 7 / IR 1, the position maxima, byes {5,14},
  6 playoff teams, redraft. `tests/test_profile.py` asserts this, which is
  simultaneously a regression guard and the evidence the ENGINE was never
  league-specific; only `config.py` was. The auto-built board is **identical to
  the shipped one in the top 40, name and VOR**.
- **`espn_api` does not expose position limits at all.** They live in the raw
  `mSettings` endpoint under `rosterSettings.positionLimits`, keyed by ESPN's
  numeric position ids (`-1` = no limit). Without them M7's failure returns:
  eight QBs, no TE, ~50 points a week. A league whose caps cannot be read is
  warned about loudly rather than run uncapped.
- **Identity is the SWID cookie, not the team name.** Names are mutable — §1's
  "A Chane Reaction" is "TBD" now — so the logged-in account is what says which
  team is yours. `--team` is a fallback for co-owned or orphaned accounts.
- **VOR replacement had to become a parameter.** It is measured against the
  STARTER replacement, so a 1-QB or 12-team league gets a genuinely different
  board, not a rescaled one: QB18/RB23.1 here, QB12/RB30.8 for 1-QB 12-team.
  `replacement_ranks(starter_slots=...)`; the default is byte-identical.
- **IDP and exotic slots are REFUSED, not approximated** (§10). The projections
  cover QB/RB/WR/TE/K/D-ST, so a league starting linebackers would get a
  confident wrong board. Keeper leagues run but warn.
- A first-year league with no draft history degrades to `neutral_context` —
  everyone drafts off consensus, stated out loud. M7 measured that arm at
  +29.9 weekly points against the full model's +32.1, so the loss is small.

**M10a (post-draft analysis → `postdraft.json`) — ADDED 2026-08-22.** 375 pass, 2 fail.
All 24 new tests pass; both failures are unrelated to this milestone (see below).
`uv run python -m ff_agent.cli postdraft` grades every pick, rates every roster,
and projects the season. It reads this tool's own session log by default and
ESPN's draft export otherwise (`--source espn`).

- **A pick is graded over a PAIR of picks, not one, and post-hoc that
  counterfactual is EXACT.** M8's central finding — "Josh Allen has the highest
  VOR on the board and is the THIRD best choice, because he comes back 97% of
  the time" — needs the opponent model to guess survival while the draft is
  live. Afterwards the survivors are *observed*, so the question "which legal
  pair (a at my pick, b at my next pick) was worth the most" is answered with
  arithmetic instead of simulation. The one assumption, printed on the output,
  is that the other managers would have picked the same players regardless.
- **Scoring those pairs by VOR reproduced M7's four-QB bug from the other
  side.** The first version handed round one a wall of Ds — Ja'Marr Chase at
  pick 6 graded F against "Drake Maye + Joe Burrow" — because VOR is measured
  against the STARTER replacement and prices QB3/QB4 as though they would start.
  Pairs are now scored with `surrogate.lineup_stats`, the same optimal-lineup
  solver M7 used, against **each manager's own play weeks**. Measured
  consequence, pinned as a test: **QB3 realises ~3.8 weekly points against the
  ~8.6 his VOR implies — about 44% — and what he DOES realise is bye coverage.**
  That is M8's "§2.1's QB COUNT IS THREE" visible in one line.
- **Hindsight regret has no meaningful zero, so the letters are a curve and say
  so.** The better pair is chosen knowing who survived, and on the 2025 draft the
  MEDIAN pick fell **8-10 weekly points** short of it; absolute thresholds graded
  four picks in five at D or F, which says nothing about any of them. Letters
  rank picks **within their own round** — the ratio carries a structural trend
  (median 0.74 in round 1 climbing to 1.7 by round 15 as the value spread
  collapses), so a global curve would fail the late rounds for being late.
  `regret_weekly_points` carries the magnitude beside every letter.
- **Lifting §3.5's late-K/D-ST guard produced KICKER WORSHIP.** It was removed on
  the argument that the rule is my discipline and grading a rival against it is
  unfair (`allowed_positions` says exactly that). What came back was "Texans
  D/ST + Brandon Aubrey" as the better pair at **nine consecutive picks of
  mine**: the rank→points curve gives the top kicker a real edge over the
  twentieth and nothing stopped the counterfactual banking it fifteen rounds
  early. §3.5 exists because that edge does not survive a season. Restored.
- **The evidence that motivated lifting it was a TRUNCATED-CORPUS ARTEFACT.**
  Rounds 15-16 showed inflated regret (median ratio 1.9 and 3.2 against ~1.0
  elsewhere) — but measured on the 2025 replay log, which stops at **128 of 136
  picks** and in which **no team ever drafted a defense**. The last rounds were
  being graded against rosters that structurally could not be finished. On a full
  draft `allowed_positions`' feasibility branch handles it: three picks left and
  two owed to K and D/ST still leaves round 15 free.
- **A mid-draft report is read off PHANTOM HOLES unless the rest is projected.**
  Team strength is the optimal starting lineup, so a roster that has not reached
  the kicker yet has an EMPTY kicker slot — and every counterfactual then
  discovers that the best available action at every pick is to take a kicker.
  `postdraft/finish.py` projects the remaining picks greedily on **marginal
  lineup value, never VOR** (padding by VOR would hand every team a quarterback
  room and no tight end), in real snake order so two managers cannot be given the
  same player. No-op on a finished draft; roster SHAPE always reports real picks.
- **§2.5 BIT AGAIN, in the projected finishing order.** Ranking on raw
  `mean_wins` put a 13-game team above a 12-game team it trailed on every other
  measure, so the projected order openly contradicted the playoff odds printed
  beside it. The league seeds on win PERCENTAGE (confirmed 2026-08-20) and now so
  does this. Five teams play 12 games in 2026 and four play 13, so this is not
  hypothetical.
- **polars hands `group_by` the key as a TUPLE.** Keying the schedule join on
  `('Personality Hires',)` matched nothing, silently, and gave **all nine teams
  my own bye weeks** — the same failure class as the Rams trap, where a left join
  fails as a NULL rather than an error. Fixed and asserted: the nine teams' byes
  must differ, and must sum to exactly one per week.
- **§1's "R. Sharrett" vs ESPN's "Rayne Sharrett" silently listed 3 of the 4
  double-up opponents.** M5 already built `canonical_manager` for exactly this;
  not calling it was the bug. Asserted at 4, and unmatched names are now printed
  rather than dropped. (Measured for 2026: my four double-up opponents average
  **118.6** projected weekly points against a league mean of **122.9** — §2.3's
  2x weighting cuts in my favour this year.)
- **K and D/ST have a board VOR RANK but no board ADP, and mixing the two made
  every "biggest reach" a defense.** A top defense lands around overall 35-76 by
  VOR, while this league takes exactly one each per team, late (26/26 and 27/27).
  Left in the market-timing lists, the six biggest reaches of the draft were all
  D/ST taken in rounds 12-17 — with ECR sitting calmly beside them saying the
  picks were normal. That is two of our own scales disagreeing, not a decision
  anybody made. Steals and reaches are now skill-positions-only; regret still
  covers all six positions, because THAT scale is common to them.
- **Ties in regret must SHARE a grade.** Once the counterfactual got good enough
  that most picks in a round came back at exactly zero regret, ordinal ranking
  broke ties on row order and split two identical picks between A+ and C.
  Rank-averaging over tie groups.
- **§2.1 generalises to all nine teams and the code already supported it.** Each
  team has its own fantasy byes (14 bye slots across 9 teams), so bye cost is
  measured against each manager's own play weeks rather than mine. My free-bye
  count is a property of my schedule; theirs is a property of theirs.
- **The standings forecast ships M6's measured ceiling, in the report body.**
  Preseason projections scored Spearman **-0.21** on this exact task in 2025; a
  PERFECT simulator scores **+0.52**, perfect player knowledge **+0.43**. The
  probabilities are the honest output; the finishing order is decoration. A test
  asserts those three numbers are still printed.
- §0.1 is asserted here too — `test_postdraft_package_cannot_write_to_espn`
  scans the package for the same write verbs `ff_agent/live/` is scanned for.
- **A TEAM WAS RENAMED DURING THE DRAFT, for the third time in this project.**
  leah gottlieb's "leah's team" became **"Chase-ing Dubs"** minutes after she
  took Ja'Marr Chase, and `test_league_settings.py::test_my_schedule_matches_the_spec`
  now fails on it: `team_names_by_manager` has the new spelling while the cached
  `league_schedule` still has the old one, which is the independent-cache
  divergence already recorded for "A Chane Reaction" → "TBD". Not caused by M10a
  and not fixed by it — the post-draft path resolves manager → team_id →
  the schedule's own spelling and is unaffected — but the settings test is
  pinned to §1's literal names and wants the same treatment.
- **Two clock tests fail under CPU load and are not real failures.**
  `test_advice_fits_the_clock` (1.13s against a 1.0s budget) and
  `test_full_mock_replays_the_2025_draft_under_the_clock` both fail while the
  draft GUI is still running at ~39% CPU, and both pass standalone. Verified by
  removing every M10a change and re-running: the failure is byte-identical.

**M10b (in-season monitor) — DESIGNED 2026-08-23, not yet built.**
Full plan: `docs/M10B_MONITOR_DESIGN.md`. A Docker container on an always-on box runs
the §9.1 cadence, recomputes ROS value and Δ P(title) with the M2–M7 engines, and
**emails** an ordered list of actions. §0.1 holds absolutely: it recommends, I click.
Settled: home box + Docker Compose + supercronic · **email only** · deterministic core
with a narrow Claude layer for news and prose · `TZ=America/New_York`, asserted.

Four reconnaissance findings shape it, all measured 2026-08-23:

- **THERE IS NO FORMAT-MATCHED REST-OF-SEASON CONSENSUS.** `rsf` — the superflex list
  M3 shipped on — is scraped **preseason only**: one snapshot at 2025-09-05, then
  nothing until the next August. In-season the weekly-updating lists are
  `do dp drk dsf ro rp wo wp wsf`. Only `wsf` is superflex-and-redraft, and it is
  **this week only**. `ro`/`rp` are rest-of-season but **1-QB**: on 2025-10-31 `ro`'s
  top 24 held **zero** QBs against `wsf`'s **13**, and `ro`'s overall #1 was a
  linebacker (the page ships IDP-polluted). M5 already recorded the consequence of
  ignoring format — "format difference posing as personality" — and here it would
  manufacture fake waiver value at QB, exactly where §9.3 says to spend priority.
  **Resolution: anchor ROS on POINTS, not RANKS.** ESPN's per-week projected points are
  already in §1 scoring and points carry no format; `wsf` serves the weekly jobs, where
  it is exactly right; `rp` is a non-QB cross-check only.
- **The historical FA pool IS reconstructible, so the §11 step 10 gate is possible.**
  `League.load_roster_week(week)` hits `mRoster` with `scoringPeriodId`;
  `League.transactions(scoring_period, types={FREEAGENT, WAIVER, WAIVER_ERROR})` gives
  adds, drops and **failed claims**; `League.box_scores(week)` gives `slot_position`,
  i.e. what was actually STARTED. `WAIVER_ERROR` is the only observed counterfactual in
  the league and is what calibrates §9.3's P(claim succeeds).
- **`player_owned_espn` is null in-season** (0 of 6,038 `wsf` rows from Oct 2025), so
  the obvious "trending adds" control is rebuilt from the transaction log instead —
  "most-added player across the league that week", which is what the other eight
  managers actually did rather than a proxy for it.
- **ESPN carries per-week `projected_points` AND `points_breakdown`** (per-rule applied
  points — M2's Layer A object). So the M2 gate becomes a **weekly tripwire**: this
  league already changed its scoring once, after 2024.

- **THE LINEUP DOES NOT LOCK ALL AT ONCE, and the lock calendar is not weekly-periodic.**
  ESPN locks each player at HIS OWN kickoff, so "set the lineup" is a sequence of
  irreversible per-slot commitments under increasing information. Measured on the real
  2026 schedule: **week 1 opens on a WEDNESDAY** (9/9 20:20, with Thursday at 20:35 not
  20:15); **week 16 — my semifinal — has three Christmas Day games** (12/25 at 13:00,
  16:30, 20:15); **week 15 has two Saturday games** (17:00, 20:20); Sunday's late slate
  is TWO windows (16:05 and 16:25); and 2025 week 4 had a **Sunday 09:30 London kick**.
  Six to nine distinct lock times a week, worst in the fantasy playoffs. §9.1's four
  fixed slots would miss the Wednesday opener, all three Christmas games, and the whole
  Sunday late slate. **Resolution: derive checkpoints from the schedule, never the
  clock** — the crontab becomes a 15-minute tick and `clock.py` fires at kickoff −24h,
  −3h and **−75min** (inactives drop at −90). Consequence: `season/lineup.py::optimal_lineup`
  needs `pinned={canonical_id: slot}`, since once Thursday locks the rest must be solved
  AROUND a fixed partial assignment. And the Thursday call is not "who is better" — it is
  `E[p] > E[best Sunday alternative chosen under SUNDAY information]`, a strictly higher
  bar, decided by M9's force-and-simulate pattern. Two effects partly cancel and BOTH get
  measured: locking forfeits the option on that slot, but a Thursday player who *starts*
  reveals my partial score three days early and sharpens Sunday's floor/ceiling posture —
  benching him preserves nothing, since he is locked out either way.
- **AVAILABILITY BY DESIGNATION, and the QB asymmetry is large.** nflverse `injuries`
  keys on **`gsis_id`** — joins straight to canonical, no crosswalk hop. 2025 REG, skill
  positions, P(did not play | Friday designation): Questionable **0.418**, Doubtful and
  Out 1.000, and *no designation* **0.149**, which is the size of the measurement bias
  (the proxy is absence from the weekly stats table, which conflates inactive with
  "dressed and recorded nothing"). Two sub-findings: **a Questionable QB is 0.735 against
  a Questionable RB's 0.360** — roughly double, and in a 2-QB league that is where it
  matters; and **full practice participation does NOT separate from limited** (0.421 vs
  0.404), which is the opposite of the folk rule a hand-written heuristic would encode.
  Redo against snap counts over 2016-2025 in M10b-4; recorded now because the positional
  asymmetry survives any plausible bias correction and the Thursday decision turns on it.

Two shipped modules need changes: `season/simulate.py` must accept completed results
(it re-simulates all 14 weeks, which is wrong from week 2 on), and
`season/strength.py::roster_strength` hardcodes `/17` where in-season needs remaining
games; and `season/lineup.py` needs slot pinning (above). And a gift: M7 needed the
P(title) surrogate to score 10,000 rosters; a weekly job scores ~30, so **M10b can
afford the real simulator**.

**Next: M10b-1 (container, scheduler, notifier, pre-flight) — shippable alone.**

**M9 design constraint, stated by the human 2026-08-21:** the live draft agent
must hand over **a concrete shortlist of players to choose from**, not a score or
a strategy note. M8's `per_turn[].shortlist.candidates` is built to be exactly
that — names, positions, and P(I take him here) — so M9 reads it rather than
recomputing.

**A `data-validator` pass on M7 (§0.4) found four real defects. All fixed:**
- **§0.2 WAS BEING VIOLATED, silently.** `model.project` emitted **one row per
  (player, prior NFL team)** for anyone who changed teams mid-season. Those fan
  out through `blend()`'s left join (1→2) and again through the board's tier join
  (2→4): **15 players reached the 2026 board four times each**, 60 of 600 pool
  rows. The engine marks pool INDICES taken, not people, so two fantasy teams
  could draft the same person in one simulated draft. Worst-affected was ranked
  127 (Jakobi Meyers), so nothing in the draftable top 100 — but the mechanism is
  generic to any player who changes teams. Deduplicated at the root, and §0.2 is
  now **asserted at both boundaries** (`board.assert_one_row_per_player`,
  `pool._to_pool`). Resolution was never the whole of §0.2; a join that fans out
  breaks it just as thoroughly.
- **`weekly_scale` divided by 16, inflating every player by 6.25%.** The reasoning
  ("a bye costs a game") was wrong: **the NFL plays 18 WEEKS and 17 GAMES** — the
  bye is the eighteenth week, not one of the seventeen. Verified on 2024 actuals,
  where 215 players logged `games == 17`. Double-charging was already prevented at
  the week level in `_week_points` and never depended on the divisor. It also
  disagreed silently with `board/build.py`, `season/strength.py` and
  `projections/consensus.py`, which all use 17. **A test had pinned the wrong
  answer**, which is why it survived. Roster means moved ~145 → ~137.8.
- **`draft_history` could never be cached as immutable.** It spans 2023–25 in one
  table so it is cached with `season=None`, and `effective_ttl`'s "a completed
  season never expires" rule can only fire when a season is given. With no
  `TTL_POLICY` entry it fell back to the 1-day default and went stale daily —
  harmless online, **fatal to the §0.3 offline draft-day path**, since the board,
  the calibration and the opponent model all read it. Now 90 days.
- **Travis Hunter carries nflverse's `CB`, not ESPN's `WR`.** The ID match is
  right (M1 confirmed it) but the POSITION is nflverse's, and he is silently
  dropped from every positional aggregate while still counting toward the phase
  total in `target_rates` — so the fitted offsets are diluted. 1 pick in 424.
  Now **warned loudly** rather than corrected: guessing a fantasy position from an
  nflverse one is the fuzzy matching §0.2 forbids. The fix, when it matters, is an
  explicit override row.

**Knock-on:** the dedupe changed M3's plateau. The blend got slightly BETTER
(mean Spearman 0.767 → **0.770**) and the range of weights winning all five
seasons narrowed from 0.05–0.20 to **0.05–0.15** (0.20 now wins 4/5). The shipped
weight of **0.12** sits inside either version. Recorded in `tests/test_projection_gate.py::PLATEAU`.

**M9 design constraint, stated by the human 2026-08-21:** the live draft agent
must hand over **a concrete shortlist of players to choose from**, not a score or
a strategy note. Build M8's plans so that shortlist falls out of them directly.

```bash
uv run python -m ff_agent.cli status      # cache inventory + staleness
uv run python -m ff_agent.cli byes        # §2.1 free-bye teams
uv run python -m ff_agent.cli crosswalk   # THE GATE — needs .env
uv run python -m ff_agent.cli score       # M2 GATE — recomputed scores vs ESPN
uv run python -m ff_agent.cli project     # M3 — build projections for the season
uv run python -m ff_agent.cli project --backtest   # M3 GATE — walk-forward
uv run python -m ff_agent.cli xfp --validate      # M3b — xFP + validity tests
uv run python -m ff_agent.cli board       # M4 — VOR/tiers/byes -> board.json
uv run python -m ff_agent.cli opponents --validate  # M5 — opponents.json + gate
uv run python -m ff_agent.cli simulate --validate   # M6 — season_sim.json
uv run python -m ff_agent.cli draftsim            # M7 — draft_sim.json, 9 slots
uv run python -m ff_agent.cli draftsim --validate  # M7 GATE — replay 2025
uv run python -m ff_agent.cli plans       # M8 — plan_1..9.json (PRECOMPUTE)
uv run python -m ff_agent.cli draft --slot 6   # T-60 drill: file read, no compute
uv run python -m ff_agent.cli live --slot 6    # M9 — the live draft loop (terminal)
uv run python -m ff_agent.cli gui --slot 6     # M9 — the SAME loop in a browser
uv run python -m ff_agent.cli setup            # ANY league: preflight + build its board
uv run python -m ff_agent.cli gui --auto           # ANY league: detects slot AND shape
uv run python -m ff_agent.cli mock             # M9 GATE — replay 2025 live
uv run python -m ff_agent.cli postdraft        # M10a — grades, ratings, predictions
uv run python -m ff_agent.cli postdraft --source espn --csv picks.csv
uv run python -m ff_agent.cli settings    # refresh league settings JSON
uv run python -m ff_agent.cli verify      # cookie pre-flight, run draft morning
uv run python -m ff_agent.cli offline     # prove the draft-day path
uv run pytest                             # 377: 375 pass, 2 unrelated
```

Layout: `ff_agent/config.py` (§1 constants, credentials) · `data/cache.py`
(parquet + staleness) · `data/nflverse.py` (ingest + column contract) ·
`data/byes.py` (§2.1) · `data/espn.py` (league state, read-only) ·
`data/crosswalk.py` (§0.2 gate) · `overrides/player_id_overrides.csv` (tracked —
human decisions belong in git).

**Scoring rules are LOADED from ESPN, not hardcoded** (`ff_agent/scoring/rules.py`).
§1 is demoted to an assertion against what was loaded, so a transcription slip
cannot corrupt the engine and a mid-season settings change fails loudly.

**M2 findings — every one of these silently breaks a generic implementation:**
- **The league changed its scoring after 2024.** 2023 and 2024 also scored `PC`
  (0.25/completion) and `INC` (-0.1/incompletion); both removed for 2025. So
  **only 2025 is a valid exact-match target**, and ESPN's *recorded* historical
  points are not comparable across that boundary. Recomputing from stat lines
  under current rules — what this engine does, per §7.2 step 2 — is the fix.
- **D/ST yards allowed = opponent `passing_yards + sack_yards_lost + rushing_yards`.**
  `sack_yards_lost` is stored NEGATIVE, so it ADDS. Gross passing+rushing is
  wrong. Exact on 119/119 team-weeks.
- **D/ST points allowed excludes only what MY OFFENSE conceded** on a scrimmage
  play — pick-sixes, strip-sack returns, offensive safeties. Points scored
  against my SPECIAL TEAMS still count as allowed (kickoff-recovery TDs,
  blocked-FG returns, a punter tackled in his own end zone), because the fantasy
  D/ST is defense *and* special teams. Using the opponent's final score is wrong;
  so is excluding every non-offensive TD. Each is off on 8 of 119, on different rows.
- A punt/FG snap that becomes a run or pass is typed `run`/`pass` by nflverse with
  every special-teams flag at 0 — the `(Punt formation)` tag in `desc` is the only
  signal. 17 such plays conceded points across 2016-2025.
- **Blocked field goals count as misses** under §1's flat -1. nflverse tracks
  blocks separately: `fg_missed` alone is 109/113, `fg_missed + fg_blocked` 113/113.
- **D/ST touchdowns span three nflverse columns** — `def_tds` +
  `fumble_recovery_tds` + `special_teams_tds`. `def_tds` alone misses 11 of 135.
- **D/ST fumble recoveries are `fumble_recovery_opp`**, not `def_fumbles` (which
  matches neither direction). Sacks come from **counting pbp sack plays**, not
  `team_stats.def_sacks`, which aggregates fractional player credits and drops one.
- nflverse spells the Rams **`LA`**; ESPN and the crosswalk use `LAR`.
- **Never store a dict column in parquet.** polars infers struct fields from a
  sample, so categories appearing only in later rows vanish — this silently
  understated every D/ST touchdown until the totals disagreed. Fixed columns or
  a JSON string.

**M3 findings:**
- **§3.1's VOR table understates QB.** Measured on 10 seasons recomputed under §1
  scoring: QB18 is **232.5**, not the ~265 §3.1 guessed, so QB VOR is **181.9**
  against RB's **155.9**. §3.1 concluded QB and RB were "near-tied"; they are not
  — QB is clearly ahead. (Elite values from realised data are upward-biased, so
  trust the replacement levels more than the elite figures; M4 should recompute
  VOR from PROJECTED points.)
- **Use superflex ECR (`ecr_type="rsf"`), never standard.** 11 QBs in the top 24
  vs 0 for standard. nflverse carries weekly ECR snapshots back to Dec 2019,
  which is what makes a no-lookahead preseason backtest possible at all.
- **The FantasyPros scrape has artifacts that land at the very top.** A literal
  "Player Name" placeholder ranked 12th overall, and single-expert entries
  (`sd == 0`) ranked 8th and 42nd. Filtered explicitly, never silently.
- **ESPN's SEASON projections report yardage PER GAME** while every other field
  is a season total (Gibbs: `rushingYards 80.74` beside 283 carries at 4.85/carry
  = 1372.6 = 80.74 x 17). Read literally, every projection is ~17x wrong.
- ESPN's kicker projection merges 50-59 and 60+ into one `50Plus` bucket, but §1
  pays 5 and 6 respectively — it collapses exactly the distinction ADD-§E says is
  worth money here.
- **ADD-§B confirmed on our data**: volume averages **0.584** year-over-year,
  efficiency **0.292**. WR/TE target share ~0.77, RB carries/game 0.776, QB sack
  rate 0.394 (ADD-§B.1 predicted ~0.50) and it correlates **-0.134** with next-season
  PPG — sack-prone QBs are worse, and this league charges them twice.
- **Per-game rates alone are a trap.** They ranked Jimmy Garoppolo and Joe Milton
  III as top-10 QBs. Prior-season `games` and `points` are needed as role features.
- Historical actuals are recomputed under **current** rules, deliberately opposite
  to M2 validation which uses each season's own rules.

**M9 findings:**
- **The live loop RE-SIMULATES rather than filtering the plan, and that is
  affordable.** Measured: simulating the REMAINDER of the draft costs 0.61s at
  500 sims from pick 40 and 0.27s from pick 100. So `engine.simulate_drafts`
  gained a `history=` argument (every sim shares the real prefix and diverges
  after it), and advice conditions on what actually happened instead of on a
  plan reality has already left behind.
- **"Candidates with probabilities" COLLAPSES at the current turn.** A
  deterministic policy has exactly one answer on a known board, so the top
  candidate is 100% and the sheet says nothing. What decides a pick is the
  counterfactual: force each option, simulate the rest, compare the roster. The
  budget is split between the options, so it costs the same total pick-sims.
  Live example at pick 5 — Josh Allen has the highest VOR on the board (167.5)
  and is the **third** best choice, because he comes back 97% of the time.
- **The QB scenario resolves ITSELF once the draft starts.** M8 left 7-10 of 17
  turns scenario-dependent because one 2-QB draft cannot pin QB timing. But the
  world is observable: comparing quarterbacks actually taken against each
  scenario's fitted rate identifies it inside about two rounds, and the sim then
  runs under that scenario rather than the blend. M8's largest uncertainty is
  retired at the table.
- **A query LONGER than the stored name failed every match path.** ESPN shows
  "James Cook III"; the board says "James Cook". Prefix, surname and substring
  matching all miss, so typing the name exactly as ESPN displays it matched
  **nothing**. Found by the replay gate, fixed by stripping an enumerated set of
  generational suffixes from both sides. Same class: defenses are "Texans D/ST"
  in ESPN's pool and "HOU D/ST" in the prior-season fallback, so a static
  32-entry nickname table now resolves either. Neither is the fuzzy matching
  §0.2 forbids — the transformations are fixed and enumerated, ambiguity still
  prompts, and **no id resolves without a human looking at it**.
- **`normalize_team()` ECHOES anything it does not recognise**, so it is always
  truthy and `normalize_team(q) or nickname(q)` meant the nickname branch never
  ran. Check membership in `CURRENT_TEAMS` before trusting it.
- **D/ST had a null `canonical_id` in the entire draft history, every season.**
  They are stored as NEGATIVE espn ids, which appear nowhere in
  `canonical_players`. Harmless for positional aggregates (those key on
  `position`) and fatal for anything identifying the pick — the replay could not
  place 8 of 136. `resolve_dst()` handles exactly that id form; now all 424
  historical picks resolve.
- **TEAM NAMES CHANGE, AND ONE ALREADY DID.** §1 records "A Chane Reaction" for
  Jeff Boyd; that team is now called **"TBD"**. Every hardcoded name was a join
  waiting to break — and worse, the two ESPN tables cache independently, so for
  a while `team_names_by_manager` said "TBD" while `league_schedule` still said
  "A Chane Reaction". Fixed by resolving **manager → team_id → the schedule's
  own spelling**: managers are the stable identity (which is how §2.3 already
  thinks about the league), and team_id is the only field the two tables agree
  on. The §1 names are now a documented fallback, not the source of truth.
- **A bare `except Exception: pass` hid a `NameError`** in the first version of
  that lookup, so it silently returned stale §1 names while appearing to work.
  Narrowed to the offline/credential errors it was meant to absorb.
- **§0.1 is asserted, not trusted.** `test_live_package_cannot_write_to_espn`
  scans every module under `ff_agent/live/` for write verbs and roster-mutation
  calls, and checks `data/espn.py` exposes none either. This is the milestone
  where that rule would have been broken.

**M8 findings:**
- **§2.1's QB COUNT IS THREE, and M4's TWO was wrong for a locatable reason.**
  M4 compared QB3's *expected* bye coverage (17.9 pts) against the best stash's
  *ceiling minus mean* (26.4) — an expected value against a variance measure,
  which counts upside as free. Scored on P(title), which §8 says is the actual
  objective: **cap 3 → 0.2253, cap 4 → 0.2168, cap 2 → 0.2001**, margin
  **+0.0085** over the runner-up. The ordering is identical at the inflated and
  the noise-corrected delta, so it does not rest on the correction. And it holds
  in the regime that *favours* M4's argument: my honest delta is **+6.9** against
  a variance crossover at **+16.6**, so variance is still paying here — three
  quarterbacks win anyway.
- **§11 step 8 PASSES, measured the only way that means anything.** Between-slot
  divergence **0.0571** against re-seed noise **0.0063** = **9.0x**. Nine Monte
  Carlo plans always differ a little, so the comparison has to be against
  re-running one slot on a different seed, not against zero. The first version of
  this gate passed with a ratio of **infinity** because `run_slot` derives its
  default seed from the slot number and the "re-seed" run reproduced the draft
  exactly. Now pinned by `test_reseed_actually_changes_the_draft`.
  Wheel-vs-wheel is the biggest gap (slot1-vs-slot9 **0.124**), slot1-vs-slot5
  **0.047** — §5's structural prediction shows up in the numbers.
- **ONLY ABOUT HALF THE TURNS ARE ROBUST TO QB TIMING — 7 to 10 of 17.** Every
  plan is built under three positional-offset fits (`qb_cold` from the two 1-QB
  seasons, `shipped`, `qb_hot` from 2025 alone; early-QB multipliers **0.089 /
  0.173 / 0.241**) and the sheet marks any turn where the call changes. This is
  the honest headline of M8: the single 2-QB draft in the league's history is not
  enough to pin quarterback timing, and roughly half the draft is a bet on which
  world you are in. The shipped fit reproduces 2025's first-QB TIMING (pick 6 vs
  7) but understates QB **volume** — 3 in rounds 1-3 against ~5.6 scaled from
  2025 — because `FORMAT_MATCH_WEIGHT` puts only 60% of the evidence on the
  format-matched season.
- **Conditional survival, not availability, is what decides a pick** — and it has
  to exclude my own pick or it says nothing. "Lamar Jackson is 93% available at
  pick 14" is a fact about the draft; "if I pass he comes back 96% of the time"
  is a fact about my decision. Counting the sims where I took him makes every
  player I want read as vanishing. A NaN (the policy NEVER passes) is the
  strongest take-now signal and compares False against any threshold — flagged
  explicitly rather than left to silently read as "not urgent".
- **A VOR-ranked shortlist is useless in this league.** The board is QB-heavy and
  the simulated league drafts quarterbacks late, so at a mid-draft turn the five
  highest-VOR available players are all QBs the policy takes 0% of the time. The
  sheet carries two lists instead: `candidates` (who I actually take, with
  probability) and `slipping_away` (available, above replacement, does not come
  back).

**M7 findings:**
- **M5's opponent model CANNOT move the league's positional mix, and running it
  forward is the only way that shows.** Manager tilts are
  `log(manager rate / league rate)`, so across eight managers they cancel; the
  aggregate mix is therefore whatever superflex ECR implies. Simulating 2025 off
  the shipped model took the **first QB at pick 1** (median) and **10 QBs inside
  three rounds**. The league actually took the first at **7**, and **five**.
  Fixed with a (phase × position) log-offset fitted by iterative proportional
  fitting (`draft/calibrate.py`). Fitting offsets makes gate measure (a) true by
  construction, which is stated loudly rather than quietly banked.
- **The offsets add structure, not vagueness — proved with a σ sweep.** At the
  shipped `ADP_SIGMA = 0.055`: **with** offsets 77% coverage on 35-pick
  intervals; **without**, reaching 81% needs a 104-pick interval (77% of the
  whole draft). No kernel width reaches the offset arm's precision. `ADP_SIGMA`
  was reasoned, never fitted, and a sweep on M5's own log-prob gate prefers a
  *wider* kernel (−3.66 at σ=0.2 vs −4.89 at 0.055) while `top1` and median rank
  barely move — log-prob is optimised by hedging, which is the wrong objective
  for a simulator. **Left at M5's shipped value.**
- **The QB error FLIPS SIGN between the two arms, which measures the format
  change from a new direction.** No offsets → 10 QBs in 3 rounds (too many,
  following superflex ECR). Offsets fit on two ONE-QB seasons → 2 (too few).
  Actual 5, between them. An in-sample **lookahead ceiling** arm gets 4 [2,6] and
  the first QB at 5 [1,13] — so the mechanics are sound and the miss is the
  format change, exactly the structure of M6's ceiling test.
- **Most of the simulator's apparent edge is not real, and there is now a number
  for it.** `best_consensus` — a control policy with **zero** board edge, ranking
  by the same superflex ECR the opponent model is built on — beats the model by
  **+29.9 weekly points**, against `best_vor`'s **+32.1**. So the board
  contributes **+2.2** and the rest is bought from M5's opponents drafting with
  an 8-pick spread. Consistent with M3's +0.0034 Spearman. `model_seat` (me
  drafting exactly like the model) lands at **−1.5**, confirming no seat bias.
  **Compare policies and slots to each other; do not read the absolute level.**
- **UNCAPPED VOR TAKES FOUR QUARTERBACKS IN THE FIRST FIVE ROUNDS, EVERY TIME.**
  Not a policy bug — the board really does rank Allen/Jackson/Maye/Burrow at VOR
  167/148/147/122. The flaw is using VOR for a bench pick at all: **VOR is
  measured against the STARTER replacement (QB18), so it prices QB3 and QB4 as
  though they would start**, in a league that starts two. This is M4's flagged
  risk ("if it is wrong, it is wrong in the first two rounds") arriving as a
  concrete draft-day behaviour. Cap sweep at slot 5, weekly points vs league:

  | QB cap | `best_vor` | `need_weighted` |
  |---|---|---|
  | 2 | +31.6 | +36.5 |
  | **3** | **+33.5** | **+39.2** |
  | 4 | +32.1 | +38.2 |

- **§2.4's "roster variance is good" is TRUE ONLY BELOW A CROSSOVER, and the
  top-2 crossover comes FIRST.** Measured directly (40k sims per point, no
  surrogate), season-mean uncertainty sd 1 → 12:

  | delta | P(title) | P(top-2) |
  |---|---|---|
  | −15 | 0.010 → **0.036** (3.6×) | 0.023 → 0.073 |
  | 0 | 0.103 → **0.146** (+41%) | 0.208 → 0.272 |
  | +15 | 0.345 → 0.352 (flat) | 0.627 → **0.591** |
  | +30 | 0.597 → **0.577** | 0.914 → **0.853** |

  Crossovers: title at **delta +15.9**, top-2 at **delta +12.1**. Above it,
  variance spends the first-round bye — the thing §2.4 itself calls worth about
  as much as everything else combined. **Chase ceiling when average or behind;
  protect the mean once clearly ahead.** §2.4 states this without qualification
  and is right only for the first case.
- **M4's board is skill-only; K and D/ST had to be built in M7.** 537 QB/RB/WR/TE
  and zero K or D/ST, because superflex ECR ranks skill positions only. Without
  them **18 of 153 picks — 12% of the draft — do not exist** and every
  availability curve is off by a round. Added from ESPN's pool by
  `percent_owned`, valued off the M3 rank→points curve (which already covers both
  under §1), timed from this league's own history: K mean pick fraction **0.855**,
  DST **0.779**, one each per team in **26/26** and **27/27** cases. Weaker than
  the skill projections and proportionate — §3.5 says stream both. Backtests use
  a `prior_season` source instead, since ESPN's ownership percentages are today's.
- **The Rams trap bit again.** nflverse `LA` vs ESPN `LAR` would have handed the
  Rams K and D/ST a 17-week season with no bye. Now normalised and asserted.
- **The opponent model had NO roster constraints at all** — it was only ever
  asked to score picks that had already happened. Run forward it drafted eight
  QBs and no tight end, then fielded lineups with empty slots; the gap against a
  constrained policy was **~50 points a week**, which is how the omission
  announced itself. `policy.allowed_positions` now applies ESPN's maxima and
  starter feasibility to **every** team.
- **A season-mean-uncertainty term was added to M6's simulator** (`team_mean_sds`,
  opt-in, default off, existing behaviour unchanged). Weekly noise is redrawn
  every week; projection error is not — being wrong about a player is being wrong
  in all fourteen weeks, and correlated error moves standings far more than
  independent noise of the same size.
- **Bilinear interpolation clamps SILENTLY at the grid edge.** The surrogate's
  first run fixed the grid at −30..+30 while draft outcomes reached **+49**,
  scoring **0.21** absolute error on P(title) that looked exactly like an
  interpolation result — every sample *inside* the grid was accurate to 0.003.
  The grid is now sized from the simulated range and the clamped fraction is
  reported. After the fix: error **0.021** against a mean policy gap of 0.075.
  The validator had already refused to rank policies at 0.21, which is the
  machinery working.
- **§10's alarms are about MY roster, not the league's.** Running "only 1 QB
  after round 11" across all eight simulated opponents turns an alarm into a fact
  about rivals. Split into `sanity_alarms(team)` and `league_diagnostics()`.
- Seat assignment is unknown (I learn my slot at T−60 and never learn theirs).
  Reshuffling opponents moves my mean by **0.4 weekly points**, so filling seats
  in config order is an earned shortcut.
- `need_weighted` beats `best_vor` at **every** slot, by 0.053–0.090 P(title),
  and the margin shrinks monotonically from slot 1 to slot 9.

**M6 findings:**
- **§2.5 CLOSED — the league seeds on WIN PERCENTAGE**, confirmed from the live
  standings page (ranks on `PCT`, with `GB`). So **no structural handicap** despite
  playing 12 games while four teams play 13. This was worth confirming rather than
  assuming: with every team identical, raw-wins seeding would have cost a 12-game
  team **−7.9 pts of P(playoffs)** and **−8.1 pts of P(top-2 seed)**. The simulator
  still ships both rules, because that sensitivity is the reason the question
  mattered. Practical consequence: **marginal regular-season wins are worth the
  same to me as to anyone**, so §2.4's "chase seeding and playoff ceiling" stands
  without an offsetting penalty.
- **`first_round_byes = 2` is now OBSERVED, not inferred** — the top two seeds
  (Unsolicited Dak Pics, Personality Hires) both sat out week 15 of the real 2025
  bracket.
- **Weekly noise is ~3x the talent spread**: within-team weekly sd **25.7**,
  between-team season-mean sd **8.7** (measured on 2025's 112 team-weeks). Over 12
  games, H2H results are luck-dominated. This is *why* §2.4's "variance is good,
  marginal wins are cheap" holds — it is now measured, not asserted.
- **The §11 standings test does not pass cleanly, and the cause is identified.**
  Simulating 2025 from opening-day rosters gives Spearman **−0.21** vs actual
  standings. But: a **perfect** simulator scores only **+0.52** on this test
  (5th pct −0.05), and the same simulator fed **perfect player knowledge** on the
  same rosters scores **+0.43**. So the mechanics are sound and **the preseason
  projections are the limitation**. Roster churn is NOT the culprit — 81% of 2025
  starter points came from drafted players.
- 2025 had **no regular-season byes** (8 teams, all 14 games), so bye and
  unequal-game handling **cannot be backtested** — it is verified as invariants.

**M5 findings — the league changed shape underneath itself:**
- **2023 and 2024 were ONE-QB leagues. Only 2025 was 2-QB.** First QB off the
  board: pick **16**, then **35**, then **7**. QBs in the first three rounds:
  2, 0, **5**. Pre-2025 QB behaviour is not thin evidence, it is evidence about a
  different game. Combined with the M2 scoring change, **2025 is the only season
  resembling 2026** — and it was 8 teams.
- Team counts ran **8, 10, 8, now 9** — a size this league has never played. All
  pick numbers are normalised to a fraction of the draft.
- **Kylie Leahy — faced TWICE — has one draft (17 picks).** leah gottlieb has one,
  and it is 2024, so she has **zero** 2-QB evidence. §2.3 weights the four
  double-up managers most heavily and they are not the ones with the most data.
- Consensus baseline is **format-matched**: superflex (`rsf`) for 2025, standard
  (`ro`) for the 1-QB seasons. Scoring a 1-QB draft against superflex ranks would
  manufacture huge fake "reaches" at QB — format difference posing as personality.
- **A manager's own history can MISLEAD across the format change.** In the strict
  backtest two managers got worse than pure ADP, worst being **Jordan Britt at
  −0.27** — his 1-QB drafts give no hint he would take the first QB at pick 7.
  Hence `FORMAT_MATCH_WEIGHT = 3.0` on the format-matched season.
- Tendencies are shrunk toward the league prior with a 30-pick pseudo-count. With
  17–49 picks per manager this is not conservatism, it is the only way to avoid
  publishing confident nonsense.
- QB timing splits the league: **Britt, Leahy, Rogo, Sims** take their first QB
  inside the first 18% of the draft; **Benca, Jeff Boyd, Josh Boyd, Sharrett**
  wait past 33%.
- §1 spells a manager "R. Sharrett"; ESPN says "Rayne Sharrett" — explicit alias
  map, never fuzzy matching.

**M4 findings:**
- **The flex has NEVER started a tight end.** 364 flex starts across 2023–2025:
  56.3% RB, 43.7% WR, **0.0% TE**. §7.3 estimated ~4 RB / 4 WR / 1 TE. So
  replacement is **TE9** (not TE10) and **RB23.1** (not RB22); WR22 confirmed.
  Coherent with §3.4 — half-PPR plus 0.05/carry pushes flex value to backs.
  Caveat: measured in 8- and 10-team seasons, so direction is safe, level less so.
- **The §2.1 bye adjustment is tiebreaker-sized, not "first-class".** Computed
  honestly it is one week of VOR: mean **1.79 points** for free-bye players in the
  top 100, max 4.63, against VORs of 100–170. It still produces a visible rank
  lift (+8.1 mean rank gain vs ECR against +4.9 for everyone else) because players
  bunch tightly at the margins — but §2.1 hoped for more than the arithmetic gives.
  Only 4 teams qualify in 2026, which concentrates it.
- **The sack edge needed an explicit board term or it would have been lost.**
  Consensus prices sacks at zero (§3.3), and the M3 opportunity model DROPPED
  `sack_rate` for falling under the 0.45 stability cutoff. So `sack_adjustment =
  −0.434 × prior_sacks_over_expected` for QBs, using M3b's measured persistence.
  Range in 2026: **−8.3 to +9.2 points**. It moved Lamar Jackson from #2 to #4.
- **2026 QB-count answer: TWO.** Josh Allen (bye wk 7) and Lamar Jackson (wk 13) —
  neither bye is covered by my weeks 5/14, so QB3 would start 2 weeks and is worth
  17.9 points, against 26.4 points of upside from the best non-QB stash (§2.1, §3.2).
- Board is QB-heavy at the top (12 QB / 10 RB / 6 WR / 2 TE in the top 30),
  following from QB VOR leading and 18 QB starters. **This is the thing to
  sanity-check** — if it is wrong, it is wrong in the first two rounds.
- Tier breaks are re-run under ±5% projection noise; `tier_stability` records how
  often each survives, so a cliff that exists at only one exact projection is
  visible as false precision.
- **THE RAMS TRAP BIT A THIRD TIME, in the M4 board — fixed at the source
  2026-08-22.** nflverse spells them `LA`; ESPN, the crosswalk and therefore the
  board spell them `LAR`. `board_inputs.build` left-joined the bye table and the
  playoff-SOS table on the raw abbreviation, so **all 18 Rams carried a null
  `bye_week` AND a null `playoff_sos`** — Puka Nacua at overall rank 13,
  Stafford 36, Kyren Williams 41. `playoff_schedule_strength` had its own
  independent copy of the same bug, since it reads nflverse schedules directly.
- **What made it survive is that it was ACCIDENTALLY CORRECT.** `build.py` does
  `free_bye_week_5_or_14.fill_null(False)`, and LA's 2026 bye is week 11 — which
  genuinely is not in {5, 14}. So the §2.1 flag was right by luck, VOR and every
  `overall_rank` were **byte-identical before and after the fix**, and nothing
  looked wrong. In any season where the Rams bye in week 5 or 14 the whole roster
  silently loses the bonus. `playoff_sos` had no such luck: §2.4 calls weeks
  15–17 schedule strength a real draft criterion and it was simply missing for a
  top-15 player.
- **Fixed once, at the nflverse boundary, not a fourth time downstream.**
  `byes.team_weeks_played` now canonicalises through `crosswalk.normalize_team`,
  so the whole module speaks one spelling; `bye_weeks` asserts the vocabulary
  against `CURRENT_TEAMS` via `byes.assert_canonical_teams`, and
  `board_inputs.assert_team_fields_resolved` refuses to return a board where any
  non-free-agent has a null `bye_week` or `playoff_sos`. A left join fails as a
  NULL, not an error — that is the whole reason this class of bug is silent, so
  the assert is the fix and the normalisation is only the repair.
- **It was never only the Rams.** Normalising also folds 2016 `SD` → `LAC` and
  2016–19 `OAK` → `LV`, so a backtest board on those seasons was dropping two
  more franchises the same way. Verified 2016–2026: exactly 32 canonical teams
  every season, no collisions. `WSH → WAS` was already handled on the ESPN side
  and is now asserted from both directions.
- **`tier_stability` and tie-broken `overall_rank` are NOT reproducible run to
  run** — found while diffing the fix, pre-existing and NOT caused by it. Two
  runs of identical code differ on 77 `tier_stability` values and 4
  `overall_rank`s. `model.project`'s `.sort()` is not stable, so equal
  `model_points` land in a different row order; `tier_stability`'s `seed=17` then
  hands its noise vector to different players, and `rank("ordinal")` breaks ties
  differently. The VALUES are stable (0 players' `model_points` differ) — only
  the ORDER is. Same class as M7's "an unordered unique makes the whole sim
  irreproducible". Not fixed here; out of scope.

**M3b findings:**
- `ff_opportunity` lives in a **separate repo** (`nflverse/ffopportunity`), covers
  2016–2025, uses gsis ids, and exposes expected COMPONENTS — so they can be fed
  through the M2 engine and priced in §1 instead of its own scoring.
- Its headline `total_fantasy_points_exp` is **full PPR with 4-point passing TDs**
  (verified: 0.091 mean abs diff). Every public xFP number is priced for a game
  nobody in this league is playing — exactly ADD-§A's premise.
- **It ships the postseason** (weeks 19–22). Compared against regular-season
  actuals, deep-run players accrue phantom expected points and FPOE goes negative
  for almost everyone. Filter on `game_type == "REG"` via `game_id`.
- Null-`player_id` rows aggregate into a phantom player with 400+ games.
- `model_version` is **pinned to v1.0.0**; `latest` would silently rewrite history.
- **0.05/carry needs no model** — a carry IS the opportunity, so those points are
  expected by construction. **−1/sack has no upstream model** and had to be built
  from dropbacks against an opponent-adjusted, QB-neutral baseline.

**M1 findings worth remembering:**
- Sacks taken is **`sacks_suffered`**. Not `sacks` (doesn't exist), and NOT
  `def_sacks` (that is sacks *recorded by* a defender). Rushing attempts is
  **`carries`**; `attempts` is PASS attempts. Both are §1 scoring categories, so
  both are pinned by an ingest-time column contract.
- **2026 has only 4 free-bye teams: ARI, CAR, DAL, KC** (2025 had 8). Week-14 NFL
  byes do exist in 2026, so §2.1 holds — but the pool is half as wide, which makes
  the flag more selective, not less valuable.
- **The §11 gate is necessary but NOT sufficient.** "Resolves to exactly one ID"
  can pass while resolving to the *wrong person*. Live case found: nflverse assigns
  `espn_id 4686658` to a DB who last played in **1984**, but that id belongs to the
  2026 rookie RB Mike Washington Jr. (LV), who was ~10% rostered and carries no
  `espn_id` in nflverse. The direct-espn_id tier — the highest-confidence tier —
  produced exactly §6's "board recommends a retired player" failure while passing
  every assertion. `assert_resolutions_plausible()` now flags any resolution onto a
  player whose career ended >10 seasons before the target season. Position mismatch
  is deliberately NOT used: Travis Hunter is a CB in nflverse and a WR in ESPN, and
  that match is correct.
- **D/ST use negative ESPN ids**: `-16000 − proTeamId` (SF = `-16025`, BAL = `-16033`).
  Draft history stores them this way. `resolve_dst()` handles both that and team
  abbreviations; `WSH → WAS` is the only live abbreviation difference.
- 5 override rows are in `overrides/player_id_overrides.csv`, each with its evidence:
  1 bad nflverse `espn_id` (Mike Washington Jr.), 1 stale nflverse `espn_id`
  (Chris Manhertz, nflverse says 4071345, ESPN uses 2531358), and 3 UDFAs with
  genuinely no nflverse counterpart, marked `NO_NFLVERSE_MATCH` → `has_nflverse_data
  = False` so no projection is ever invented for them.
- Two real schedule anomalies live in the history window and are handled
  explicitly: 2022 BUF/CIN week 17 (cancelled, Damar Hamlin — both played 16 games,
  relevant to §2.5) and 2017 MIA/TB week 1 (postponed for Hurricane Irma, replayed
  in their week-11 bye, so week 1 became their functional bye).

### CONFIRMED 2026-08-20 from ESPN (`artifacts/espn_settings_2026.json`)
Everything §1 marked CONFIRM is now verified from the source, not inferred.

| Question | Answer |
|---|---|
| League | **Wildcats League**, 9 teams, `league_id` in `.env` |
| `playoff_weeks` | **[15, 16, 17]** — `reg_season_count=14`, matchups run to 17, `playoff_matchup_period_length=1` |
| `first_round_byes` | **2** — forced by structure: 6 teams over 3 one-week rounds. §2.4 holds |
| Keeper or redraft | **FULL REDRAFT** (`keeper_count = 0`) |
| Waivers | **rolling priority** (`faab = false`), confirming §9.3's model |
| Roster | matches §1 exactly; **IR is an 18th slot** outside the 17 — the assumption was right |
| D/ST `18-21` and `22-27` | **both really are 0.** ESPN omits zero-valued rules; both absent. §1 was NOT a transcription slip |
| D/ST yards `300-349` | **0**, likewise absent. §1 correct |
| Every §1 scoring rule | verified exactly, incl. `SKD −1` (sack taken) and `RA 0.05` (rush attempt) |
| My schedule | verified: 12 games, byes weeks 5 & 14, the four double-up opponents in weeks 1/10, 2/11, 3/12, 4/13 |
| Trade deadline | 2026-12-02 |
| **Seeding rule (§2.5)** | **WIN PERCENTAGE — confirmed 2026-08-20** from the standings page (ranks on `PCT`, with a `GB` column). **No structural handicap** from playing 12 games. Had it been raw wins, M6 measured the cost at **−7.9 pts P(playoffs)** and **−8.1 pts P(top-2)** — which is why it was worth confirming |
| League history | **3 seasons: 2023, 2024, 2025.** All drafts + rosters cached |

### STILL OPEN
- [x] `draft_date_time` — **RESOLVED 2026-08-22.** `2026-08-22 18:30` — not
      from asking, but because ESPN's `draftSettings.date` was just sitting in
      the `mSettings` payload the whole time. `draftSettings.pickOrder` (a list
      of `team_id`s in round-1 order) was there too, and it happened to already
      have my slot — **4** — hours before the draft, contradicting §5's
      "revealed ~1hr before" framing. §5 wasn't wrong that it's unknowable
      *until the commissioner sets it*; it just doesn't say WHEN that happens,
      and "roughly an hour before" is one commissioner's habit, not a platform
      guarantee. `live/profile.py::_detect_slot` reads both fields now, and
      `cli gui --auto` needs no `--slot` at all once they're set — it falls
      back to asking for one when they aren't.
- [ ] **M4 and M7 disagree on the QB count.** M4 says **TWO**, comparing a
      specific pair (Allen wk 7, Jackson wk 13, neither bye free) against the best
      upside stash: 17.9 points vs 26.4. M7 says **THREE**, over the distribution
      of quarterbacks actually drafted, scoring expected weekly points rather than
      ceiling. They do not measure the same thing. Resolve in M8, where the slot
      plans have to commit to one.
- [ ] League **team count changed every year** (2023: 8, 2024: 10, 2025: 8, 2026: 9).
      Positional-run dynamics do not transfer cleanly across seasons — M5 must
      weight by roster-slot count, not treat the three drafts as one sample.

# LEAGUE SPEC — §1, §2, §3 (verbatim from FANTASY_SPEC.md)

## 1. LEAGUE CONFIG

```yaml
platform: espn
league_id:                    # ← from your league URL
my_team: "First Down Syndrome"
teams: 9
draft_type: snake
my_draft_slot: unknown        # revealed ~1hr before draft — see §5
draft_date_time:              # ← still needed
waivers: rolling_priority

season:
  regular_season: weeks 1-14
  my_byes: [5, 14]            # I play 12 games, not 14
  playoff_teams: 6            # of 9 → 67% qualify
  playoff_weeks: [15, 16, 17] # inferred — CONFIRM in settings
  first_round_byes: 2         # standard for a 6-team, 3-week bracket — CONFIRM

opponents:
  # Faced TWICE — 8 of my 12 games:
  - {team: "Hodor's Hodors",       manager: "Camden Sims",    weeks: [1, 10]}
  - {team: "Personality Hires",    manager: "Kylie Leahy",    weeks: [2, 11]}
  - {team: "Clearing the Fields",  manager: "R. Sharrett",    weeks: [3, 12]}
  - {team: "Gibbs Me My Money",    manager: "Matthew Benca",  weeks: [4, 13]}
  # Faced ONCE:
  - {team: "A Chane Reaction",     manager: "Jeff Boyd",      weeks: [6]}
  - {team: "Unsolicited Dak Pics", manager: "Josh Boyd",      weeks: [7]}
  - {team: "Nothing Beats a JJett 2 H...", manager: "Hanna Rogo", weeks: [8]}
  - {team: "leah's team",          manager: "leah gottlieb",  weeks: [9]}

roster:
  total: 17
  starters: 10
  bench: 7
  ir: 1
  slots: {QB: 2, RB: 2, WR: 2, TE: 1, FLEX: 1, DST: 1, K: 1}
  maxima: {QB: 4, RB: 8, WR: 8, TE: 3, DST: 3, K: 3}
  # All IDP (DT/DE/LB/DL/CB/S/DB/DP), TQB, RB/WR, WR/TE, OP, P, HC = 0

scoring:
  passing:
    pass_yd: 0.04            # 1 pt / 25 yds
    pass_td: 6               # NOT the standard 4
    int: -2
    sack_taken: -1           # unusual, and material
    two_pt_pass: 2
    # OFF: attempts, completions, incompletions, PFD, 40+/50+ TD bonus, 300/400yd games
  rushing:
    rush_yd: 0.1
    rush_att: 0.05           # unusual: 0.05 pts per carry
    rush_td: 6
    two_pt_rush: 2
    # OFF: RFD, 40+/50+ TD bonus, 100/200yd games
  receiving:
    rec_yd: 0.1
    reception: 0.5           # half PPR
    rec_td: 6
    two_pt_rec: 2
    # OFF: targets, REFD, 40+/50+ TD bonus, 100/200yd games
  misc:
    fumble_lost: -2
    kr_td: 6
    pr_td: 6
    fumble_recovered_td: 6
    # OFF: total fumbles, team win, team loss, KR/PR yards
  kicking:
    pat_made: 1
    fg_0_39: 3
    fg_40_49: 4
    fg_50_59: 5
    fg_60_plus: 6
    fg_missed: -1            # flat, any distance
    # OFF: PAT att/missed, FG yards, total FG made/att, bucketed att/miss
  dst:
    sack: 1
    interception: 2
    fumble_recovered: 2
    safety: 2
    def_td: 6                # INT ret, fumble ret, KR, PR, blocked punt/FG ret
    blocked_kick: 2
    two_pt_return: 2
    one_pt_safety: 1
    points_allowed:          # bucketed; continuous PA is OFF
      {0: 5, 1-6: 4, 7-13: 3, 14-17: 1, 18-21: 0, 22-27: 0, 28-34: -1, 35-45: -3, 46+: -5}
    yards_allowed:           # bucketed; continuous YA is OFF
      {"<100": 5, 100-199: 3, 200-299: 2, 300-349: 0, 350-399: -1,
       400-449: -3, 450-499: -5, 500-549: -6, "550+": -7}
    # OFF: forced fumbles, stuffs, passes defensed, total tackles, KR/PR yards
```

---

## 2. SCHEDULE STRUCTURE — the biggest edges live here

### 2.1 Bye-week arbitrage ← unique to your schedule
Your fantasy byes are **weeks 5 and 14**. In those weeks nothing on your roster matters.

Therefore: **any player whose NFL bye falls in week 5 or week 14 has a free bye.** For you, and only for the four other teams sharing those bye weeks, those players are strictly worth more. Nobody's rankings price this, because it's a property of *your schedule*, not the player.

The payoff compounds at QB. You're carrying 3 QBs largely to cover byes — but if both your starters bye in weeks 5 or 14, **you may only need two, freeing a bench spot for an upside swing.** In a league with a rich waiver wire (§3.2), a spare bench slot is worth real points.

Make this a first-class term in the draft board, not a tiebreaker:
```
bye_adjustment[player] = +full_week_value   if nfl_bye ∈ {5, 14}
                       = 0                  otherwise
```
Compute the actual point value (roughly one start's worth of expected points over a replacement-level fill-in) rather than a hand-tuned nudge. Then run the QB-count optimization explicitly: *given these two QBs' byes, does a third QB beat the best available RB/WR upside stash?*

### 2.2 Your season ends in week 13
Week 14 is your bye, so **your record is locked a full week before everyone else's.** Two consequences:

1. **Front-load urgency.** Your must-win window is weeks 1–13. Waiver aggression, trade deadline pushes, and win-now moves should peak earlier than your leaguemates' instincts. If you're on the seeding bubble in week 13, that's your last chance to act on it.
2. **Week 14 is a completely free week.** No lineup to set, no game to lose. Use it to churn the roster purely for weeks 15–17: drop anyone who doesn't help in the playoffs, claim upside stashes, absorb risk you'd never take during the season. Zero downside. Build this as an explicit `/week14` command — it's a different job from a normal waiver run.

### 2.3 Two-thirds of your schedule is four managers
You play these four **twice** — 8 of your 12 games:

| Opponent | Manager | Weeks |
|---|---|---|
| Hodor's Hodors | Camden Sims | 1, 10 |
| Personality Hires | Kylie Leahy | 2, 11 |
| Clearing the Fields | R. Sharrett | 3, 12 |
| Gibbs Me My Money | Matthew Benca | 4, 13 |

And these four **once**: A Chane Reaction (Jeff Boyd, wk 6), Unsolicited Dak Pics (Josh Boyd, wk 7), Nothing Beats a JJett 2 H (Hanna Rogo, wk 8), leah's team (leah gottlieb, wk 9).

**Weight the four double-up managers at 2× in the opponent model and in in-season scouting.** A juggernaut you face twice hurts far more than one you face once. During the draft this also creates mild blocking value: denying a double-up opponent a player at their position of need has double the downstream leverage. Keep it as a tiebreaker, never a reason to reach.

### 2.4 Six of nine make the playoffs — this changes your objective function
A 67% qualification rate means **making the playoffs is the default outcome, not the achievement.** Only three teams miss. Meanwhile, in a 6-team/3-week bracket the top two seeds get first-round byes — they need to win 2 games instead of 3. Between evenly matched teams that's roughly **12.5% title odds without a bye vs. 25% with one.** A first-round bye is worth about as much as everything else combined.

So the system should optimize for **championship probability**, not points or wins:
- Marginal regular-season wins are worth little until you're near the bubble, and a lot when contending for a top-2 seed.
- Roster variance is *good* — you almost certainly make the tournament, so you want the version of your team with the highest ceiling in weeks 15–17.
- **NFL weeks 15–17 schedule strength is a real draft criterion.** Prefer players on offenses with favorable late-season matchups; flag dome/weather risk for outdoor players in December; note NFL teams likely to rest starters once seeding is clinched.

Every in-season recommendation should report its delta in championship probability, not just projected points.

### 2.5 Verify: unequal games
With 9 teams over a 14-week regular season, there are 14 bye slots across 9 teams — so **some teams play 13 games and you play 12.** Check whether your league seeds on raw wins or win percentage. On raw wins you're structurally disadvantaged, which raises the value of every single win. The season simulator must model this correctly or its seeding output will be wrong.

---

## 3. WHAT YOUR SETTINGS MEAN

### 3.1 Nine teams flattens the 2-QB effect
A 12-team 2-QB league starts 24 QBs against ~32 NFL starters, making QB brutally scarce. **Yours starts 18.** QB18 is a real starter. Rough back-of-envelope under your exact rules (the pipeline computes real ones):

| Pos | Elite | Replacement | VOR |
|---|---|---|---|
| QB | ~400 | ~265 (QB18) | **~135** |
| RB | ~295 | ~170 (RB22) | **~125** |
| WR | ~250 | ~155 (WR22) | ~95 |
| TE | ~185 | ~115 (TE10) | ~70 |

QB leads, but it's near-tied with RB, not a blowout. Secure two quality QBs without panicking; don't torch your first two picks there. The residual risk is a **run** — 18 slots across 9 teams means once two or three managers grab their second QB, the rest stampede.

### 3.2 The waiver wire stays rich all season
Only **153 roster spots exist** (9 × 17) against 500+ fantasy-relevant players. Consequences: startable players available every week · **draft for upside, not floor** — a safe pick is replaceable from the wire, a league-winner isn't · don't hoard handcuffs · even QB is streamable at the margins (~27 rostered vs ~32 starters, plus injury promotions).

### 3.3 Sacks taken at −1 is a real QB differentiator
A QB sacked 50 times loses 50 points — 12–15% of his season. Effectively no other league uses this, so no public ranking prices it. Model projected sacks from O-line pressure rate allowed, scheme, and the QB's own historical sack rate. Cheap, durable edge.

### 3.4 0.05/carry + half PPR favors bell-cow RBs
*300-carry/30-catch grinder vs. 150-carry/70-catch receiving back:*
- **Your league:** 30.0 vs 42.5 from carries+catches → gap of 12.5
- **Full PPR, no carry points:** 30 vs 70 → gap of 40

The pass-catching-back premium shrinks ~two-thirds. Treat projected **carries** as a first-class input.

### 3.5 D/ST and K reward active management
D/ST has big negatives (−7 at 550+ yards, −5 at 46+ points) against modest upside (5 for a shutout). Yardage buckets mean you want opponents who are low-volume and run-heavy, not merely turnover-prone. Stream weekly; always check the downside case. Kicker is distance-weighted (5 at 50–59, 6 at 60+) with −1 per miss — favor accurate big legs on offenses that stall in FG range. Still never before the last two rounds.

### 3.6 Roster construction
| | Count | Note |
|---|---|---|
| QB | **2–3** | 3 by default; **2 if both bye in wk 5/14** (§2.1) |
| RB | 5–6 | |
| WR | 5–6 | |
| TE | 1–2 | |
| K | 1 | |
| D/ST | 1 | stream weekly |
