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

**Next: M7 (Monte Carlo draft simulator), then M8 (nine slot plans).**

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
uv run python -m ff_agent.cli settings    # refresh league settings JSON
uv run python -m ff_agent.cli verify      # cookie pre-flight, run draft morning
uv run python -m ff_agent.cli offline     # prove the draft-day path
uv run pytest                             # 221 pass
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

**M6 findings:**
- **§2.5 ANSWERED, and it is not a technicality.** 2026: five teams play 12 games
  (you, JJett, Gibbs, Clearing, leah), four play 13. With every team identical,
  seeding on **raw wins** costs a 12-game team **−7.9 pts of P(playoffs)** and
  **−8.1 pts of P(top-2 seed)** versus win percentage. Under win percentage the
  gap vanishes. **Confirm the rule on the standings page** — it is worth ~8 points
  of the thing §2.4 calls as valuable as everything else combined. The simulator
  ships BOTH rules and reports the delta.
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
| League history | **3 seasons: 2023, 2024, 2025.** All drafts + rosters cached |

### STILL OPEN
- [ ] `draft_date_time` — unknown. If it compresses, triage order is 1 → 2 → 3 → 9.
- [ ] **Seeding on raw wins or win pct (§2.5).** `playoff_seed_tie_rule = H2H_RECORD`
      is only the *tiebreak*. ESPN standings are win-pct based by default, which
      would neutralise the 12-vs-13-game disadvantage — but confirm visually in the
      standings page before the season simulator (M6) relies on it.
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
