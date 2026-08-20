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
32 D/ST), 2025 rosters (127+10), and the 2023/2024/2025 drafts. 105 tests pass.
**Next: M2, the scoring engine.**

```bash
uv run python -m ff_agent.cli status      # cache inventory + staleness
uv run python -m ff_agent.cli byes        # §2.1 free-bye teams
uv run python -m ff_agent.cli crosswalk   # THE GATE — needs .env
uv run python -m ff_agent.cli verify      # cookie pre-flight, run draft morning
uv run python -m ff_agent.cli offline     # prove the draft-day path
uv run pytest                             # 105 pass
```

Layout: `ff_agent/config.py` (§1 constants, credentials) · `data/cache.py`
(parquet + staleness) · `data/nflverse.py` (ingest + column contract) ·
`data/byes.py` (§2.1) · `data/espn.py` (league state, read-only) ·
`data/crosswalk.py` (§0.2 gate) · `overrides/player_id_overrides.csv` (tracked —
human decisions belong in git).

**Findings worth remembering:**
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
