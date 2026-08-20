# Fantasy Football Draft Agent + League Manager — Build Spec
### "First Down Syndrome" · ESPN · 9 teams · 2-QB · Half-PPR · 6-pt Pass TD · Snake · Rolling waivers

> **How to use this file.**
> ```bash
> mkdir ff-agent && cd ff-agent
> # save this file here as FANTASY_SPEC.md
> claude
> ```
> First message to Claude Code:
> *"Read FANTASY_SPEC.md. Don't write code yet. Ask me anything ambiguous, then give me a build plan for Milestone 1 only."*
>
> Build one milestone at a time (§11). Don't let it one-shot the whole thing — you'll get code that silently produces wrong numbers, which is worse than no code.

---

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

---

## 4. ARCHITECTURE

**The pick clock problem.** You get 60–90 seconds. An LLM round trip with searches takes 20–60s and is non-deterministic. Never build "ask Claude every pick."

| Phase | When | Thinking by | Latency budget |
|---|---|---|---|
| **Pre-draft** | Weeks before | LLM + research + full pipeline | Unlimited |
| **Slot reveal** | T−60 min | *Nothing.* Load a precomputed file | **< 5 sec** |
| **Live draft** | On the clock | Deterministic Python over the board | **< 1 sec** |
| **In-season** | Tue/Wed/Thu/Sun | LLM + scheduled jobs | Minutes |

Artifacts: **`board.json`** (slot-independent: projections, tiers, VOR, ADP, bye adjustment, flags) · **`plan_1..9.json`** (one per possible slot) · **`opponents.json`** (the 8 managers' pick behavior) · **`season_sim.json`** (championship-probability model).

**Must run fully offline.** Draft-day WiFi will betray you.

---

## 5. THE UNKNOWN-SLOT PROBLEM

You learn your slot one hour before. Design so that hour needs **zero computation**.

**Slot-independent (~90% of the work):** projections, scoring engine, VOR, tiers, bye adjustments, opponent model, ADP variance.
**Slot-dependent:** turn schedule, realistic availability at each turn, round-by-round plan.

**Precompute all nine plans.** Run 10,000+ simulated drafts per slot, all nine slots. Other teams pick per `opponents.json`; you pick per the §7.2 rule. Score each resulting roster by **championship probability** from the season simulator (§8), not raw points — that's the actual objective.

Each `plan_{slot}.json`: your exact pick numbers · expected best-available tier per position at each turn with probabilities · recommended positional shape by round · branch points ("if two QBs go before pick 14, pivot to X") · distribution of final roster strength.

**Snake math for 9 teams.** Gaps alternate `2(9−d)+1` after odd rounds, `2d−1` after even rounds:

| Slot | Gaps | Character |
|---|---|---|
| 1 | 17, 1 | Long droughts, then back-to-back pairs |
| 5 | 9, 9 | **Perfectly even** — odd team counts make the middle slot uniquely smooth |
| 9 | 1, 17 | Immediate pair, then long droughts |

Wheel slots (1, 9) allow correlated pairs but force 17-pick waits, so survival probability dominates there. Slot 5 is the easiest to plan. The nine plans should be genuinely different strategies, not one list re-indexed.

**The T−60 drill.** One command, no computation:
```bash
./draft --slot 6     # loads plan_6.json + board.json, prints the sheet
```
Rehearse twice with a random slot. Print all nine cheat sheets on paper — a laptop failure shouldn't end your draft.

---

## 6. DATA LAYER

**Historical / performance.** `nflreadpy` (or `nfl_data_py`) — nflverse, free, no key. Play-by-play to 1999, weekly stats, snap counts (2012+), depth charts, Next Gen Stats, rosters, **schedules including NFL bye weeks**, ID crosswalks. Pull 5 seasons. You specifically need **sacks taken** and **rushing attempts** — both are scoring categories here.

**League state — ESPN.** Unofficial v3 endpoints; private leagues need `espn_s2` and `SWID` cookies from browser dev tools (Application → Cookies → fantasy.espn.com). The `espn-api` package wraps this. Two warnings: **cookies expire** — re-grab draft morning and verify before you're on the clock; **ESPN's live draft feed is flakier than Sleeper's** — build and rehearse manual entry first, treat polling as a bonus. ESPN also exposes **waiver priority order for all teams**, needed for §9.3.

**Signals.** Vegas implied team totals — `(total/2) ± (spread/2)`, strongest single predictor of scoring environment; free tiers exist, verify NFL coverage and cache hard. Weather — Open-Meteo, free, no key; map stadium coords, flag domes, only **wind >15 mph** matters much. News — scheduled web research, respect robots.txt and terms of service.

**⚠️ Build the ID crosswalk first.** nflverse uses `gsis_id`, ESPN its own. **Names do not join reliably** — suffixes, initials, duplicates. `import_ids()` gives a crosswalk. Build a canonical player table and **hard-assert every rostered/drafted player resolves to exactly one ID; fail loudly on any miss.** More of these projects die here than anywhere else, and silently — you find out when the board recommends a retired player.

---

## 7. PRE-DRAFT PIPELINE

### 7.1 Scoring engine
Stat line → points under §1. **Validate by recomputing last season's weekly scores against ESPN's recorded scores. They must match exactly.** Sacks taken, rushing attempts, and the two bucketed D/ST tables are where a generic function will silently be wrong.

### 7.2 Projections
1. Ingest a consensus baseline as an anchor.
2. **Recompute into your scoring.**
3. Build your own opportunity-based projections from nflverse: target share, route participation, air yards (WR/TE); snap share, **carry share**, RZ touches (RB); pass volume, team total, **projected sacks taken** (QB).
4. Blend with consensus rather than replacing it — consensus encodes offseason news your historical data can't see. Weight your model higher where you have signal it lacks (sacks, carries), lower where it doesn't.
5. Attach an uncertainty band per player. The Monte Carlo needs these.
6. Attach **weeks 15–17 matchup strength** and the **week 5/14 bye flag** (§2.1) as separate fields.

### 7.3 VOR with 9-team replacement levels
```
replacement_rank[QB] = 9 × 2 = 18
replacement_rank[RB] = 9 × 2 + flex_share[RB] ≈ 22
replacement_rank[WR] = 9 × 2 + flex_share[WR] ≈ 22
replacement_rank[TE] = 9 × 1 + flex_share[TE] ≈ 10
VOR = proj − proj[replacement_rank[pos]]  +  bye_adjustment
```
Allocate the 9 flex slots by historical flex usage (~4 RB / 4 WR / 1 TE in half PPR), not evenly.

### 7.4 Tiers
Cluster by projected-point gaps within position. Tiers beat ranks — #6 vs #7 RB is noise, a tier cliff is real. Store `tier_id` and `players_remaining_in_tier`. Build QB tiers carefully; where the cliff sits determines your draft shape.

### 7.5 Opponent model
National ADP describes the average drafter, not your eight leaguemates, and in a 2-QB league it misstates positional demand entirely. Pull your league's past ESPN drafts and build per manager: positional tendency by round, average reach vs. ADP, team bias, and **how early QBs actually went in your league**. Where history is thin, fall back to 2-QB/superflex ADP — never standard ADP.

Weight the four double-up managers (§2.3) most heavily; their rosters affect two-thirds of your schedule.

---

## 8. SEASON SIMULATOR ← the objective function

Everything else optimizes against this. Build it before the draft, run it weekly after.

1. Simulate weekly scores for all 9 teams from their rosters, projections, and variance bands.
2. Run your **actual schedule** (§1), including your weeks 5 and 14 byes and the unequal-games issue (§2.5).
3. Seed per league rules, apply the 6-team bracket with top-2 first-round byes.
4. Output: **P(playoffs), P(top-2 seed), P(championship)** for you.

Then every recommendation — a draft pick, a lineup, a waiver claim, a trade — reports its **delta in championship probability**. Given a 67% qualification rate and a bye worth roughly a doubling of title odds, this will systematically tell you to chase seeding and playoff-week ceiling over marginal regular-season floor. That's the correct answer for this league and it's the opposite of default fantasy advice.

---

## 9. IN-SEASON MANAGER

### 9.1 Weekly cadence
| When | Job |
|---|---|
| **Tue AM** | Waiver claim list + priority-spend decision |
| **Wed AM** | **Post-clear free-agent sweep** (§9.3) — a lot of the edge is here |
| **Thu 4pm ET** | Lock TNF decisions — irreversible after |
| **Sat** | Friday injury designations; flag every Q/D |
| **Sun ~11:15am ET** | **Inactives drop ~90 min before kickoff.** Final swap. Highest-leverage 15 minutes of the week, and the one most people miss |
| **Sun night / Mon** | Next-week D/ST stream, waiver prep, refresh season sim |
| **Week 14** | `/week14` — free-week roster churn for the playoffs (§2.2) |

### 9.2 Start/sit
Solve the lineup as an assignment problem across slots — greedy slot-filling gets the FLEX wrong.

Inputs by rough importance: projected points under your scoring · **implied team total** · volume trend (3-game rolling target share, route %, snap %, **carry share**, RZ touches — volume is far stickier than efficiency) · opponent strength by position via EPA/success rate allowed (not raw fantasy points allowed, which is schedule-contaminated) · game script · weather · injury designation and practice participation. **For QBs**, add projected sacks taken — an elite rush against a bad line costs 5–6 points before anything else.

**Floor vs. ceiling, driven by the season sim.** Project the matchup margin first: heavy favorite → high floor, protect the win; big underdog → volatile ceiling, you need variance. Layer the standings context on top — when you're safely in but chasing a top-2 seed, variance is cheap; when a win swings the bye, floor wins.

### 9.3 Waivers — rolling priority, not FAAB
**Priority is a depleting, indivisible asset.** You can't bid small; every successful claim drops you to last.
```
claim if:  marginal_value(player) > option_value(holding priority)
```
`marginal_value` = expected ROS points gained over whoever they'd replace **in your starting lineup** (bench upgrades ≈ 0), expressed as championship-probability delta. `option_value` = P(better target appears soon) × what you'd forgo at the back of the line.

- **Nine teams makes priority cheap.** The queue is only 9 long; you climb back fast. Spend more freely than FAAB intuition suggests — hoarding #1 priority into November is usually a mistake here.
- **Model P(claim succeeds).** ESPN exposes every team's priority. Estimate competitor interest from their roster holes. If you're 7th and three teams ahead share the gap, the claim likely fails — which costs nothing.
- **Order your claim list.** ESPN processes claims in your order and priority only drops on a *successful* claim. Put the player you'd genuinely burn priority on first, then fallbacks. Strictly better than a single claim.
- **Flag the free-agency path.** Predict who clears unclaimed and grab them Wednesday at **zero priority cost**. In a 9-team league a lot of useful players clear. The tool should say explicitly: *"claim this one; this one will clear — just grab him Wednesday."*
- **QB claims are special.** Any startable QB hitting the wire is worth top priority in a 2-QB league.
- **Front-load aggression through week 13** (§2.2), then go maximally speculative in week 14.

### 9.4 Trades
Evaluate on championship-probability delta, not generic trade charts — all of which assume 1-QB, 12-team leagues and will misprice both QBs and depth here. Depth is cheap in a 9-team league, so consolidating two mid pieces into one stud is usually favorable. Account for weeks 15–17 NFL schedules, your weeks 5/14 byes, and whether the trade partner is one of your four double-up opponents — strengthening them costs you twice.

---

## 10. GUARDRAILS

- **Never auto-submit** lineups, claims, or trades. Recommend; you confirm.
- **Log every recommendation with timestamp and inputs.** At season's end you can audit and improve. Without logs you learn nothing.
- Cache aggressively; nflverse weekly, odds hourly at most.
- **Fail loudly on stale data** — an optimizer silently running week-3 data in week 8 is worse than none.
- Re-verify ESPN cookies each session; fail with a clear message, not a stack trace.
- Sanity checks: a K before round 13, 1 QB rostered after round 11, or a lineup set for week 5 or 14 all mean something is broken.

---

## 11. BUILD ORDER

Unlimited time, so build the full pipeline. Each milestone ships with a passing test.

1. **Data layer + ID crosswalk.** Test: every player on every league roster resolves to exactly one ID, zero unmatched.
2. **Scoring engine.** Test: recomputed weekly scores match ESPN exactly — including sacks taken, carries, both D/ST bucket tables.
3. **Projection model.** Test: backtest last season; blended projections beat raw consensus on rank correlation under your scoring.
4. **VOR + tiers + bye adjustment → `board.json`.** Test: top 50 sanity-checked; every divergence from public rankings has an explanation you agree with, and week 5/14 bye players visibly rise.
5. **Opponent model → `opponents.json`.** Test: predicts last year's picks in your league better than national ADP.
6. **Season simulator → `season_sim.json`.** Test: reproduces last season's final standings from preseason projections within a reasonable band; handles your byes and unequal games correctly.
7. **Monte Carlo draft simulator.** Test: 10k sims from a known slot reproduce last year's draft's broad shape.
8. **All nine plans → `plan_1..9.json`.** Test: plans differ meaningfully; slot 5 shouldn't read like slot 1.
9. **Live draft loop.** Test: full mock, sub-second, manual entry works with WiFi off, `--slot N` loads in under 5 seconds.
10. **In-season jobs, incl. `/week14`.** Test: replay several weeks of last season; compare optimizer lineups to what was actually started and recommended claims to what actually paid off.

**Two rehearsals** before draft day: a full mock at a randomly assigned slot, and a cold-start T−60 drill. Print all nine cheat sheets.

---

## 12. CLAUDE CODE SETUP

- **`CLAUDE.md`** in the project root: paste §1, §2, and §3 into it, plus "never auto-submit" and the ID-crosswalk rule. Read every session, so you stop re-explaining that this is a 9-team 2-QB league with weeks 5 and 14 free.
- Recurring jobs as **custom slash commands** in `.claude/commands/` — `/waivers`, `/lineup`, `/draft`, `/simulate`, `/week14`.
- A **subagent** in `.claude/agents/` for news and injury research, so it doesn't flood your main context.
- Ask for **plan mode** on the data layer, projection model, and both simulators.
- Keep it in git; commit after every milestone. You will want to roll back.
