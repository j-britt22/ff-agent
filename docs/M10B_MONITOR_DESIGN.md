# M10b — THE IN-SEASON MONITOR

### A container that watches the league on a cadence and emails me what to do about it

> **Status: DESIGN. Nothing here is built yet.** This is the plan §11 step 10 asks for
> ("in-season jobs, incl. `/week14`"), written after a reconnaissance pass over the
> data sources. §2 records what that pass found — several of those findings contradict
> what the spec assumed, and they are the reason this document exists rather than a
> ticket saying "add a cron job".

---

## 0. WHAT THIS IS, IN ONE PARAGRAPH

A Docker container on an always-on machine wakes up on the §9.1 cadence, reads the
league through the existing read-only ESPN layer, recomputes rest-of-season value
and championship probability with the M2–M7 engines, and **emails me an ordered list
of actions to take myself**. It monitors the waiver wire, it profiles the other eight
rosters, it hunts trades, and it does all of it against the objective §8 already
established: **Δ P(championship)**, not points. It never touches a write endpoint.

---

## 1. DECISIONS (settled 2026-08-23)

| Decision | Choice | Why |
|---|---|---|
| **Host** | Always-on box I own, Docker Compose | ESPN cookies never leave the house; the parquet cache persists between runs; cron fires to the minute, which the Sunday 11:15am ET job needs |
| **Channel** | **Email only** (SMTP, Gmail app password) | Free, rich, threaded, archived — and the archive *is* §10's recommendation log in human-readable form. See §6.3 for the problem this creates and the fix |
| **LLM role** | Deterministic core + narrow LLM layer | Every number is auditable Python. Claude adds the news/injury/role layer and writes the prose. Degrades to numbers-only when the API is unreachable |
| **Scheduler** | `supercronic` inside the container | Purpose-built for containers: logs to stdout, no PID-1 reaping problems, honours `TZ`. Host cron would work but puts the schedule outside the artifact that git tracks |
| **Timezone** | `TZ=America/New_York`, asserted at startup | Every time in §9.1 is ET. See finding **F7** |
| **Autonomy** | **Zero.** Recommend, never act | §0.1. Not a phase-two feature — see §9 |

---

## 2. RECONNAISSANCE FINDINGS — what shapes the build

These were measured against the live data sources on 2026-08-23, not assumed.

### F1. There is NO format-matched rest-of-season consensus. This is the biggest problem in M10b.

M3 shipped on FantasyPros superflex ECR (`ecr_type = "rsf"`) because §7.5 is explicit
that standard ranks misstate positional demand in a 2-QB league. That list is
**scraped only in the preseason**. Verified on the DynastyProcess ECR table (14
`ecr_type` values, 2019-12-27 → 2026-08-21): `rsf` has exactly **one** snapshot
between 2025-09-01 and 2025-12-31 — 2025-09-05 — and then nothing until the
following August.

The lists that *do* update weekly in-season (13 snapshots, 2025-10-03 → 2025-12-26)
are `do dp drk dsf ro rp wo wp wsf`. Of those:

| List | Redraft? | Superflex? | Horizon | Usable for ROS? |
|---|---|---|---|---|
| `rsf` | yes | **yes** | rest-of-season | **not scraped in-season** |
| `wsf` | yes | **yes** | **this week only** | no — but exactly right for the weekly jobs |
| `ro` / `rp` | yes | **no (1-QB)** | rest-of-season | **wrong format**, and IDP-polluted |
| `dsf` | dynasty | yes | multi-year | wrong question |

Measured on the 2025-10-31 snapshot: **`ro`'s top 24 contains ZERO quarterbacks.
`wsf`'s contains 13.** And `ro`'s overall #1 that week is Zack Baun, a linebacker —
the redraft-overall page is scraped with IDP folded in.

M5 already recorded exactly what happens if this is ignored:

> *"Scoring a 1-QB draft against superflex ranks would manufacture huge fake reaches
> at QB — format difference posing as personality."*

The same mistake here manufactures **fake waiver value at QB**, in a league where
§9.3 says "any startable QB hitting the wire is worth top priority" — so the error
would land precisely on the decisions it most matters for.

**Resolution: anchor rest-of-season on POINTS, not RANKS.** A rank has a format; a
points projection does not. ESPN publishes per-week projected points already
expressed in this league's exact scoring — and M2 proved to the decimal that our
ruleset reproduces ESPN's totals from ESPN's own stat line, on both rulesets. So:

- **ROS anchor** = Σ ESPN projected points over remaining weeks, recomputed through
  our M2 engine from the projected stat line (never read as a raw total — see F2).
- **Weekly anchor** = `wsf`, which is format-matched and exactly the right horizon.
- **Cross-check only** = `rp`, for non-QB ordering. Never for QB. Never as the anchor.
- Our own model blends on top at a **fitted, walk-forward weight**, exactly as M3
  did — and on M3's evidence (model alone 0.687 vs consensus 0.761) the expectation
  is that the weight comes back small. If it comes back at zero, that is the finding.

### F2. ESPN gives per-week projected AND actual points, with the applied-rule breakdown

`Player.stats[scoring_period]` carries `points`, `projected_points`, `breakdown`,
`projected_breakdown`, and `points_breakdown` — the last being **per-rule applied
points**, the same object M2's Layer A gate matched exactly.

Two consequences:

1. The M3 trap is live here. ESPN's **season** projections report yardage *per game*
   while every other field is a season total. Any in-season code reading the
   projected stat line must apply the same correction, or every projection is ~17×
   wrong in the yardage terms.
2. **The M2 gate becomes a weekly tripwire.** Recompute last week's actual scores
   from ESPN's own stat line and compare to ESPN's recorded total. This league
   already changed its scoring once, after 2024 (`PC` and `INC` removed). If a rule
   moves mid-season, the monitor finds out the following Tuesday instead of in
   January. Cheap, and it fails loudly.

### F3. The historical free-agent pool IS reconstructible — which is what makes the gate possible

I had this down as the main threat to §11 step 10: ESPN does not retain "who was a
free agent in week 6", so a no-lookahead waiver backtest looked impossible. It isn't.

- `League.load_roster_week(week)` requests `mRoster` with `scoringPeriodId=week` and
  rebuilds **every team's roster as of that week**.
- `League.transactions(scoring_period=N, types={'FREEAGENT','WAIVER','WAIVER_ERROR'})`
  returns adds, drops, waiver claims — **and failed claims**.
- `League.box_scores(week)` returns each lineup with `slot_position`, i.e. exactly
  who was **started** versus benched.

So the week-W free-agent pool is `draftable universe − ∪(week-W rosters)`, and "did
the claim pay off" has real ground truth. **`WAIVER_ERROR` is the gift**: it records
the claims that *lost*, which is the only direct evidence available for calibrating
§9.3's P(claim succeeds) model. Nothing else in the league exposes a counterfactual.

### F4. `player_owned_espn` is null in-season, so the obvious control has to be rebuilt

The ECR table carries an ESPN-rostership column, which would have been a ready-made
control baseline ("just take whoever's rostership is spiking"). It is **null for
every in-season row** — 0 of 6,038 `wsf` rows from October 2025 onward. Same class as
M7's discovery that ESPN's ownership percentages are *today's*, not historical.

The control is instead built from the transaction log: **"the player most added
across the league that week"**, recoverable from `transactions()`. That is a better
control anyway — it is not a proxy for what the other eight managers did, it *is*
what they did.

### F5. ESPN publishes its own playoff odds, free

`Team.playoff_pct` is on every team object. M6 shipped with a measured ceiling and a
standings test that does not pass cleanly; an independent weekly second opinion on
P(playoffs) costs nothing and is worth having beside our own number. Where they
disagree sharply, that is a prompt to look, not a number to average.

### F6. Vegas odds and weather were deferred out of M1. The debt comes due here.

§0.5 records them as "§9 weekly signals, not crosswalk inputs" and pushed them out of
Milestone 1. M10b is §9. ADD-§F ranks **opponent implied total** as the single best
D/ST streaming input, and §1's bucketed yards-allowed (+5 under 100 down to **−7 at
550+**, with continuous yards off) needs an explicit **opponent total-YARDS**
projection that essentially no public model produces. New external dependencies,
both cached hard and both optional-degrading (§7.4).

### F7. Timezone is load-bearing and trivially easy to get wrong

Every time in §9.1 is Eastern. A container defaults to UTC. A job written as `15 11 *
* 0` in a UTC container fires at **06:15 or 07:15 ET** depending on daylight saving —
four or five hours before the inactives it exists to read, on the one job §9.1 calls
"the highest-leverage 15 minutes of the week". Set `TZ=America/New_York` in the image
and **assert it at startup**, alongside the §10 alarm that already refuses to set a
lineup for weeks 5 or 14.

### F8. Email-only makes the urgent tier a design problem, not a plumbing problem

Email is the right home for the Tuesday waiver list and the weekly trade report. It
is a **weak channel for an 11:15am Sunday alert**, because it buzzes the phone only
if a rule says it should. Stated plainly rather than papered over. The fix is in §6.3
and it takes about two minutes; the design also keeps the notifier behind an
interface so a push channel is a ~30-line addition, never a rewrite.


### F9. The lineup does not lock all at once, and the lock calendar is not weekly-periodic

ESPN locks each player individually at **his own game's kickoff**. So "set the lineup"
is not one decision — it is a sequence of irreversible per-slot commitments made under
increasing information, and the sequence is a different shape every week.

Measured on the real 2026 schedule (`nflverse` `schedules`, which already carries
2026 with `weekday`, `gameday` and `gametime`):

| 2026 week | Lock times for that week |
|---|---|
| **1** | **Wed 9/9 20:20** · Thu 9/10 **20:35** · Sun 13:00 / 16:25 / 20:20 · Mon 20:15 |
| 15 *(playoff R1)* | Thu 12/17 · **Sat 12/19 17:00** · **Sat 12/19 20:20** · Sun 13:00 / 16:05 / 16:25 / 20:20 · Mon |
| 16 *(semifinal)* | Thu 12/24 · **Fri 12/25 13:00** · **Fri 12/25 16:30** · **Fri 12/25 20:15** · Sun ×4 · Mon |
| 17 *(final)* | Thu 12/31 · Sun 1/3 13:00 / 16:05 / 16:25 / 20:20 · Mon 1/4 |

That is **six to nine distinct lock times per week**, and the irregularities land
exactly where §2.4 says the value is:

- **Week 1 opens on a WEDNESDAY.** The first lock of the 2026 season is not Thursday.
- **Week 16 — my semifinal — has three Christmas Day games**, all Friday.
- **Week 15 — my quarterfinal — has two Saturday games**, at 17:00 and 20:20.
- Thursday kickoff is not even a fixed time: 20:35 in week 1, 20:15 thereafter.
- Sunday's "late" window is really **two** windows twenty minutes apart, 16:05 and 16:25.
- 2025 had a **Sunday 09:30 London kickoff** in week 4 — which a fixed 11:15 check
  misses by nearly two hours.

§9.1's four fixed slots (Thu 12:00 · Sat 10:00 · Sun 09:00 · Sun 11:15) would miss the
Wednesday opener entirely, miss all three Christmas games, run its Saturday check
seven hours before a 17:00 Saturday kickoff, and leave the whole Sunday late slate with
no inactive check at all.

**Resolution: derive decision points from the schedule, never from the clock.** The
crontab becomes a frequent tick; `clock.py` decides whether a checkpoint is due from
the actual kickoff times of *my own starters*. That is robust to Wednesday openers,
Christmas, London, Saturday playoff games and flex scheduling without a single
special case.

### F10. Availability by injury designation, measured — and the QB asymmetry is large

The option value of waiting is mostly the probability that a player's status *changes*,
so it needs `P(does not play | Friday designation)`. `nflverse` `injuries` carries
`report_status` and `practice_status` weekly and keys on **`gsis_id`** — it joins
straight to our canonical ID with no crosswalk hop.

2025 regular season, skill positions, against appearance in the weekly stats table:

| Friday designation | n | P(did not play) |
|---|---|---|
| *(on report, no designation)* | 942 | **0.149** ← baseline artefact |
| Questionable | 352 | 0.418 |
| Doubtful | 26 | 1.000 |
| Out | 430 | 1.000 |

Two things in the Questionable split are worth more than the headline:

| Questionable, by position | n | P(did not play) |
|---|---|---|
| **QB** | 34 | **0.735** |
| TE | 67 | 0.418 |
| WR | 165 | 0.382 |
| RB | 86 | 0.360 |

| Questionable, by practice | n | P(did not play) |
|---|---|---|
| Did not participate | 43 | 0.512 |
| Limited | 223 | 0.404 |
| **Full participation** | 76 | **0.421** |

1. **A Questionable QB is a different animal.** Roughly twice the absence rate of a
   Questionable skill player. Plausible mechanism: a QB who cannot go is replaced
   outright and takes no snap, while a WR who suits up nearly always records
   *something*. In a 2-QB league that is the position where this matters most, and it
   is also where §9.3 says waiver priority gets spent.
2. **Full practice participation does NOT separate from Limited** — 0.421 against
   0.404. The folk rule ("full practice Friday means he plays") does not hold on this
   data. Small cell (n=76), so suggestive rather than settled, but it is the opposite
   of the assumption a hand-written rule would encode.

**Caveats, because the levels are biased upward.** "Did not appear in the weekly stats
table" conflates a true inactive with a player who dressed and recorded nothing. The
no-designation row's **0.149** is a direct estimate of that bias, so the honest read is
roughly *Questionable ≈ 0.27, Questionable QB ≈ 0.59* once it is subtracted. One
season, and Doubtful has 26 rows. M10b-4 redoes this properly against snap counts
across 2016–2025; it is recorded here because the positional asymmetry is large enough
to survive any plausible correction, and it is the term the Thursday decision turns on.

---

## 3. ARCHITECTURE

### 3.1 The split

```
┌─────────────────────────── always-on box ────────────────────────────┐
│                                                                       │
│  docker compose                                                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  ff-monitor  (one image, one long-running container)          │   │
│  │                                                               │   │
│  │  supercronic ── crontab ──┬─ jobs.waivers    (Tue 08:00 ET)   │   │
│  │                           ├─ jobs.freeagents (Wed 08:00)      │   │
│  │                           ├─ jobs.tick       (every 15 min) ★ │   │
│  │                           │    └─ clock.py fires a lineup     │   │
│  │                           │       checkpoint when a kickoff   │   │
│  │                           │       is 24h / 3h / 75min away    │   │
│  │                           ├─ jobs.injuries   (Sat 10:00)      │   │
│  │                           ├─ jobs.trades     (Mon 09:00)      │   │
│  │                           ├─ jobs.refresh    (Tue 05:00)      │   │
│  │                           ├─ jobs.week14     (wk14 Tue)       │   │
│  │                           └─ jobs.heartbeat  (daily 07:00)    │   │
│  │                                                               │   │
│  │  every job:  preflight → build → decide → log → notify        │   │
│  └───────────────────────────────────────────────────────────────┘   │
│         │                    │                    │                  │
│    ./data/cache         ./logs (jsonl)       ./state (dedupe)         │
│    (volume, persists)   (volume)             (volume)                 │
│         │                                                             │
│    .env  (read-only mount, chmod 600, never in the image)             │
└───────────────────────────────────────────────────────────────────────┘
          │                        │                       │
          ▼                        ▼                       ▼
   ESPN v3 (READ ONLY)      nflverse / ffverse      SMTP → my inbox
   Vegas odds, Open-Meteo   Claude API (news layer)
```

### 3.2 Package layout

New code lives under `ff_agent/inseason/`. Nothing outside it changes except the two
modules named in §3.3.

```
ff_agent/inseason/
  clock.py        # week + LOCK CALENDAR; which checkpoint is due; ET; wk 5/14 guards
  state.py        # league state as of week W: rosters, FA pool, waiver order, records
  freeagents.py   # the FA pool, resolved through §0.2 — unresolved REPORTED, never dropped
  ros.py          # rest-of-season projections (F1's resolution)
  weekly.py       # single-week projections: matchup, implied total, weather
  value.py        # roster -> Δ P(title). the common currency for every recommendation
  waivers.py      # §9.3 ordered claim list, P(success), who will clear to Wednesday
  lineup.py       # §5.3 the lineup SEQUENCE — lock state, TNF option value, checkpoints
  trades.py       # §9.4 two-sided search across all eight opponents
  dst.py          # §3.5 + ADD-§F streaming: opponent YARDS, not points
  kicker.py       # §3.5 + ADD-§E
  playoffs.py     # weeks 15-17 view
  week14.py       # §2.2 free-week churn
  news.py         # the LLM layer: news-scout via Claude API, cached, optional
  audit.py        # did last week's recommendations pay off?
  digest.py       # the report object -> HTML + plain text
  backtest.py     # the §11 step 10 gate
  jobs.py         # cron entry points: preflight, run, log, notify
  notify/
    base.py       # Notifier interface + Digest dataclass
    email.py      # SMTP. the only backend shipped
docker/
  Dockerfile  ·  compose.yml  ·  crontab  ·  entrypoint.sh
```

### 3.3 Two existing modules need real changes

Surfacing these now because they are the only places M10b reaches back into shipped code.

1. **`season/simulate.py` must accept completed results.** It currently simulates all
   fourteen weeks from scratch, which is correct preseason and wrong from week 2
   onward — a week-9 P(title) that re-simulates weeks 1–8 is not a forecast, it is a
   different league. Needs `simulate(..., completed=DataFrame[week, team, points, won])`
   that seeds wins/losses from the actual record and simulates only remaining weeks.
   The bracket, seeding rule and bye logic are unchanged.
2. **`season/strength.py::roster_strength` divides `blended_points` by 17.** In-season
   the divisor is *remaining* games, and the numerator is ROS points. Parameterise;
   the preseason call site keeps its current behaviour byte-for-byte.
3. **`season/lineup.py::optimal_lineup` cannot pin a slot.** It assumes every player is
   simultaneously assignable, which stops being true the moment a Thursday player locks
   (**F9**). Needs `pinned={canonical_id: slot}`; see §5.3.

**A gift from the calendar:** M7 needed the P(title) surrogate because it had to score
10,000 rosters per slot. In-season a job scores perhaps 20–40 candidate rosters. So
**M10b can afford the real simulator** and skips the bilinear-interpolation surface
entirely — along with the silent-clamping-at-the-grid-edge bug that cost M7 a
debugging cycle. The surrogate stays available for the trade search's first pass
(§5.4), where the candidate count genuinely is large.

---

## 4. THE CADENCE (§9.1)

★ = urgent tier. All times ET.

| Job | When | What it produces | Notifies |
|---|---|---|---|
| `refresh` | Tue 05:00 | Ingest last week's results; re-fit ROS; re-run the season sim; run the F2 scoring tripwire | only on failure |
| `waivers` | Tue 08:00 | **Ordered** claim list, priority-spend call, P(each claim succeeds), who will clear to Wednesday | always |
| `freeagents` | Wed 08:00 | Post-clear sweep — who actually cleared, grab now at **zero** priority cost. §9.3 says a lot of the edge is here | if anything cleared |
| `injuries` | Sat 10:00 | Friday designations; every Q/D flagged with practice participation | if a starter is Q or worse |
| `lineup` ★ | **schedule-derived** | The lineup sequence — see below and §5.3 | on a change, or a lock inside 3h |
| `trades` | Mon 09:00 | Two-sided search across all eight rosters; opponent roster profiles | if any candidate clears threshold |
| `week14` | wk 14 Tue | §2.2 free-week churn — playoff-only value, zero downside | always |
| `heartbeat` | daily 07:00 | Cookies valid · cache fresh · last successful run of every job | **only when something is wrong** |

### The lineup job has no fixed time, because the NFL has no fixed schedule

Per **F9**, kickoffs are not weekly-periodic — 2026 opens on a **Wednesday**, my
semifinal has **three Christmas Day games**, my quarterfinal has two Saturday games,
and Sunday's late slate is two windows twenty minutes apart. So `lineup` is not a cron
time. The crontab runs a cheap **tick every 15 minutes**, and `clock.py` fires a
checkpoint only when one is actually due:

| Checkpoint | Fires at | Emails |
|---|---|---|
| **Advisory** | 24h before the week's **first** lock — once per week | Always. The full lineup, every window, one message |
| **Confirm** | 3h before **each** lock window | Only if that window's call changed since the last email |
| **Inactives** ★ | **kickoff − 75 min**, each window | Only if a starter in that window is out or downgraded |
| **Monday close** | 3h before the last Monday kick | Only when two Monday-eligible players make a swap possible |

**Checkpoints are not emails.** Nine checkpoints a week that each sent a message would
be precisely the fatigue §6.4 warns about, and the digest would stop being read by
October. Only the weekly advisory is unconditional; every later checkpoint is a silent
re-check that speaks **only when the answer moved**. A typical week is one email plus
zero or one more; a bad week is four, and every one of them is load-bearing.

A tick with nothing due exits in well under a second and sends nothing. On 2026 week 15
that schedules checkpoints around the Thursday game, both Saturday kicks, all four
Sunday windows and Monday night — nine or so — against §9.1's four fixed slots, and
without a special case for the Wednesday opener, Christmas, London or flex scheduling.

**Two schedule facts specific to me.** My season ends in **week 13** (§2.2), so waiver
aggression front-loads and the `trades` job escalates its urgency through weeks 11–13,
then flips entirely to weeks 15–17 value in week 14. And **weeks 5 and 14 are my
fantasy byes** — the lineup tick must no-op those weeks, loudly, because §10 lists "a
lineup being set for week 5 or week 14" as an alarm meaning something is broken.

---

## 5. THE ENGINES

### 5.1 Rest-of-season projections (`ros.py`)

Per F1: anchor on ESPN's per-week projected points recomputed through the M2 engine,
blend our model on top at a fitted weight.

Our side of the blend re-fits M3's opportunity model on **current-season** volume,
with two in-season adjustments:

- **Volume over efficiency, harder than preseason.** ADD-§B measured 0.584 stability
  for volume against 0.292 for efficiency, on full seasons. Three weeks of efficiency
  is noise wearing a decimal point. Weight 3-game rolling target share, carry share,
  route %, snap % — and give yards-per-touch nothing.
- **Shrink toward the preseason projection by games played.** A week-3 usage rate on
  three games is a wide band; a week-11 rate is not. This is the in-season analogue of
  M5's 30-pick pseudo-count, and it is the only thing that stops the week-3 digest
  from being a list of people who had one good game.

**Gate:** walk-forward on 2025 weeks 4–13. Does the blend beat the ROS anchor alone on
rank correlation with actual rest-of-season points? M3's precedent says the honest
answer may be no. **If it is no, ship the anchor and record it** — that is ADD-§H's
seventh entry, found the same way M3b's was.

### 5.2 Weekly projections (`weekly.py`)

ROS per-game rate × matchup, in §9.2's stated order of importance: implied team total ·
volume trend · opponent strength by position via **EPA/success rate allowed** (not raw
fantasy points allowed, which is schedule-contaminated) · game script · weather ·
injury designation and practice participation. For QBs, **projected sacks taken** — §9.2
notes an elite rush against a bad line costs 5–6 points here before anything else
happens.

Floor-versus-ceiling posture comes from the season sim, per §9.2: heavy favourite →
protect the floor; big underdog → buy variance. Layered on standings context, and the
crossover is already measured — M7 found title odds favour variance below a roster
delta of **+15.9** and against it above, with the top-2 crossover arriving first at
**+12.1**. §2.4 states "roster variance is good" without qualification; it is true
only below that line, and the monitor should know which side of it I am on.

### 5.3 The lineup sequence (`lineup.py`) — Thursday, the gap, and Sunday

Per **F9**, a lineup is not a decision, it is a schedule of decisions. This module owns
that schedule.

#### The state, not the answer

One `LineupState` per week. Every rostered player carries a `lock_at` timestamp — from
ESPN's own per-player `game_date`, which is authoritative because ESPN is what enforces
the lock, cross-checked against nflverse `gameday + gametime`. A disagreement between
the two is an **alarm, never an average**: it means one of them has the wrong game.

Each slot is then in exactly one of three states, and the emails never re-litigate the
first one:

| State | Meaning | What the digest does with it |
|---|---|---|
| `locked` | Kickoff has passed | Shows it greyed, with actual points once final. No advice |
| `open` | Kickoff is in the future | Advises; shows the deadline |
| `at_risk` | Open, but the recommended player is Q / D / trending out | Advises, and flags the fallback and the check time |

#### The Thursday decision is not "who is better"

Starting a Thursday player is **option-destroying**. Once he is in, that slot cannot
respond to anything that happens Friday, Saturday or Sunday morning. So the bar is not
*"is p better than the best alternative I can see today"* — it is:

```
start Thursday player p  iff
    E[p]  >  E[ best Sunday alternative, chosen under SUNDAY information ]
```

which is a strictly higher bar, because the Sunday choice gets to be made knowing
things Thursday does not. Computed with the same force-and-simulate pattern M9 used at
the draft table, and for the same reason — the counterfactual is what decides it:

```
for each candidate Thursday commitment c:                # who goes in which slot
    draw N Sunday information states:
        availability ~ F10's P(plays | designation, practice, position)
        plus a modest continuous revision term on those who play
    for each state:  solve optimal_lineup(remaining players, pinned=c)
    value(c) = mean total
choose argmax; report the gap to the runner-up
```

#### Two effects that partly cancel, and both get measured rather than asserted

1. **Option cost.** Locking a slot forfeits the ability to react on that slot. This is
   the effect everyone knows about, and it argues against Thursday starts.
2. **Information value, which cuts the other way and is usually missed.** A Thursday
   player who *starts* tells me my partial score three days early, which sharpens
   Sunday's floor-versus-ceiling posture on every remaining slot (§9.2, and M7's
   measured variance crossover at a roster delta of **+15.9**). Benching him does not
   preserve any flexibility — he is locked out either way — so this is a real argument
   *for* the Thursday start that the naive "never lock in early" instinct discards.

When the two players are close the option cost usually dominates, but the penalty is
smaller than "never lock early" implies. The simulation above captures both without
either being hand-tuned: pinning `c` removes flexibility, and the Sunday solve happens
under a drawn information state. **Whether the machinery beats naively starting the
higher projection is the M10b-4 gate**, not an assumption.

#### The gap between Thursday and Sunday

Between the Thursday lock and the Sunday slate, three things change: Friday's official
game-status designations, Saturday's practice reports and beat news, and Sunday's
inactive lists. The weekly advisory covers the whole lineup once; after that a
checkpoint only speaks when the answer moved, and when it does it reports only:

- **what changed since the last one** — never the whole lineup again;
- **what is still open**, with the current call and the runner-up;
- **what is already locked**, greyed out, with actual points when final;
- **the next deadline**, always, in one line at the top.

#### The Sunday sequence, derived from kickoffs

Inactives drop ninety minutes before each kickoff. So the checkpoint is
**kickoff − 75 minutes for every distinct window containing one of my starters**, not a
fixed hour. On 2026 week 15 that means checks before 17:00 and 20:20 Saturday, before
13:00, 16:05, 16:25 and 20:20 Sunday, and before 20:15 Monday.

§9.1's single "Sun ~11:15" is *fifteen minutes early* for the official 1pm-slate drop,
covers none of the late slate, and misses a 09:30 London game by two hours. It
described one slate, and the week has six.

#### Monday night is the only decision made against a known number

By Monday evening my opponent's score is final or nearly so. If a starter and a bench
player are both on Monday night, that swap is decided under **certainty about the
target** rather than a projection — the one place all week where §9.2's floor-versus-
ceiling rule needs no forecast, and where §2.4's "chase variance when behind" is either
exactly right or exactly wrong with nothing in between. Narrow (it needs two
Monday-eligible players) but free to support once the state machine exists.

#### What this needs that does not exist yet

1. **`optimal_lineup(players, pinned={canonical_id: slot})`.** The solver assumes every
   player is simultaneously assignable. Once Thursday locks, the rest must be solved
   *around* a fixed partial assignment. The existing optimality argument survives —
   pinning only removes players and slots from the free problem, and the FLEX still
   accepts a superset of what the strict slots do — but it gets the same brute-force
   assertion test on random rosters that `optimal_lineup` already has.
2. **The availability model** from **F10**, redone against snap counts over 2016–2025
   rather than one season of a stats-table proxy.
3. **Projection snapshots on every run.** ESPN does not retain projection *history* —
   a past week's projected points is whatever it last was — so the revision
   distribution cannot be reconstructed backwards. Snapshotting our own inputs every
   run costs nothing and is the only way this is ever measurable properly. By 2027
   there is a real dataset; until then the availability event carries the model, which
   is honest because the binary out/in event dominates the continuous revision anyway.

#### Guardrails specific to a lineup that arrives by email

- **Weeks 5 and 14: no lineup, ever.** §10 lists a lineup set in either as a sign
  something is broken. The job asserts and no-ops.
- **A locked slot cannot be advised on.** The state machine makes it unrepresentable
  rather than merely discouraged.
- **A recommendation that arrives after its deadline is worse than none.** Every email
  leads with the deadline, and the job refuses to send a lineup change with under
  twenty minutes of runway — it escalates the subject line instead.
- **Clock skew breaks all of this silently.** If the container clock drifts, every lock
  time is wrong and nothing looks wrong. The heartbeat checks it.

#### What the Sunday email looks like

```
Subject: [FDS URGENT] wk 7 — Nacua OUT. 1 swap, deadline 12:58

  NEXT LOCK  Sun 13:00 ET, in 74 min  ·  4 slots open, 3 locked

  ▸ SWAP   WR2   Puka Nacua (OUT, 11:31 inactives)  →  Jayden Reed
           +4.1 pts · +0.6% title · Reed is the only WR with a 13:00 kick
           and a route share above 70%

    HOLD   RB2   Bucky Irving (Q, limited Fri)  — plays; F10 puts a Q RB at
           ~36% out, and the fallback loses 5.2 pts. Re-checked at 12:45.

  LOCKED   QB1 Allen 24.8 · QB2 Maye 19.1 · TE Bowers (Thu, 14.2)
```

### 5.4 The waiver engine (`waivers.py`) — the core deliverable

§9.3's rule, made concrete:

```
marginal_value(add p, drop d)
    = Σ over my REMAINING play weeks w:
        optimal_lineup_points(roster ∪ {p} \ {d}, w) − optimal_lineup_points(roster, w)
    → converted to Δ P(title) by re-running the season simulator

claim if  marginal_value > option_value(current priority, weeks remaining)
```

Six things fall out of writing it this way:

1. **"Bench upgrades ≈ 0" needs no heuristic.** Value is measured *through* the lineup
   solver, so a player who never starts contributes exactly zero. §3.2's "don't hoard
   handcuffs" is enforced by the arithmetic rather than by a rule someone has to
   remember.
2. **Add and drop are ONE action, scored jointly.** Enumerate (p, d) pairs; the dropped
   player's own remaining starts are the cost. Constrained by the IR slot, §1's
   position maxima, and starter feasibility — the same `allowed_positions` machinery
   M7 needed when the unconstrained policy drafted eight QBs and no tight end.
3. **§2.1's free-bye arbitrage is a WEEKLY edge, not just a draft-board term.** My
   remaining play weeks exclude 5 and 14. A player whose NFL bye lands there costs me
   literally nothing and costs the other five teams a start — and nobody's rankings
   price it, because it is a property of my schedule. On the board this was worth a
   measured 1.79 points and M4 called it "tiebreaker-sized"; on a *waiver* decision
   between two similar players it is often the whole margin.
4. **QB claims are special, twice over.** §9.3: any startable QB on the wire is worth
   top priority in a 2-QB league. §3.3: this league charges −1 per sack and **no
   public ranking prices it**. M3b measured sacks-over-expected persisting at **0.434**
   against TD-over-expected's 0.104 — four times the carry-forward — so the board's
   `sack_adjustment = −0.434 × prior_sacks_over_expected` term applies to a waiver QB
   exactly as it applies to a drafted one, updated with current-season sacks. This is
   the single most durable unpriced edge in the league and it is available every week.
5. **P(claim succeeds) is modelled, and it is calibratable.** ESPN exposes every team's
   `waiver_rank`; `postdraft/teams.py::_starter_holes` already computes who has a hole
   at which slot; `Team.acquisitions` and the transaction log say who actually bothers.
   Teams ahead of me with a matching hole and a history of claiming → P(he survives to
   me). Calibrated against 2025's `WAIVER_ERROR` rows (F3), which are the observed
   failures. Note a low P(success) is **not** a reason to skip a claim: a failed claim
   costs nothing. It is a reason to order the list differently.
6. **The list is ORDERED, and ordering is the deliverable.** ESPN processes claims in
   my order and priority drops only on a *success*, so a single claim is strictly
   worse than a ranked list. Put the player worth burning priority on first, then
   cheap fallbacks.

**`option_value` is measured, not guessed.** §9.3 argues nine teams makes priority
cheap — the queue is only nine long and you climb back fast. That is a testable claim:
the 2025 transaction log says how fast waiver rank actually recovered in this league.
Combined with an empirical distribution of "best available add per week" from the same
replay, `option_value` becomes an expectation rather than a vibe. If the league turns
out to be inactive, priority is nearly free and the correct policy is to claim almost
anything above threshold — which is §9.3's own conclusion, arrived at with a number.

**And the thing to say out loud, every week:** §9.3 asks the tool to distinguish
*"claim this one"* from *"this one will clear — just grab him Wednesday."* In a
nine-team league a lot of useful players clear. Predicted-to-clear players go in their
own section with zero priority cost attached, and Wednesday's job checks whether the
prediction held.

**D/ST and K get their own modules** (`dst.py`, `kicker.py`) because ADD-§I-3e says
their drivers are genuinely different, and because of two league-specific traps:

- **The D/ST trap worth encoding** (ADD-§F): a defence facing a high-volume, methodical
  offence that stalls in the red zone allows few points and 400+ yards, and posts a
  **negative** score here while looking fine in every other league's box score.
  Bucketed yards-allowed runs +5 to −7 and continuous yards are off, so the projection
  must be of opponent **total yards** — driven by pace (seconds per play), plays per
  game and yards per play — not opponent scoring. The ideal stream is a fast,
  three-and-out-prone opponent.
- **The K trap:** M3 found ESPN's kicker projection **merges 50-59 and 60+ into one
  `50Plus` bucket**, while §1 pays 5 and 6 respectively. ESPN's own projection
  therefore cannot price the exact distinction this league pays for, which means our
  own kicker projection is required rather than optional.

Both stream weekly (§3.5), and neither is ever recommended before the last two rounds
of anything — M10a already recorded what happens when that guard is lifted
("KICKER WORSHIP": Texans D/ST + Brandon Aubrey came back as the better pair at nine
consecutive picks).

### 5.5 The trade finder (`trades.py`) — "analyse other people's teams"

Runs weekly and profiles all eight opponents whether or not it finds a trade.

**Roster profiling.** For each manager: ROS-projected optimal lineup, starting-slot
surplus and deficit, bye exposure over *their* play weeks, and playoff-week (15–17)
strength. Surplus is defined in starting terms — a third good RB on a team starting
2 RB + 1 FLEX is surplus; a second QB hole in a 2-QB league is acute. This is the same
`_starter_holes` machinery the waiver P(success) model uses, so the two agree by
construction.

**Search, in two stages** (the M7 surrogate pattern, applied where it belongs):

1. Enumerate 1-for-1, 2-for-1 and 1-for-2 packages within a value band, constrained by
   position maxima and the 17-man roster. Roughly 1,800 one-for-ones and ~1,500
   two-for-ones per opponent — far too many for the real simulator, exactly right for
   a fast lineup-delta screen.
2. Take the top ~20 and run the **real season simulator** on both sides for Δ P(title).

**The acceptance model is where the edge actually lives.** Score their side under
*their* value function — public consensus, `wsf`/`rp` — and my side under *mine*: our
blended ROS carrying the sack term, the free-bye term, 0.05 per carry and 6-point
passing TDs. **Propose only trades positive under both.** The edge is precisely the
wedge between the two value functions, and every proposal must **name the wedge**:
*"this works because our board prices sacks at −1 and consensus prices them at zero,"*
not *"our model likes him."* A trade nobody would accept is not a recommendation, and
a trade whose rationale can't be stated in one sentence is probably an artefact.

**§2.3 is priced automatically and flagged anyway.** Strengthening one of the four
double-up managers costs me twice, because their higher mean runs through two of my
twelve games instead of one. Running both sides through the season sim captures that
without a special case — but the digest says it in words too, because it changes
whether I should send the offer at all.

**§9.4's structural bet:** depth is cheap in a nine-team league with a rich wire, so
consolidating two mid pieces into one stud is usually favourable. That also falls out
of the lineup solver rather than needing a rule — two bench-quality players contribute
zero, one starter contributes.

From roughly week 9, weeks 15–17 schedule strength and December rest-risk enter the
score. Output is a **copy-pasteable message plus the ESPN trade URL**. I send it. §0.1.

### 5.6 Standings and the common currency (`value.py`)

Every recommendation in every job reports Δ P(championship) (§8, §2.4). One module owns
`roster → (weekly mean, uncertainty) → P(title)` so the waiver engine, the trade finder
and the lineup solver cannot drift apart.

The weekly sim refresh reports P(playoffs), P(top-2 seed), P(title) — **with M6's
measured ceiling printed beside them**, the way M10a's standings forecast does.
Preseason projections scored Spearman **−0.21** on the 2025 standings task; a *perfect*
simulator scores **+0.52**; perfect player knowledge **+0.43**. The probabilities are
the honest output and the finishing order is decoration. In-season the projections get
better every week as actual results replace forecasts, so the band should visibly
narrow — and if it doesn't, that is itself worth knowing. ESPN's own `playoff_pct`
(F5) sits in the same table as an independent check.

### 5.7 Playoffs and `/week14`

From ~week 10 a **playoff view** appears in every digest: roster strength scored over
weeks 15–17 only, with December weather exposure and "team likely to rest starters
once seeding is clinched" flagged (§2.4).

**`/week14` is a different job, not a louder waiver run** (§2.2). Week 14 is my bye:
no lineup to set, no game to lose, my record already locked. So every drop is free and
every add is a pure playoff bet. Rank purely on weeks 15–17 value, ignore week 14
entirely, absorb risk I would never take in-season, and drop anyone who cannot help in
the bracket. The §10 alarm that fires on "a lineup being set for week 14" stays armed
throughout — this job explicitly does not set one.

### 5.8 The news layer (`news.py`) — the only LLM in the loop

A Claude API call with web search, scoped to the `.claude/agents/news-scout.md` brief
already in the repo: depth-chart changes, snap/route/carry inflections, injury
designations with practice participation, coordinator changes, returning players who
reclaim vacated volume. **Facts and sources, never a start/sit verdict** — the
optimizer decides. Beat-reporter speculation labelled as such.

Three hard rules:

- **It never touches a number.** It annotates candidates the deterministic engines
  already ranked. If the LLM layer is unreachable the digest ships with the numbers and
  a line saying the news layer was skipped.
- **It is called on a shortlist, not the pool.** ~15 players a week, not 500.
- **Its output is cached and logged** with the rest of the inputs, so a replay of the
  season sees what the recommendation actually saw.

---

## 6. NOTIFICATION DESIGN

### 6.1 The digest

One `Digest` object per job → HTML and plain text. Self-contained: no external images
or stylesheets, because half of them get blocked and the Sunday one has to be readable
in ninety seconds on a phone.

Every actionable line carries: the action, the Δ P(title), the one-sentence reason
naming the mechanism, and a **deep link into ESPN** so acting is one tap. The waiver
digest additionally carries the claim **order**, because that is the decision (§9.3).

### 6.2 Two tiers, one channel

`[FDS] Tue waivers — 3 claims, top is Ekeler +1.8% title` — routine, threaded by week.
`[FDS URGENT] Sun inactives — Nacua OUT, swap to Dell` — short, actionable, one tap.

### 6.3 The email-only problem, and the fix

Email does not buzz a phone by default. For the Sunday 11:15 job that is the difference
between the tool working and the tool existing. **Fix, once, at setup:** a Gmail filter
on `subject:([FDS URGENT])` → *always mark as important*, plus phone notifications set
to high-priority-only; or add the sender to VIP on iOS Mail. Two minutes, and
`SETUP_MONITOR.md` will walk through it.

The notifier stays behind an interface (`Notifier.send(Digest) -> SendResult`) with
`EmailNotifier` as the only backend shipped. If the Sunday email proves too slow in
practice — and the job logs its own send latency precisely so that is measurable rather
than a matter of opinion — an ntfy or Pushover backend is about thirty lines and no
other file changes.

### 6.4 Silence must mean "nothing to do", never "it died"

- **No-action days send nothing.** A digest that arrives every day saying "no action"
  is a digest that stops being read by week 4.
- **The daily heartbeat only emails when something is wrong** — expired cookies, stale
  cache, a job that has not succeeded in its expected window.
- **Deadman:** if no job has succeeded in 36 hours, that *is* something wrong, and it
  is the one case where the absence of news has to generate news.
- **Dedupe on state.** Each job writes `state/{job}_{season}_wk{N}.json`; re-running
  Tuesday's job twice produces one email, not two. And a recommendation already sent
  is not re-sent unless its Δ has moved materially.

---

## 7. GUARDRAILS FOR AN UNATTENDED AGENT

This is the section that differs most from everything built so far. M1–M10a all ran
with a human watching. This one does not, and it breaks differently.

### 7.1 §0.1 is enforced by test, extended to the container

`test_inseason_package_cannot_write_to_espn` scans `ff_agent/inseason/` for the same
write verbs `ff_agent/live/` is already scanned for. M9 recorded that this rule needed
a *second* guard once a browser was involved; a container that runs unattended on a
schedule is the third place it could break, and the first where nobody would notice.
The container also runs as a non-root user with no ESPN write code in the image at all.

**This is not a phase-two feature.** §0.1 says never, including when the
recommendation is obvious and time is short — which describes the Sunday 11:15 job
exactly. The tool's job is to make the human's ten seconds count, not to replace them.

### 7.2 Cookie expiry is the #1 operational failure mode

A background agent whose cookies expire in week 3 silently stops working until
November. So: every job pre-flights `verify_credentials()`. On failure the job **exits
without sending a digest** and instead emails the re-grab instructions the existing
`ESPNAuthError` already writes. `state/auth.json` tracks the last successful
authentication; the heartbeat escalates daily until it is fixed.

### 7.3 Stale data blocks the digest — it does not decorate it

§10: an optimizer silently running week-3 data in week 8 is worse than no optimizer.
The cache layer already fails loudly; M10b adds the in-season assertion that matters —
**the nflverse weekly data must contain the most recent COMPLETED NFL week**. If it
does not, no digest goes out; an alarm does. New short TTLs for the in-season ESPN
tables, and the F2 scoring tripwire runs every Tuesday.

### 7.4 Every external dependency degrades, none blocks

Odds unavailable → weekly projections run without implied totals, **flagged in the
digest**. Weather unavailable → same. Claude API unavailable → numbers-only digest.
FantasyPros scrape not resuming in September (a real risk — it is a volunteer project,
and F1 shows the in-season lists simply stopped on 2025-12-26) → ESPN's own projections
carry it, which is why F1 chose them as the anchor rather than as a cross-check.
**The only hard dependency is ESPN itself**, and losing that is not a degraded digest,
it is an alarm.

### 7.5 §0.2 is HARDER in-season than it was preseason, and it is the sneakiest risk here

M1 closed the crosswalk gate against five *known, finite* populations — the 2026
draftable pool, the 2025 rosters, three drafts. The in-season free-agent pool is not
finite and not known in advance: it grows every week with practice-squad callups and
UDFAs that nflverse has never issued a `gsis_id` for. §0.2's rule does not bend for
them.

So: an unresolvable free agent is **REPORTED** in its own digest section —
*"3 free agents could not be resolved to a canonical ID and were not evaluated"* — with
names and ESPN ids, and it goes nowhere near a recommendation. Never fuzzy-matched,
never silently dropped, never defaulted. `overrides/player_id_overrides.csv` becomes a
living file with a human decision per row, which is what §0.2 always said it was for.
The failure this prevents is the in-season version of §6's warning: a board that
recommends a retired player, discovered on draft day. Here it would be a Tuesday claim
on somebody whose projection came from a different human being.

### 7.6 Rate limits, secrets, disk

- **Rate limits.** ESPN's v3 endpoints are unofficial and a cron hammering them is how
  a league gets throttled. Per-job request budget, exponential backoff, and the
  existing parquet cache absorbing the repeats.
- **Secrets.** `.env` mounted read-only, never baked into the image, never logged. A
  test scans the rendered digest for the cookie values before any send — the one path
  in this project where a credential could plausibly leave the machine is an outbound
  email, so that is where the check belongs.
- **Disk.** Weekly pbp across eighteen weeks is not small. Prune policy plus a
  heartbeat warning at 80% of budget.

### 7.7 Log everything, then actually read it back

Every job appends to `logs/inseason_{season}.jsonl` in `live/log.py`'s format:
timestamp, job, inputs, candidates considered, what was recommended, the Δ, and the
alternatives rejected. §10: *"or the season teaches you nothing."*

`audit.py` closes the loop weekly — **did last week's recommendations pay off?** Did
the claim I recommended outscore the control? Did the player predicted to clear
actually clear? Was the start/sit call right, and by how much? That is literally §11
step 10's test, run continuously instead of once, and it is the only mechanism that
turns a season of emails into a season of evidence.

---

## 8. GATES AND STAGING

### 8.1 The §11 step 10 gate

> *"Replay several weeks of last season; compare optimizer lineups to what was actually
> started and recommended claims to what actually paid off."*

Replay **2025 weeks 4–13** with no lookahead — state reconstructed as of each Tuesday
morning via F3's `load_roster_week` + `transactions`.

| Arm | Measured against | Control |
|---|---|---|
| **Lineup** | Points gained vs what was actually started (`box_scores[].slot_position`) | ESPN's own projection-optimal lineup |
| **Thursday call** | Points gained by the option-value-aware commitment | **Naively starting the higher projection** |
| **Waivers** | ROS points added **to the starting lineup** by the top claim | **The most-added player across the league that week** (F4) |
| **P(claim succeeds)** | Predicted vs observed | — calibration only, on `WAIVER_ERROR` rows |
| **Trades** | Positive-sum under both value functions; not bad in hindsight | — weak arm, stated as weak |

**The controls are the point, not decoration.** M7's `best_consensus` arm found that a
policy with *zero* board edge captured +29.9 weekly points against the full model's
+32.1 — the board contributed 2.2 and the rest was bought from the opponents' spread.
Without a control we would have banked the whole 32 as skill. If our claim list cannot
beat "take whoever the league is adding", **we ship the control and say so.**

**Caveats stated up front, in the M5/M6 tradition:**

- 2025 was an **8-team** league; 2026 is 9. Waiver depth scales with roster slots, so
  the replay's pool is *shallower* than 2026's will be — the replay therefore
  **understates** how rich the wire is (§3.2). Direction safe, level not.
- 2025 *was* 2-QB and *did* use the post-2024 ruleset, so format and scoring are
  matched. That is a better starting position than M5 had.
- There is exactly **one** season of 2-QB in-season history. Same thin evidence M5 and
  M8 wrestled with; the honest response is wide bands and a stated preference for the
  control when the margin is inside them.
- The trade arm has essentially no ground truth. Few trades happen in any league. It is
  a plausibility check, not a gate, and is labelled that way.

### 8.2 Staging — six shippable stages, each with its own gate

| Stage | Ships | Gate |
|---|---|---|
| **M10b-1** Infrastructure | Container, scheduler, notifier, heartbeat, pre-flight, logging | A job runs on schedule in the container and emails; a deliberately broken cookie produces the fix-it email and no digest; a deliberately stale cache **blocks** the send; `TZ` asserted; §0.1 scan passes |
| **M10b-2** ROS projections | `ros.py` | Walk-forward on 2025 wk 4–13 beats the anchor — **or is recorded as not beating it**, and the anchor ships |
| **M10b-3** Waiver engine | `waivers.py`, `freeagents.py`, `dst.py`, `kicker.py` | Beats the most-added control on the 2025 replay; P(success) calibrated on `WAIVER_ERROR`; unresolvable FAs reported, never dropped |
| **M10b-4** Lineup sequence | `lineup.py`, `weekly.py`, `clock.py`'s lock calendar | Beats what was actually started **and** ESPN's projection-optimal lineup; **the option-value-aware Thursday call beats naively starting the higher projection — or it is dropped**; every 2025 lock window reproduced from the schedule; wk 5/14 no-op asserted |
| **M10b-5** Trades | `trades.py` | Every proposal positive-sum under both value functions; each names its wedge; §2.3 penalty visible |
| **M10b-6** Playoffs + `/week14` | `playoffs.py`, `week14.py` | Playoff view scores weeks 15–17 only; `/week14` recommends no lineup and prices week 14 at zero |

M10b-1 is worth shipping alone and early. A container that reliably emails "cookies
still valid, nothing to do" every Tuesday is a real asset — it proves the plumbing
before the season starts, when fixing it is free.

---

## 9. WHAT I AM DELIBERATELY NOT BUILDING

ADD-§H lists five things; M3b's xFP added a sixth. In the same spirit, and for the same
reason — saving the time:

- **Auto-submit of anything.** §0.1. Not a later phase, not behind a flag, not "when
  it's obviously right". Write-capable ESPN endpoints are out of scope for this project
  and code hitting one is a bug.
- **Confident individual injury forecasting.** ADD-§G: a peer-reviewed model in another
  elite sport fell to a C-index near 0.59 after overfitting adjustment. Injury risk
  **widens the variance band**; it never reorders the list.
- **Trade-value charts.** §9.4 is explicit that they all assume 1-QB, 12-team and
  misprice both QBs and depth here. The season simulator is the value chart.
- **Chasing trending adds.** That is the *control* (F4). If we cannot beat it we ship
  it and admit it; we do not dress it up as a model.
- **FAAB logic.** Confirmed `faab = false` — rolling priority, and the two need
  genuinely different mathematics (§9.3: priority is indivisible; you cannot bid small).
- **IDP anything.** `ro`/`rp` arrive polluted with it (F1). Filter it out; never
  approximate a position we do not project. M9 refused IDP leagues outright for the
  same reason.
- **Cross-year D/ST projection from prior-year sacks and turnovers.** ADD-§H: unstable.
  Within-season streaming is fine and is what §3.5 asks for.

---

## 10. RISKS, RANKED

| # | Risk | Mitigation | Residual |
|---|---|---|---|
| 1 | **Cookies expire mid-season, silently** | Pre-flight every job; deadman at 36h; escalating daily email | Low — but only because silence is monitored |
| 2 | **No format-matched ROS consensus** (F1) | Anchor on points not ranks; gate the blend; ship the anchor if the blend loses | Medium — the anchor is ESPN's, so we inherit its errors |
| 3 | **§0.2 in-season: unrostered callups with no `gsis_id`** (§7.5) | Report, never resolve; living override file | Medium — needs a human every few weeks, by design |
| 4 | **FantasyPros scrape does not resume in Sept 2026** | Anchor already does not depend on it; `wsf` degrades to ESPN weekly projections | Low |
| 5 | **ESPN changes its v3 endpoints mid-season** | Schema-hash checks already in the cache layer; loud failure | Medium — unfixable in advance, but never silent |
| 6 | **Notification fatigue** | No-action days send nothing; dedupe on state; Δ threshold | Low |
| 7 | **The recommendations are not actually better than doing nothing clever** | Every arm has a control (§8.1) | **This is the real one.** M7's precedent says most apparent edge isn't real. The gate exists to find that out in August rather than believe it until January |

---

## 11. OPEN QUESTIONS

- [ ] **Does ESPN process my second waiver claim at my OLD or NEW priority within a
      single run?** §9.3's ordering advice assumes claims are processed in my order and
      priority drops only on success, but not what happens to claim #2 in the same run
      after claim #1 wins. Answerable from the 2025 transaction log: find a week where
      I won two claims, or won one and lost the next. Changes the ordering policy.
- [ ] **Does `League.free_agents(week=N)` return the pool as of week N, or today's pool
      with week-N stats attached?** Almost certainly the latter, which is why F3's
      `load_roster_week` reconstruction is the plan of record — but worth five minutes
      to confirm, because if it is the former the backtest gets simpler.
- [ ] **Does ESPN lock a player at kickoff, or at the start of the scoring period?**
      The design assumes per-player kickoff locking, which is ESPN's documented
      behaviour and what `BoxPlayer.game_date` implies. Worth confirming against a
      real Thursday in week 1 before the playoffs depend on it — the whole §5.3
      state machine rests on it, and week 16's Christmas games are where being
      wrong would cost the most.
- [ ] **Which odds source?** The Odds API's free tier (500 requests/month) is ample at
      §10's hourly-max caching, but needs NFL coverage verified. ESPN's own odds are
      already behind cookies we hold. Decide in M10b-4, not before.
- [ ] **How fast does waiver priority actually recover in this league?** Measurable from
      the 2025 transaction log, and `option_value` depends on it. §9.3 asserts "nine
      teams makes priority cheap"; this turns the assertion into a number.
- [ ] **Does the 2026 league keep the 2025 ruleset?** The F2 tripwire answers it in week
      1 rather than asking now. Recorded here because this league has already changed
      scoring once mid-project.

---

## 12. WHAT SUCCESS LOOKS LIKE

By week 3 of the 2026 season: a Tuesday email listing claims in priority order with a
Δ P(title) beside each and a named reason; a Wednesday email saying which of last
night's predictions cleared; a lineup email before each of that week's actual lock
windows, leading with the deadline and covering only what is still open; an inactives
email at kickoff minus 75 **only** when a starter is out; a Monday trade report that
names its wedge; and a daily heartbeat I never see because nothing is wrong. And a
`logs/inseason_2026.jsonl` that, in January, can tell me exactly how much of it was
real.

---

## APPENDIX A — THE CONTAINER, CONCRETELY

Illustrative, not shipped. These are the files M10b-1 creates.

### `docker/Dockerfile`

```dockerfile
FROM python:3.12-slim

# F7: every time in §9.1 is Eastern. A UTC container fires the Sunday
# 11:15 job at 06:15 or 07:15 ET depending on DST.
ENV TZ=America/New_York
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata ca-certificates curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && rm -rf /var/lib/apt/lists/*

# supercronic: cron built for containers — logs to stdout, no PID-1 reaping,
# honours TZ, and does not need a writable /var/spool.
ARG SUPERCRONIC=v0.2.29
RUN curl -fsSLo /usr/local/bin/supercronic \
      https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC}/supercronic-linux-amd64 \
    && chmod +x /usr/local/bin/supercronic

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY ff_agent/ ./ff_agent/
COPY overrides/ ./overrides/
COPY docker/crontab docker/entrypoint.sh ./docker/

# §7.1: nothing in here needs root, and an unattended container least of all.
RUN useradd -m -u 1000 ff && chown -R ff:ff /app
USER ff

ENTRYPOINT ["/app/docker/entrypoint.sh"]
```

### `docker/compose.yml`

```yaml
services:
  ff-monitor:
    build: { context: .., dockerfile: docker/Dockerfile }
    restart: unless-stopped
    environment:
      TZ: America/New_York
      FF_SEASON: "2026"
    env_file:
      - ../.env              # ESPN cookies. chmod 600, gitignored, never in the image
      - ../.env.notify       # SMTP host/user/app-password/to-address
    volumes:
      - ../data/cache:/app/data/cache   # parquet cache persists across runs
      - ../logs:/app/logs               # §10's recommendation log
      - ../state:/app/state             # dedupe + last-success timestamps
    # §7.6: bound the blast radius of a runaway job
    mem_limit: 2g
    cpus: 2.0
```

### `docker/crontab` (supercronic format; times are ET via `TZ`)

```cron
# ── §9.1 weekly cadence ─────────────────────────────────────────────
 0  5 * * 2   uv run python -m ff_agent.cli monitor --job refresh
 0  8 * * 2   uv run python -m ff_agent.cli monitor --job waivers
 0  8 * * 3   uv run python -m ff_agent.cli monitor --job freeagents
 0 10 * * 6   uv run python -m ff_agent.cli monitor --job injuries
 0  9 * * 1   uv run python -m ff_agent.cli monitor --job trades
# ── the lineup has NO fixed time: F9. tick, and let clock.py decide ──
*/15 * * * *  uv run python -m ff_agent.cli monitor --job tick
# ── health ──────────────────────────────────────────────────────────
 0  7 * * *   uv run python -m ff_agent.cli monitor --job heartbeat
```

`week14` is not a cron line either — `clock.py` fires it from the `waivers` slot when
the league week is 14, because it *replaces* that job rather than adding to it (§2.2).

The tick is deliberately dumb and cheap: it reads the cached lock calendar, compares it
to the clock, and exits in well under a second when nothing is due. Everything that
knows about Wednesday openers, Christmas Day kickoffs and London games lives in
`clock.py`, where it is testable, rather than in a crontab, where it is not.

### `docker/entrypoint.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# F7 again, as an assertion rather than a hope. A container whose TZ silently
# reverted to UTC would run every job at the wrong hour and never say so.
uv run python -m ff_agent.cli monitor --job preflight --strict

exec supercronic -passthrough-logs /app/docker/crontab
```

`--job preflight --strict` verifies: `TZ` is `America/New_York`; ESPN cookies
authenticate; the cache is present and not stale; `artifacts/` is readable; SMTP
accepts a connection. **It emails and exits non-zero on any failure**, so a container
that comes up broken says so immediately instead of at 08:00 on Tuesday.

---

## APPENDIX B — NEW CLI SURFACE

Consistent with the existing commands, so nothing here is a new idiom.

```bash
uv run python -m ff_agent.cli monitor --job waivers   # run one job now, print + email
uv run python -m ff_agent.cli monitor --job waivers --dry-run   # print, do NOT email
uv run python -m ff_agent.cli monitor --job preflight --strict  # the entrypoint check

uv run python -m ff_agent.cli waivers        # §9.3 ordered claim list
uv run python -m ff_agent.cli lineup         # the sequence: locked / open / next deadline
uv run python -m ff_agent.cli lineup --at '2026-12-25 12:00'  # replay any decision point
uv run python -m ff_agent.cli trades         # §9.4 two-sided search, all 8 opponents
uv run python -m ff_agent.cli week14         # §2.2 free-week churn
uv run python -m ff_agent.cli audit          # did last week's recommendations pay off?
uv run python -m ff_agent.cli inseason --backtest   # THE M10b GATE — replay 2025 wk 4-13
```

`--dry-run` on every job. The first thing anyone does with a tool that emails is run it
once without emailing.
