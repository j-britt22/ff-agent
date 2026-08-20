# Spec Addendum — Predictive Metrics Research
### Append to FANTASY_SPEC.md. Findings ranked by expected impact on *your* league specifically.

---

## A. THE BIG ONE: expected fantasy points (xFP) under your scoring

**What it is.** xFP models what a player *should* have scored from opportunity alone — no talent, no luck. Each play gets an expected catch rate, expected yardage, and expected TD probability, bucketed by air yards, field position, and rush direction. The gap between actual points and xFP is **FPOE** (fantasy points over expected).

**Why it matters.** xFP is substantially more stable and predictive year-to-year than actual fantasy points. FPOE is mostly efficiency and luck, and it regresses. Drafting off last year's *points* means buying a mix of opportunity (sticky) and luck (not).

**Why it's a genuine edge for you.** Public xFP tools all compute under standard or PPR scoring. **You can build xFP under your exact rules from nflverse play-by-play** — including your 0.05/carry, half-PPR reception value, and 6-point passing TDs. Nobody in your league has a version of this metric that matches the game they're actually playing.

Build it: every play in the PBP data → expected outcome from historical rates in that bucket → convert to points using your §1 scoring function → aggregate per player per game. Then rank by xFP/game, and use FPOE as a *separate* regression flag.

### A.1 Expected touchdowns (xTD) — amplified in your league
TDs carry the highest per-event fantasy value but the **lowest year-over-year correlation of any major skill-position metric**. Red-zone TD conversion rate in particular shows minimal season-to-season predictive correlation.

Your league pays **6 points for every TD type**, including passing. That means TD variance — and therefore TD *regression* — is worth more here than in a standard 4-point-pass-TD league. Compute xTD vs. actual TD for every player and treat large gaps as buy/fade signals.

**Caveat, and it's real:** a minority of players persistently beat xTD because they're genuinely better than the league-average outcome in those situations (explosive backs who score from distance, elite red-zone contested-catch WRs). Check a player's multi-year xTD history before flagging regression. One year over expectation is noise; three is a trait.

---

## B. STICKINESS — what actually carries year over year

The core finding across every source: **volume is sticky, efficiency is not.** Per-game rates beat season totals, because season totals are contaminated by games missed.

| Metric | Stability | Use |
|---|---|---|
| **Target share** | ~0.70 correlation — the stickiest skill-position metric | Primary WR/TE input |
| **RB touches per game** | Top RB predictor, correlations approaching 0.60 | Primary RB input |
| QB fantasy points/game | Lowest carryover of the four positions | Weak prior — don't anchor on it |
| **QB sack rate** | ~0.50, and substantially attributable to the QB himself | See B.1 |
| Efficiency / rate stats (YPC, TD rate) | Minimal | Do **not** project forward |

Practical consequence: when you see a player who was inefficient on high volume, that's a buy. Efficient on low volume is a fade. This is the opposite of what last year's points leaderboard tells you.

### B.1 Refinement to what I told you about sacks
I framed sacks taken as mostly an offensive-line property. The research corrects that: **sack rate is roughly 0.50 sticky and is substantially a quarterback stat**, not just a line stat. QBs who hold the ball keep holding the ball.

So the sack projection should weight the QB's own multi-year sack rate *first*, and O-line pressure allowed second. Given your −1/sack, a QB with a chronic 9–10% sack rate is a structurally worse fantasy asset in your league than anywhere else, and no ranking reflects it.

### B.2 Rate stats for WR comparison
Compare receivers on **yards per route run** rather than raw yards — it normalizes for opportunity. Also useful: first downs per route run, separation/open rate, and air yards share. A composite of target share and air yards share (WOPR-style) captures both volume and the *quality* of that volume.

Note the one exception in the research: **yards per game**, while slightly less consistent year-over-year than targets or receptions, was the single most predictive WR stat for next-season fantasy points. Include it alongside the rate stats.

---

## C. VACATED OPPORTUNITY — a preseason-only signal your historical data can't see

Every offseason departure leaves targets, air yards, red-zone looks, and carries behind. This is the main thing a purely historical model misses, and it's available *now*, before the draft.

Compute per team: vacated targets, vacated **air yards**, vacated **red-zone targets**, vacated rush attempts, vacated **goal-line carries**.

**The refinement that matters:** raw vacated targets is too blunt. Isolate vacated volume from *real contributors* only, then **subtract talent added back** (free agents, draft picks, returning injured starters). A team that lost 170 targets and signed two starters has no real vacancy. Net vacated opportunity is the signal; gross is noise.

---

## D. AGE CURVES — quantified, with the nuance that matters

- **RB decline** clusters at **ages 28–30** (~42% of declines) and **league years 6–8** (~46%). RB breakouts are most common in the **rookie year**, with declining probability every year after.
- **WR decline** clusters later, **ages 30–32** (~38%).
- **The cliff is specific to feature backs.** It's real for backs carrying 250+ touches per season; pass-catching and committee backs decline much more gradually. Encode *usage type*, not just age.
- **WR peak age varies by archetype.** Speed-dependent deep threats peak earlier (~24–26); route-precision slot and possession receivers often peak at 28–30. A single "WRs peak at 27" rule destroys real information.
- Elite RB windows are short — a large share of qualifying backs hit their peak in only **one** season. Treat one-year RB breakouts with more skepticism than one-year WR breakouts.

**How this interacts with your league:** with 153 roster spots and a rich waiver wire (§3.2), age-cliff risk is *cheap for you to absorb*. An aging RB who busts costs you a roster spot you can refill from the wire. Weight age as a real term but don't let it override a large opportunity edge the way you would in a 14-team league.

---

## E. KICKER — more predictable than folklore, and your scoring rewards knowing it

This surprised me, and it's directly exploitable given your distance-weighted FG scoring.

**Attempts are nearly everything; accuracy barely matters.** Field goals made and attempted are by far the closest analogs to kicker fantasy points. Efficiency metrics like FG% and distance-adjusted FG-over-expected correlate only weakly.

**Best season-long predictors:** the team's points scored last season and the team's **preseason Vegas win total**. Draft kickers on good offenses with high win totals.

**Best weekly predictor: implied team total.** Kickers whose team's implied total is 27+ hit double-digit fantasy scores at more than double the rate of teams implied at 26 or less.

**Domes matter twice over.** Easier kicking conditions, *and* coaches attempt longer field goals indoors. Given your **FG50 = 5 and FG60 = 6**, dome kickers are worth meaningfully more in your league than in a flat-3-point league.

**The ideal profile:** a team that moves the ball well but stalls before the end zone. Opponent red-zone TD rate allowed is a real weekly input — a defense that's leaky between the 20s but stingy inside the red zone manufactures field goal attempts.

**Missed kicks are somewhat predictive of future missed kicks.** With your −1/miss, fade chronically inaccurate kickers rather than treating misses as pure noise.

**Negative finding — skip it:** 4th-down aggressiveness has *no* measurable impact on kicker scoring, measured multiple ways. Don't build it.

---

## F. D/ST — and your yardage buckets change the model

**Season-long, the most predictive input is the team's Vegas win total**, ahead of prior-year D/ST fantasy points and point differential. Sacks and turnovers correlate strongly with fantasy points *within* a season but are unstable year to year — do not project next year's from last year's.

**Weekly streaming inputs, in rough order:**
1. **Opponent implied total** (from spread + game total)
2. **Opponent QB sack rate taken** — sack-prone QBs are the single best streaming tell
3. Opponent **pressure rate allowed** (O-line quality)
4. EPA per play allowed, 3rd-down conversion rate allowed, opponent scoring rate
5. Home vs. road as a tiebreaker

**What's unique to you:** your scoring has **bucketed yards allowed** running from +5 (under 100) to **−7 (550+)**, with continuous yards-allowed off. Almost every public DST model ignores yardage entirely and projects points allowed plus turnovers. You need an explicit **opponent total-yards projection**, driven by opponent pace (seconds per play), plays per game, and yards per play — not just their scoring.

This creates a specific trap worth encoding: a defense facing a **high-volume, methodical offense that stalls in the red zone** can allow few points but 400+ yards, and post a *negative* score in your league while looking fine in every other league's box score. Conversely a fast, three-and-out-prone opponent is ideal — low plays, low yards, punts.

---

## G. INJURY RISK — model it, but keep the prior humble

Two inputs carry consistently:
1. **Projected workload.** More touches means more exposure. Counterintuitively, models predict *more* games missed for players expected to *play* more — playing football is itself the hazard.
2. **Injury history**, especially soft-tissue recurrence and concussion history, which predicts both incidence and *duration* of future absence.

Per-touch fragility × projected workload gives a games-missed estimate.

**Honest caveat, and this is why it's ranked last:** rigorous prognostic injury models in other elite sports validate poorly — one peer-reviewed soccer model retaining only age and prior-injury frequency dropped to a C-index near 0.59 after overfitting adjustment, barely better than a coin flip. Treat injury risk as a **wide-banded prior that widens the uncertainty distribution**, not a confident forecast that reorders your board. It belongs in the Monte Carlo's variance term, not in the point projection.

---

## H. NEGATIVE FINDINGS — don't build these

Saving you the time:
- 4th-down aggressiveness → kicker scoring: no measurable effect
- Prior-year efficiency/rate stats → next-year production: minimal stability
- Prior-year red-zone TD conversion rate: minimal predictive correlation
- Prior-year DST sacks and turnovers → next-year DST points: unstable
- Confident individual injury forecasting: the science doesn't support it

---

## I. HOW THIS CHANGES THE BUILD

Insert between Milestones 3 and 4 in §11:

**3b. xFP/xTD engine.** Compute expected fantasy points and expected TDs per player-game from nflverse PBP, under your §1 scoring. Test: recomputed xFP correlates with next-season actual points better than prior-season actual points does. That test is the whole justification for the milestone — if it fails, the model is wrong.

**3c. Stickiness-weighted projection blend.** Weight inputs by measured year-over-year stability on *your* data rather than by intuition. Target share and touches/game get heavy weight; efficiency gets near-zero.

**3d. Vacated opportunity table.** Per team, net of talent added back. Preseason only.

**3e. Kicker and D/ST projection modules.** They have genuinely different drivers than skill positions (Vegas win totals, opponent volume, dome status) and shouldn't run through the same pipeline.

Also add to §7.2 projections: `age_archetype` (feature back vs. committee; deep threat vs. slot), `injury_band_width`, and `sack_rate_multi_year` as first-class fields.

---

## J. SUBAGENT DEFINITIONS

Have Claude Code write these to `.claude/agents/`. Each is a markdown file with YAML frontmatter; the body becomes that agent's system prompt. They exist to keep heavy research and data spelunking out of your main context window.

```markdown
---
name: metrics-researcher
description: Researches predictive fantasy metrics and validates whether a proposed
  input actually carries year-over-year signal. Use before adding any new feature
  to the projection model.
tools: Read, Glob, Grep, WebSearch, WebFetch
model: sonnet
---
You evaluate whether a candidate metric deserves a place in the projection model.

For any proposed input, report: (1) measured year-over-year stability, (2) correlation
with next-season fantasy points, (3) whether it is redundant with an input already in
the model, (4) how its value changes under this league's scoring — 2-QB, half-PPR,
6-pt TDs all around, -1 per sack taken, 0.05 per carry, distance-weighted FGs,
bucketed D/ST yards allowed.

Default to recommending AGAINST inclusion. Most metrics are redundant with target
share, touches per game, or implied team total. Say so plainly when that's the case.
Always cite the specific correlation figure or study; never assert stability without
a number.
```

```markdown
---
name: news-scout
description: Gathers injury, depth chart, and role-change news for a named player or
  team. Use for weekly waiver prep and Sunday inactive checks.
tools: WebSearch, WebFetch, Read
model: sonnet
---
You gather current NFL news and return only what changes an opportunity projection.

Report in this order: depth chart changes, snap/route/carry share inflections,
injury designations with practice participation, coordinator or scheme changes,
returning players who will reclaim vacated volume.

Rules: report facts and their source, never a start/sit verdict — the optimizer
decides. Distinguish beat-reporter speculation from confirmed team statements and
label which is which. If a report is single-sourced, say so. Ignore national
hot-take content entirely. Return a compact bulleted summary, never a narrative.
```

```markdown
---
name: opponent-scout
description: Profiles a specific league manager's draft tendencies and current roster
  needs from ESPN league history. Use before the draft and before trade offers.
tools: Read, Glob, Grep, Bash
model: sonnet
---
You profile the eight other managers in this 9-team ESPN league.

For a named manager produce: positional tendency by round, average reach versus
2-QB-adjusted ADP, NFL team bias, how early they took QBs in past drafts, and
current roster holes by starting slot.

Weight these four managers most heavily — they are each faced twice, accounting for
two-thirds of the schedule: Camden Sims (Hodor's Hodors), Kylie Leahy (Personality
Hires), R. Sharrett (Clearing the Fields), Matthew Benca (Gibbs Me My Money).

Output structured data for opponents.json, not prose. Flag explicitly when a
tendency rests on fewer than two drafts of evidence.
```

```markdown
---
name: data-validator
description: Audits the data pipeline for silent corruption — ID join failures,
  stale caches, scoring mismatches. Run after every pipeline change and before the draft.
tools: Read, Glob, Grep, Bash
model: sonnet
---
You hunt for silent data corruption. Assume something is broken and find it.

Check, in order: every rostered and drafted player resolves to exactly one canonical
ID; recomputed historical weekly scores match ESPN's recorded scores exactly,
including sacks taken, rushing attempts, and both D/ST bucket tables; no cached file
is older than its refresh policy; no projection is null, negative, or an extreme
outlier; NFL bye weeks are correctly joined and the weeks 5 and 14 flags are set.

Report failures loudly with the specific rows involved. Never summarize a failure as
a warning — an unmatched player ID is a blocking error, not a note.
```

Register them in `CLAUDE.md` so the main agent knows to delegate: research questions to `metrics-researcher`, weekly news to `news-scout`, league-mate profiling to `opponent-scout`, and a `data-validator` run after every pipeline change.
