# Setup Guide — Fantasy Football Agent

Follow these in order. Steps 1–4 take about 20 minutes. After that you're building.

---

## 0. What you need before starting

- **A paid Claude plan.** Claude Code requires Pro, Max, Team, Enterprise, or Console. The free Claude.ai plan does not include it.
- **A computer** (Mac, Windows, or Linux). Not a phone.
- **Your ESPN league login.** You'll pull two cookies from it in Step 3.

You do **not** need Node.js, Python, or any prior setup. The native installer bundles everything, and Claude Code installs the rest of the project's dependencies itself.

---

## 1. Install Claude Code

Open a terminal:
- **Mac** — press ⌘+Space, type "Terminal", hit Enter
- **Windows** — press the Windows key, type "PowerShell", hit Enter
- **Linux** — you know where it is

Paste one command:

**Mac / Linux / WSL**
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows PowerShell**
```powershell
irm https://claude.ai/install.ps1 | iex
```

Then confirm it worked:
```bash
claude --version
```
You should see something like `2.1.211 (Claude Code)`. If you get "command not found," close the terminal, open a new one, and try again — the installer needs a fresh shell to pick up the path.

Run `claude doctor` if anything looks off. It prints diagnostics without starting a session.

> **Not comfortable in a terminal?** There's a Claude Code desktop app with a graphical interface that does the same thing. Download it from the link on https://code.claude.com/docs/en/desktop-quickstart and skip the terminal entirely. Everything below still applies — you're just clicking instead of typing.

**Windows note:** installing [Git for Windows](https://git-scm.com/downloads/win) is optional but recommended. Without it Claude Code uses PowerShell for shell commands instead of Bash, which works but is slightly more limited.

---

## 2. Log in

Run:
```bash
claude
```
The first time, it opens your browser and asks you to sign in to your Anthropic account. Approve it, and you're in. Type `/exit` to leave the session for now.

---

## 3. Get your ESPN credentials ← the only part Claude Code can't do for you

You need three values. Do this on a desktop browser, not the ESPN app.

**League ID.** Log into your league on the ESPN website. Look at the URL:
```
https://fantasy.espn.com/football/league?leagueId=123456
                                                   ^^^^^^ this
```

**espn_s2 and SWID.** These are auth cookies that let the tool read your private league.

1. On your league page, logged in, right-click anywhere → **Inspect**
2. Go to the **Application** tab (Chrome/Edge) or **Storage** tab (Firefox)
3. In the left sidebar, expand **Cookies** → click `https://fantasy.espn.com`
4. Find the rows named `espn_s2` and `SWID`
5. Copy both values exactly. `SWID` includes the curly braces `{...}` — keep them. `espn_s2` is a long string with `%` characters — keep those too.

There's also an open-source Chrome extension that pulls these for you if the dev-tools route is annoying, but doing it manually takes 30 seconds.

**Two warnings:**
- **These cookies are login credentials.** Treat them like a password. They go in a `.env` file that Claude Code will gitignore — never paste them into a chat, a commit, or a screenshot.
- **They expire.** Plan to re-grab them the morning of your draft and verify the connection *before* you're on the clock.

---

## 4. Create the project

In your terminal:

```bash
mkdir ff-agent
cd ff-agent
```

Now move both spec files into that folder — `FANTASY_SPEC.md` and `SPEC_ADDENDUM_METRICS.md`. Drag them there in Finder/Explorer if that's easier; just make sure they land in `ff-agent`.

Create a file called `.env` in the same folder with your three values:
```
ESPN_LEAGUE_ID=123456
ESPN_S2=paste_the_long_string_here
ESPN_SWID={paste-with-braces}
```

---

## 5. First session

```bash
claude
```

Then paste this as your first message:

> Read FANTASY_SPEC.md and SPEC_ADDENDUM_METRICS.md. Don't write any code yet.
>
> First: ask me about anything ambiguous or missing.
> Second: set up the project — git repo, .gitignore that excludes .env, a CLAUDE.md containing sections 1, 2, and 3 of the spec plus the "never auto-submit" and ID-crosswalk rules, and the four subagent files from section J of the addendum.
> Third: give me a plan for Milestone 1 only (data layer + ID crosswalk). Don't start building until I approve it.

It'll ask questions, scaffold the project, and propose a plan. **Read the plan before approving.** This is the point where you catch a misunderstanding cheaply.

---

## 6. How to work after that

**One milestone per session.** There are ten in §11 of the spec. Start each new session with:

> Read CLAUDE.md. We finished Milestone N. Plan Milestone N+1.

**Verify every milestone's test before moving on.** These are in the spec for a reason — a scoring engine that's silently 3% wrong produces a board that looks completely reasonable and is completely useless. Milestone 2's test (recomputed scores matching ESPN's exactly) is the single most important checkpoint in the build.

**Commit after each milestone.** Ask Claude Code to do it. You'll want to roll back at some point.

**Use plan mode for anything structural.** Just say "plan this first" before the data layer, projection model, or either simulator.

**Delegate research.** Once the subagents exist, say things like "have metrics-researcher evaluate whether air yards share adds anything over target share" — it keeps the heavy reading out of your main session.

---

## 7. Sanity checkpoints

Stop and investigate if any of these happen:

| Symptom | Likely cause |
|---|---|
| Recomputed scores don't exactly match ESPN | Scoring engine wrong — usually sacks taken, rushing attempts, or the D/ST buckets |
| Any player fails to resolve to one ID | The crosswalk is broken. Do not proceed |
| Board looks identical to public rankings | Your scoring isn't being applied — QBs should sit much higher |
| Kicker recommended before round 13 | Roster-need logic is broken |
| Week 5 or 14 shows a lineup to set | Bye-week flags aren't wired in |

---

## 8. If you get stuck

- `claude doctor` — diagnoses install and config problems
- `/help` inside a session — lists commands
- Tell Claude Code the error verbatim. It's better at debugging its own environment than you'd expect.
- ESPN 401 errors almost always mean expired cookies. Re-grab them (Step 3).

---

## Realistic timeline

Milestones 1–3 are a solid weekend. The full build with both simulators is a few weekends. You have unlimited time, so do it properly — but if the draft date moves up on you, the triage order is **1 → 2 → 3 → 9** (data, scoring, board, live draft loop). That alone beats everyone in your league using generic rankings.
