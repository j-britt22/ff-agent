# Running this for your own ESPN league

This tool was built for one specific 9-team, 2-QB league (that's what
`CLAUDE.md` describes). It now reads your league's settings from ESPN instead,
so it works for a different one.

**It never submits anything to ESPN.** It tells you who to draft; you click in
ESPN's own draft room. That rule is enforced by tests, not by good intentions.

---

## 1. Install

You need [uv](https://docs.astral.sh/uv/) (a Python runner). On a Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the project folder:

```bash
uv sync
```

## 2. Get your ESPN cookies

ESPN has no public API, so the tool signs in as you with two cookies.

1. Open your league on **espn.com** in Chrome and log in.
2. Open DevTools (`Cmd-Opt-I`) → **Application** → **Cookies** → `https://www.espn.com`.
3. Find **`espn_s2`** and **`SWID`**. Copy both values.
4. Your **league id** is in the league URL: `.../leagueId=XXXXXXX`.

Create a file called `.env` in the project folder:

```
ESPN_LEAGUE_ID=1234567
ESPN_S2=paste%20the%20whole%20thing%20here
ESPN_SWID={XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}
```

Keep the `%` characters in `ESPN_S2` and the `{ }` braces on `ESPN_SWID`.

> These cookies **expire**. Re-copy them the morning of your draft.

## 3. Check it works

```bash
uv run python -m ff_agent.cli setup
```

This is read-only. It confirms your cookies, prints your league's shape as it
reads it, and builds your draft board. Do this **days before** your draft, not
an hour before — it needs the network, and it is where you find out something
is wrong.

It should print your team name and your league's starting lineup. If it names
the wrong team, add `--team "Your Team Name"`.

## 4. On draft day

When you learn your draft slot:

```bash
uv run python -m ff_agent.cli gui --auto --slot 4
```

Your browser opens. Then:

- Click **connect to ESPN** to auto-detect picks as they happen. Each detected
  pick shows a **10-second countdown** — it commits itself unless you hit
  **"not right"**.
- Or type any pick yourself. Typing always wins over the feed.
- On your turn you get a ranked shortlist with the cost of each choice and
  whether that player comes back to you.

---

## What this does and does not know

**It reads from your league:** team count, roster slots, every scoring rule,
position limits, playoff format, your bye weeks, and which team is yours.

**Honest limits:**

- **Projections are the weak link, not the draft logic.** The underlying player
  projections beat consensus rankings by a small margin, and the model *alone*
  is worse than consensus. Treat the board as a well-organised second opinion.
- **A first-year league has no opponent history.** The tool falls back to
  everyone drafting off consensus, and says so. This costs little — most of the
  measured edge comes from the board, not from modelling opponents.
- **Keeper leagues are not modelled.** Every player is valued as if available,
  which overvalues anyone already kept. You'll get a warning.
- **IDP and exotic lineup slots are refused outright** rather than silently
  mis-scored — the projections cover QB/RB/WR/TE/K/D-ST only.
- **Auction drafts are not supported.** Snake only.

## If something breaks

Everything still works with the network off once `setup` has run, and the
terminal version is always there as a fallback:

```bash
uv run python -m ff_agent.cli live --slot 4
```
