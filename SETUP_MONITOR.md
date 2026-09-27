# SETUP — the in-season monitor (M10b)

A container on an always-on box that watches the league on the §9.1 cadence and
**emails** me an ordered list of actions. It recommends; I click. §0.1 is
absolute and asserted by test — there is no write path in the package and there
never will be.

---

## 1. What you need

- A machine that stays awake (Mac mini, Pi, an old laptop, a small VPS).
- Docker and Docker Compose.
- The ESPN cookies already in `.env` (SETUP.md §3).
- An SMTP account. With Gmail this means an **app password**, not the account
  password: Google Account → Security → 2-Step Verification → App passwords.

---

## 2. Configure mail

```bash
cp .env.notify.example .env.notify
chmod 600 .env.notify
$EDITOR .env.notify
```

`.env.notify` is deliberately **separate from `.env`**. The ESPN cookies are a
league login; these are a mail login. Keeping them apart means one leak is not
two.

---

## 3. Make the urgent tier actually buzz

This is the two-minute step that decides whether the tool works.

Email does not buzz a phone by default. For the Sunday inactives job — the
highest-leverage fifteen minutes of the week — that is the difference between
the tool working and the tool merely existing. The monitor prefixes every urgent
message with `[FDS URGENT]`; you need one rule that acts on it.

**Gmail (web):** Settings → Filters → *Create a new filter* → Subject:
`[FDS URGENT]` → *Create filter* → tick **Always mark it as important** and
**Never send it to Spam**. Then on the phone: Gmail app → Settings → your
account → Notifications → **High priority only**.

**iOS Mail:** open a message from the sender → tap the sender name → **Add to
VIP**. VIP mail bypasses notification summaries.

If this proves too slow in practice, the notifier is behind an interface and an
ntfy or Pushover backend is about thirty lines and touches nothing else. The
Sunday job logs its own send latency precisely so that becomes a measurement
rather than an argument.

---

## 4. Bring it up

```bash
docker compose -f docker/compose.yml up -d --build
docker compose -f docker/compose.yml logs -f
```

The entrypoint runs a pre-flight **before** the scheduler starts, and exits
non-zero if anything is wrong: timezone is `America/New_York`, ESPN cookies
authenticate, the cache is present, SMTP accepts the login. A container that
comes up broken says so immediately rather than at 08:00 on Tuesday.

Run it by hand first:

```bash
uv run python -m ff_agent.cli monitor --job preflight
uv run python -m ff_agent.cli monitor --job waivers --dry-run   # render, send nothing
uv run python -m ff_agent.cli lineup                            # this week's lock calendar
```

---

## 5. What arrives, and when

| Job | When (ET) | Emails |
|---|---|---|
| `refresh` | Tue 05:00 | only on failure |
| `waivers` | Tue 08:00 | always — the ordered claim list |
| `freeagents` | Wed 08:00 | if anything cleared |
| `injuries` | Sat 10:00 | if a starter is Q or worse |
| `lineup` | **derived from the schedule** | on a change, or a lock inside 3h |
| `trades` | Mon 09:00 | if a candidate clears the bar |
| `heartbeat` | daily 07:00 | **only when something is wrong** |

**The lineup job has no fixed time, because the NFL has no fixed schedule.**
Week 1 of 2026 opens on a **Wednesday**; week 16 — my semifinal — has three
**Christmas Day** games; week 15 has two Saturday games; Sunday's late slate is
two windows twenty minutes apart. So the crontab runs a fifteen-minute tick and
`clock.py` fires checkpoints at kickoff −24h, −3h and −75min (inactives drop at
−90).

Checkpoints are **not** emails. Only the weekly advisory is unconditional;
everything after it speaks only when the answer moved. A typical week is one
email plus zero or one more.

**Silence means "nothing to do", never "it died."** No-action days send nothing,
the heartbeat speaks only when something is wrong, and if no job has succeeded
in 36 hours the deadman fires — the one case where the absence of news has to
generate news.

---

## 6. When it breaks

**Expired cookies are the number-one failure.** An agent whose cookies expire in
week 3 silently stops working until November — except it does not, because every
job pre-flights authentication and sends the re-grab instructions *instead of* a
digest. Fix `.env`, then:

```bash
docker compose -f docker/compose.yml restart ff-monitor
```

**Stale data blocks the send rather than decorating it.** §10: an optimizer
silently running week-3 data in week 8 is worse than no optimizer. If the weekly
data does not cover the most recently completed week, no digest goes out and an
alarm does.

**Clock skew breaks every lock time silently.** If the container clock drifts,
the calendar is wrong and nothing looks wrong. The heartbeat checks it.

---

## 7. Reading the season back

Every recommendation is logged with its inputs to `logs/inseason_2026.jsonl`.

```bash
uv run python -m ff_agent.cli audit
```

§10 says to log every recommendation "or the season teaches you nothing" — and a
log nobody reads back teaches nothing either. The audit closes the loop weekly:
did the claim beat the control, did the player predicted to clear actually
clear, was the start/sit call right and by how much.

The control is the point. M7 measured a policy with *zero* board edge capturing
+29.9 weekly points against the full model's +32.1. Without a control the whole
thing gets banked as skill.
