"""Rendering a Digest to text and HTML.

Self-contained by necessity: no external images or stylesheets, because half of
them are blocked by default and the Sunday message has to be readable in ninety
seconds on a phone. Every actionable line carries the action, the number, and the
one-sentence reason naming the mechanism.

Every message leads with the DEADLINE. §5.3: a recommendation that arrives after
its deadline is worse than none.
"""

from __future__ import annotations

import html as _html
from datetime import datetime

from ff_agent.inseason.clock import ET
from ff_agent.inseason.notify.base import Digest

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,
Arial,sans-serif;font-size:15px;line-height:1.5;color:#16181d;margin:0;padding:18px}
h1{font-size:17px;margin:0 0 4px}
.deadline{font-weight:600;background:#fdf3e0;border:1px solid #e0b45c;
padding:8px 10px;border-radius:5px;margin:0 0 14px}
.alarm{background:#fbeceb;border:1px solid #d0655c;padding:8px 10px;
border-radius:5px;margin:0 0 10px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#6b7280;
margin:18px 0 6px;border-bottom:1px solid #e5e7eb;padding-bottom:3px}
ul{margin:0 0 10px;padding-left:18px}
li{margin-bottom:5px}
.note{color:#6b7280;font-size:13px}
.foot{color:#9aa1ab;font-size:12px;margin-top:20px;border-top:1px solid #e5e7eb;
padding-top:8px}
""".strip()

FOOTER = (
    "Recommendations only — nothing here has been submitted to ESPN, and nothing "
    "in this tool can (§0.1). Every recommendation is logged with its inputs."
)


def _fmt_deadline(when: datetime | None, now: datetime | None = None) -> str | None:
    if when is None:
        return None
    local = when.astimezone(ET)
    line = f"NEXT LOCK  {local:%a %-m/%-d %H:%M} ET"
    if now is not None:
        mins = int((when - now).total_seconds() // 60)
        line += f", in {mins} min" if mins >= 0 else f" — PASSED {-mins} min ago"
    return line


def to_text(d: Digest, now: datetime | None = None) -> str:
    out = [d.full_subject, "=" * min(len(d.full_subject), 72), ""]
    dl = _fmt_deadline(d.deadline, now)
    if dl:
        out += [dl, ""]
    if d.headline:
        out += [d.headline, ""]
    for a in d.alarms:
        out.append(f"!! {a}")
    if d.alarms:
        out.append("")
    for title, lines in d.sections:
        if not lines:
            continue
        out.append(title.upper())
        out += [f"  - {ln}" for ln in lines]
        out.append("")
    if d.notes:
        out.append("NOTES")
        out += [f"  {n}" for n in d.notes]
        out.append("")
    out.append(FOOTER)
    return "\n".join(out)


def to_html(d: Digest, now: datetime | None = None) -> str:
    e = _html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>{e(d.full_subject)}</h1>",
    ]
    dl = _fmt_deadline(d.deadline, now)
    if dl:
        parts.append(f"<p class='deadline'>{e(dl)}</p>")
    if d.headline:
        parts.append(f"<p>{e(d.headline)}</p>")
    for a in d.alarms:
        parts.append(f"<p class='alarm'>{e(a)}</p>")
    for title, lines in d.sections:
        if not lines:
            continue
        parts.append(f"<h2>{e(title)}</h2><ul>")
        parts += [f"<li>{e(ln)}</li>" for ln in lines]
        parts.append("</ul>")
    if d.notes:
        parts.append("<h2>Notes</h2><ul class='note'>")
        parts += [f"<li>{e(n)}</li>" for n in d.notes]
        parts.append("</ul>")
    parts.append(f"<p class='foot'>{e(FOOTER)}</p></body></html>")
    return "".join(parts)
