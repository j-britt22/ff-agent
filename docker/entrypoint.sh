#!/usr/bin/env bash
# A container that comes up broken says so IMMEDIATELY, rather than at 08:00 on
# Tuesday. Pre-flight emails and exits non-zero on any failure.
set -euo pipefail

echo "ff-monitor starting — $(date '+%Y-%m-%d %H:%M:%S %Z')"

# F7 again, as an assertion rather than a hope. A container whose TZ silently
# reverted to UTC would run every job at the wrong hour and never say so.
if [ "${TZ:-}" != "America/New_York" ]; then
  echo "FATAL: TZ is '${TZ:-unset}', not America/New_York." >&2
  echo "  Every time in §9.1 is Eastern. See finding F7." >&2
  exit 1
fi

uv run python -m ff_agent.cli monitor --job preflight

echo "pre-flight passed. handing over to supercronic."
exec supercronic -passthrough-logs /app/docker/crontab
