# M10b — the in-season monitor.
#
# One image, one long-running container, three volumes. Everything that knows
# about Wednesday openers, Christmas Day kickoffs and London games lives in
# ff_agent/inseason/clock.py, where it is testable — not in the crontab.
FROM python:3.12-slim

# FINDING F7: every time in §9.1 is Eastern and a container defaults to UTC.
# A job written as 11:15 in a UTC container fires at 06:15 or 07:15 ET depending
# on daylight saving — four or five hours before the inactives it exists to read,
# on the one job §9.1 calls the highest-leverage fifteen minutes of the week.
# Asserted at startup by the entrypoint, not merely set here.
ENV TZ=America/New_York
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata ca-certificates curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# supercronic: cron built for containers. Logs to stdout, no PID-1 reaping
# problems, honours TZ, and does not need a writable /var/spool.
ARG SUPERCRONIC_VERSION=v0.2.29
ARG SUPERCRONIC_SHA=9f27ad28c5c57cd133325b2a66bba69ba2235799
RUN curl -fsSLo /usr/local/bin/supercronic \
      "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-amd64" \
    && chmod +x /usr/local/bin/supercronic

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY ff_agent/ ./ff_agent/
COPY overrides/ ./overrides/
COPY artifacts/ ./artifacts/
COPY docker/crontab docker/entrypoint.sh ./docker/
RUN chmod +x ./docker/entrypoint.sh

# §7.1: nothing here needs root, and an unattended container least of all.
RUN useradd -m -u 1000 ff && mkdir -p /app/logs /app/state /app/data/cache \
    && chown -R ff:ff /app
USER ff

ENTRYPOINT ["/app/docker/entrypoint.sh"]
