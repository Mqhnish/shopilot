# Shopilot — the browser walkthrough, containerised for a persistent host.
#
# NOT the scored artifact. The graded path is `agent.py` plus `src/`, evaluated
# headlessly; this image exists so judges can click a link instead of cloning.
#
# Two things decide the shape of this file:
#
#   * the agent indexes 50,000 products into memory once (~25 s) and then holds
#     multi-turn session state there, so it needs a *persistent process*. It
#     cannot run as a serverless function: consecutive turns would land on
#     different instances and the conversation would reset every message.
#   * the 60 MB catalog is the organizer's data and is not in git, so it is
#     fetched and checksummed at build time by the project's own setup script.

FROM python:3.11-slim

# No build tooling needed: the agent has no dependencies at all.
WORKDIR /app

# Copy the setup script first so the catalog layer caches independently of
# source changes -- editing server.py should not re-download 60 MB.
COPY tools/setup_data.py tools/setup_data.py
COPY data/README.md data/README.md
RUN python3 tools/setup_data.py && python3 tools/setup_data.py --check

COPY . .

# Index at build time would be ideal, but the index lives in memory, so it is
# built on boot instead. Hosts that health-check immediately should be given a
# generous start period -- see docs/hosting.md.
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# A plain healthcheck on the API, so a platform can tell "still indexing" from
# "failed to start".
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/copilot/health',timeout=4).status==200 else 1)"

CMD ["python3", "server.py"]
