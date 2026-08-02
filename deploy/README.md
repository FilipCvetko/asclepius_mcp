# Deploying Asclepius MCP

Runs on a Hetzner CX23 (`mcp.filipcvetko.com`), Docker Compose behind Caddy,
which handles Let's Encrypt automatically. Migrated off Fly.io in August 2026.

## Deploy a release

```
git tag v3.1.2 && git push origin v3.1.2
```

Then publish a Release for that tag on GitHub. That fires
`.github/workflows/deploy.yml`, which SSHes to the server, runs `deploy.sh`,
and verifies the result. Progress and history live in **Actions** and in the
repo's **Environments → production** tab.

To redeploy without cutting a release: **Actions → Deploy → Run workflow**, and
give it any ref.

## Why the build happens on the server

`data/chromadb/` (2.1 GB) is gitignored, and the Dockerfile does
`COPY data/chromadb/ /seed/chromadb/`. A GitHub runner has no such directory, so
**the image can only be built on the server**. CI orchestrates and verifies; it
never builds. If you ever move the build into CI, you must first solve getting
2.1 GB of Chroma index onto the runner.

## Required GitHub secrets

| Secret | Value |
|---|---|
| `DEPLOY_SSH_KEY` | private key of the server's `deploy` user |
| `DEPLOY_HOST` | `65.109.238.135` |
| `DEPLOY_USER` | `deploy` |
| `DEPLOY_HOST_KEY` | *(optional)* pinned SSH host key; falls back to `ssh-keyscan` |

This repo is **public**. `release` and `workflow_dispatch` cannot be triggered
from a fork, so secrets stay out of untrusted hands — do not add
`pull_request_target` or any fork-reachable trigger to the workflow.

## Manual operations

```bash
ssh deploy@65.109.238.135
cd /opt/asclepius/src/deploy

./deploy.sh --ref v3.1.2      # full deploy
./deploy.sh                   # rebuild current checkout
./deploy.sh --skip-build      # restart only

docker compose logs -f asclepius
docker compose ps
docker stats --no-stream
```

The container is a slim image without `procps` — `ps` and `free` are absent
inside it. Read `/proc` directly if you need process detail.

## Rollback

`deploy.sh` tags the running image `asclepius-mcp:previous` before building and
restores it automatically if the smoke test fails. To roll back by hand:

```bash
docker tag asclepius-mcp:previous asclepius-mcp:local
docker compose up -d --no-build
```

For a specific past release, redeploy its tag — `./deploy.sh --ref v3.1.1`.

## Verification

`smoke_test.sh` is the single definition of "working" and is used by the deploy
script, the workflow, and the monitor.

```bash
./smoke_test.sh http://127.0.0.1:8000              # app directly
./smoke_test.sh https://mcp.filipcvetko.com        # through Caddy/TLS
EXPECT_VERSION=v3.1.2 ./smoke_test.sh https://mcp.filipcvetko.com
```

It does the full MCP handshake, asserts the tool count is exactly 43, and calls
`get_icd10_code`. **`/health` is not a readiness check** — `main.py`'s `_health`
route returns 200 immediately, outside the background init that loads ChromaDB,
so it reports OK while tool calls would still fail. Never gate a deploy on it.

If you add or remove a tool, update the expected count (the `43` default in
`smoke_test.sh`) or every deploy will roll itself back.

## Monitoring

**Deployments** — GitHub Actions run history plus the Environments tab.

**Health** — `asclepius-monitor.timer` runs `monitor.sh` every 10 minutes. It
emails via `notify.py` after **two consecutive** failures (so a normal deploy's
30–60 s of downtime doesn't page you), on recovery, and when the disk crosses
85%. Credentials come from the `SMTP_*` / `NOTIFY_EMAIL` values already in
`deploy/.env`.

```bash
systemctl status asclepius-monitor.timer
systemctl start asclepius-monitor.service    # run a check now
journalctl -u asclepius-monitor.service -n 50
```

**External uptime** — the local monitor cannot report a dead box; it dies with
it. Cover that with a free external check:

1. Sign up at <https://uptimerobot.com>
2. Add Monitor → HTTP(s) → `https://mcp.filipcvetko.com/health`
3. Interval 5 minutes, alert to `filipcvetko123@gmail.com`

That catches total outages; the local monitor catches "up but broken". You want
both, because neither sees the other's failure mode.

## Disk

The box has ~11 GB free and Docker build cache grows fast. `deploy.sh` prunes
before building if free space drops under 8 GB, and prunes again after a
successful deploy. If something goes wrong and the disk fills:

```bash
docker builder prune -f --keep-storage 5GB
docker image prune -f
docker system df
```

## Secrets on the server

`deploy/.env` holds `NOTION_API_KEY`, `NOTION_DATABASE_ID`,
`NOTION_TEMPLATES_DATABASE_ID`, `SMTP_USER`, `SMTP_PASS`. It is gitignored and
excluded from rsync, so deploys never overwrite it. Mode `600`.

To change one: edit the file, then `docker compose up -d` — no rebuild needed.
