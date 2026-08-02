#!/usr/bin/env bash
# Deploy Asclepius MCP on the Hetzner host.
#
# The image is built HERE, never in CI: data/chromadb (2.1GB) is gitignored, so
# a GitHub runner has nothing for the Dockerfile's `COPY data/chromadb/` to find.
# CI checks out nothing — it just calls this script over SSH and verifies the result.
#
# Usage:
#   ./deploy.sh                 # rebuild + restart at the current checkout
#   ./deploy.sh --ref v3.1.2    # fetch, check out that ref, rebuild, restart
#   ./deploy.sh --skip-build    # restart only
set -euo pipefail

SRC="${ASCLEPIUS_SRC:-/opt/asclepius/src}"
DEPLOY_DIR="$SRC/deploy"
IMAGE=asclepius-mcp
MIN_FREE_GB=8

REF=""
SKIP_BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ref)        REF="${2:?--ref needs a value}"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[deploy $(date -u +%H:%M:%S)] $*"; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

# Cap the build cache. Docker 29 renamed --keep-storage to --reserved-space,
# which means space PROTECTED from pruning — the opposite intent, and it
# silently prunes nothing. --max-used-space is the flag that caps the cache;
# fall back to a full prune on older daemons that lack it.
prune_cache() {
  docker builder prune -f --max-used-space 5GB 2>/dev/null \
    || docker builder prune -f 2>/dev/null \
    || true
  docker image prune -f >/dev/null 2>&1 || true
}

cd "$DEPLOY_DIR"

# --- preflight: disk ------------------------------------------------------
# The box runs at ~11GB free with a multi-GB build cache; an unpruned deploy
# loop fills the disk and wedges Docker. Prune before, not after, the build.
if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
  log "only $(free_gb)G free — pruning build cache"
  prune_cache
fi
[ "$(free_gb)" -ge "$MIN_FREE_GB" ] \
  || { log "ABORT: only $(free_gb)G free, need ${MIN_FREE_GB}G"; exit 1; }

# --- rollback anchor ------------------------------------------------------
ROLLBACK=0
if docker image inspect "$IMAGE:local" >/dev/null 2>&1; then
  docker tag "$IMAGE:local" "$IMAGE:previous"
  ROLLBACK=1
  log "tagged running image as :previous"
fi

restore() {
  if [ "$ROLLBACK" != 1 ]; then
    log "no :previous image — CANNOT ROLL BACK, server may be down"
    return
  fi
  log "ROLLING BACK to :previous"
  docker tag "$IMAGE:previous" "$IMAGE:local"
  docker compose up -d --no-build || log "rollback restart failed"
}

# --- checkout -------------------------------------------------------------
if [ -n "$REF" ]; then
  log "fetching origin, checking out $REF"
  git -C "$SRC" fetch --tags --prune origin
  git -C "$SRC" checkout --force "$REF"
fi
APP_VERSION="$(git -C "$SRC" describe --tags --always 2>/dev/null || echo dev)"
export APP_VERSION
log "deploying $APP_VERSION"

# --- build + start --------------------------------------------------------
if [ "$SKIP_BUILD" = 0 ]; then
  log "building (10-20 min if requirements.txt changed, ~1 min for code-only)"
  if ! docker compose build; then
    log "BUILD FAILED"
    restore
    exit 1
  fi
fi

docker compose up -d

# --- verify ---------------------------------------------------------------
log "waiting for container health"
for _ in $(seq 1 60); do
  st=$(docker inspect -f '{{.State.Health.Status}}' asclepius 2>/dev/null || echo unknown)
  [ "$st" = healthy ] && break
  sleep 10
done

log "smoke testing (app directly, bypassing Caddy)"
if ! "$DEPLOY_DIR/smoke_test.sh" http://127.0.0.1:8000; then
  log "SMOKE TEST FAILED"
  restore
  exit 1
fi

# --- success --------------------------------------------------------------
prune_cache
log "DEPLOYED $APP_VERSION — $(free_gb)G free"
