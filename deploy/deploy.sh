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
# A full rebuild peaks at roughly (old image + new image + old cache + new cache).
# Measured on this 38G box: build cache alone reached 22.7GB and filled the disk
# to 100%, which killed the running container. So drop the cache outright below
# SAFE_BUILD_GB, and refuse to start a build below MIN_FREE_GB.
MIN_FREE_GB=12
SAFE_BUILD_GB=20

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
# Cap the cache after a successful deploy so it cannot creep back to the 22.7GB
# that filled the disk. Docker 29 renamed --keep-storage to --reserved-space,
# which means space PROTECTED from pruning — the opposite intent, and it silently
# prunes nothing. --max-used-space is the flag that caps; fall back for older daemons.
prune_cache() {
  docker builder prune -f --max-used-space 8GB >/dev/null 2>&1 \
    || docker builder prune -f >/dev/null 2>&1 \
    || true
  docker image prune -f >/dev/null 2>&1 || true
}

cd "$DEPLOY_DIR"

# --- preflight: disk ------------------------------------------------------
# The box runs at ~11GB free with a multi-GB build cache; an unpruned deploy
# loop fills the disk and wedges Docker. Prune before, not after, the build.
# A bounded prune is not enough here: most cache is `Shared: true` with the live
# image and refuses to go. Below SAFE_BUILD_GB, drop the cache entirely — a slow
# rebuild is far cheaper than filling the disk and taking the server down.
if [ "$SKIP_BUILD" = 0 ] && [ "$(free_gb)" -lt "$SAFE_BUILD_GB" ]; then
  log "$(free_gb)G free (< ${SAFE_BUILD_GB}G) — dropping all build cache before building"
  docker builder prune -af >/dev/null 2>&1 || true
  docker image prune -f >/dev/null 2>&1 || true
  log "$(free_gb)G free after prune"
fi
[ "$(free_gb)" -ge "$MIN_FREE_GB" ] \
  || { log "ABORT: only $(free_gb)G free, need ${MIN_FREE_GB}G — free space manually"; exit 1; }

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
