#!/usr/bin/env bash
# Health monitor — runs the same smoke test the deploy uses.
#
# Alerts only after TWO consecutive failures, so the 30-60s of downtime during
# a normal deploy doesn't page you. Tests the app directly on loopback, so a
# failure here means the app is broken, not Caddy/TLS/DNS.
#
# Blind spot by construction: if the box or its network dies, this cannot send
# anything. That is what the external UptimeRobot check covers — see README.
#
# Installed as a systemd timer, every 10 minutes.
set -uo pipefail

DEPLOY_DIR="${ASCLEPIUS_SRC:-/opt/asclepius/src}/deploy"
STATE=/var/lib/asclepius-monitor
FAILFILE="$STATE/consecutive_failures"
mkdir -p "$STATE"

# SMTP_USER / SMTP_PASS / NOTIFY_EMAIL
set -a; [ -f "$DEPLOY_DIR/.env" ] && . "$DEPLOY_DIR/.env"; set +a

alert() { printf '%s\n' "$2" | python3 "$DEPLOY_DIR/notify.py" "$1"; }

fails=$(cat "$FAILFILE" 2>/dev/null || echo 0)
out=$("$DEPLOY_DIR/smoke_test.sh" http://127.0.0.1:8000 2>&1)
rc=$?

if [ "$rc" -ne 0 ]; then
  fails=$((fails + 1))
  echo "$fails" > "$FAILFILE"
  # Fire once, on the second failure — not on every run thereafter.
  if [ "$fails" -eq 2 ]; then
    alert "[Asclepius MCP] DOWN on $(hostname)" "Smoke test failed twice in a row.

--- smoke output ---
$out

--- containers ---
$(docker ps -a --filter name=asclepius --format '{{.Names}}  {{.Status}}' 2>&1)

--- last 30 log lines ---
$(docker logs --tail 30 asclepius 2>&1)"
  fi
  exit 1
fi

if [ "$fails" -ge 2 ]; then
  alert "[Asclepius MCP] recovered on $(hostname)" "Smoke test passing again.

$out"
fi
echo 0 > "$FAILFILE"

# Disk: warn once per crossing, not every 10 minutes.
used=$(df --output=pcent / | tail -1 | tr -dc '0-9')
if [ "$used" -ge 85 ] && [ ! -f "$STATE/disk_warned" ]; then
  alert "[Asclepius MCP] disk ${used}% full on $(hostname)" "$(df -h /)

$(docker system df)"
  touch "$STATE/disk_warned"
elif [ "$used" -lt 85 ]; then
  rm -f "$STATE/disk_warned"
fi
