#!/usr/bin/env bash
# Verify the MCP server is genuinely serving — not merely up.
#
# /health is useless as a readiness gate: main.py's _health route returns
# {"status":"ok"} immediately, deliberately outside the background init that
# loads ChromaDB. So this does the real MCP handshake and calls a real tool.
#
# Single source of truth for "is it working" — used by deploy.sh, the GitHub
# Actions workflow, and monitor.sh.
#
# Usage:
#   smoke_test.sh <base-url> [expected-tool-count]
#   EXPECT_VERSION=v3.1.2 smoke_test.sh https://mcp.filipcvetko.com
set -euo pipefail

BASE="${1:?usage: smoke_test.sh <base-url> [expected-tool-count]}"
EXPECTED="${2:-43}"
BASE="${BASE%/}"
URL="$BASE/mcp"
CT='Content-Type: application/json'
AC='Accept: application/json, text/event-stream'

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

# 1. initialize -> session id
SID=$(curl -sS --max-time 30 -D - -o /dev/null -X POST "$URL" -H "$CT" -H "$AC" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}' \
  | grep -i '^mcp-session-id:' | tr -d '\r' | awk '{print $2}') \
  || fail "initialize failed against $URL"
[ -n "$SID" ] || fail "no mcp-session-id returned by $URL"

curl -sS --max-time 30 -X POST "$URL" -H "$CT" -H "$AC" -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' >/dev/null \
  || fail "initialized notification failed"

# 2. tools/list -> count must match exactly (catches a half-registered server)
COUNT=$(curl -sS --max-time 60 -X POST "$URL" -H "$CT" -H "$AC" -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | sed -n 's/^data: //p' \
  | python3 -c 'import sys,json; print(len(json.load(sys.stdin)["result"]["tools"]))') \
  || fail "tools/list failed"
[ "$COUNT" = "$EXPECTED" ] || fail "tool count $COUNT != expected $EXPECTED"

# 3. real tool call -> proves the catalogs actually loaded
TEXT=$(curl -sS --max-time 90 -X POST "$URL" -H "$CT" -H "$AC" -H "mcp-session-id: $SID" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_icd10_code","arguments":{"query":"E11"}}}' \
  | sed -n 's/^data: //p' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"]["content"][0]["text"][:300])') \
  || fail "tools/call get_icd10_code failed"
grep -qi 'sladkorna' <<<"$TEXT" || fail "get_icd10_code returned unexpected content: $TEXT"

# 4. optional: assert the deployed release matches what CI just published
if [ -n "${EXPECT_VERSION:-}" ]; then
  V=$(curl -sS --max-time 15 "$BASE/health" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin).get("version","dev"))') \
    || fail "/health unreadable"
  [ "$V" = "$EXPECT_VERSION" ] || fail "deployed version '$V' != expected '$EXPECT_VERSION'"
  echo "SMOKE OK: $COUNT tools, get_icd10_code responded, version $V ($BASE)"
  exit 0
fi

echo "SMOKE OK: $COUNT tools, get_icd10_code responded ($BASE)"
