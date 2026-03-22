#!/bin/bash
set -e

echo "=== Asclepius MCP Server Starting ==="
echo "Date: $(date)"
echo "Memory: $(free -m 2>/dev/null || echo 'free not available')"
echo "Disk: $(df -h /data 2>/dev/null || echo '/data not mounted')"

VOLUME_DIR="/data"
CHROMADB_DIR="$VOLUME_DIR/chromadb"

# Seed ChromaDB onto volume if not already present
if [ ! -f "$CHROMADB_DIR/chroma.sqlite3" ]; then
    echo "Seeding ChromaDB data onto volume..."
    mkdir -p "$CHROMADB_DIR"
    cp -r /seed/chromadb/* "$CHROMADB_DIR/"
    echo "ChromaDB seed complete. Size: $(du -sh "$CHROMADB_DIR" 2>/dev/null)"
else
    echo "ChromaDB data already present on volume. Size: $(du -sh "$CHROMADB_DIR" 2>/dev/null)"
fi

echo "Disk after seed: $(df -h /data 2>/dev/null || echo 'n/a')"

mkdir -p /app/cache
echo "Starting python main.py..."
exec python main.py
