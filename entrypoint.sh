#!/bin/bash
set -e

echo "=== Asclepius MCP Server Starting ==="
echo "Date: $(date)"
echo "Memory: $(free -m 2>/dev/null || echo 'free not available')"
echo "Disk: $(df -h /data 2>/dev/null || echo '/data not mounted')"

# ChromaDB is read from the image (CHROMADB_PATH=/seed/chromadb) — no volume seed
# needed, so first boot is fast. The /data volume is kept only for writable caches.
echo "ChromaDB seed (in image): $(du -sh /seed/chromadb 2>/dev/null | cut -f1)"
mkdir -p /app/cache "${PDF_STORAGE_PATH:-/data/pdfs}"
echo "Starting python main.py..."
exec python main.py
