"""Resolve data files, preferring the writable volume over the image-baked copy.

The periodic refresh (refresh.py) writes updated catalogs/manifest to the Fly volume
(ASCLEPIUS_DATA_DIR, e.g. /data/catalogs). Modules should load from there when present so a
refresh is visible without a redeploy; otherwise they fall back to the image-baked data/ copy
(the cold-start baseline). Resolve at *call time* (not import) so a reload after a refresh
picks up the new file.
"""
import os
from pathlib import Path

IMAGE_DATA = Path(__file__).parent / "data"


def data_path(name: str) -> Path:
    """Volume-served file if it exists (ASCLEPIUS_DATA_DIR/<name>), else the baked data/<name>."""
    override = os.environ.get("ASCLEPIUS_DATA_DIR")
    if override:
        p = Path(override) / name
        if p.exists():
            return p
    return IMAGE_DATA / name
