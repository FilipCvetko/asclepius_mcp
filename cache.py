"""Generic caching layer for medical data."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuration
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_file(cache_name: str) -> Path:
    """Get the path to a specific cache file."""
    return CACHE_DIR / f"{cache_name}.json"


def is_cache_fresh(cache_name: str, max_age_hours: float = 24) -> bool:
    """Check if cache exists and is within the specified max age."""
    cache_file = get_cache_file(cache_name)
    if not cache_file.exists():
        return False
    age = time.time() - cache_file.stat().st_mtime
    return age <= max_age_hours * 3600


def read_cache(cache_name: str) -> Optional[List[Dict[str, Any]]]:
    """Read data from cache file."""
    try:
        cache_file = get_cache_file(cache_name)
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Cache read failed for {cache_name}: {e}")
    return None


def write_cache(cache_name: str, data: List[Dict[str, Any]]) -> None:
    """Write data to cache file."""
    try:
        cache_file = get_cache_file(cache_name)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Cache write failed for {cache_name}: {e}")
