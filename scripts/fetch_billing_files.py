#!/usr/bin/env python3
"""Fetch + download the CONFIRMED current outpatient billing documents.

Reads data/billing_recrawl.json (listings with dates), selects the newest version per
role, resolves each document's file URLs from its Domino page, downloads them into
data/billing_files/, and writes the final data/billing_sources.json (with local paths)
for the ingest step. Outpatient-only (hospital/SPP descoped).
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_egradiva_index import crawl_document_files  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
RECRAWL = DATA / "billing_recrawl.json"
FILES_DIR = DATA / "billing_files"
OUT = DATA / "billing_sources.json"


def parse_date(s: str):
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except Exception:
            pass
    return datetime.min


# role → (title predicate, how many newest to keep)
ROLES = [
    ("uredba_programi",   lambda t: re.search(r"uredba o programih storitev", t, re.I), 1),
    ("navodilo_obracun",  lambda t: re.search(r"navodilo o beleženju", t, re.I), 1),
    ("standardi_kodiranja", lambda t: re.search(r"standardi kodiranja", t, re.I), 1),
    ("billing_qa",        lambda t: re.search(r"vprašanja in odgovori", t, re.I), 2),
]


def main() -> None:
    listings = json.loads(RECRAWL.read_text(encoding="utf-8"))
    FILES_DIR.mkdir(parents=True, exist_ok=True)

    selected = []
    for role, pred, keep in ROLES:
        items = [d for d in listings if pred(d["title"])]
        items.sort(key=lambda d: parse_date(d["date"]), reverse=True)
        for d in items[:keep]:
            d = {**d, "role": role, "domain": "outpatient"}
            selected.append(d)

    print(f"Selected {len(selected)} current documents:")
    for d in selected:
        print(f"   {d['date']:12} [{d['role']:18}] {d['title'][:60]}")

    # Resolve file URLs + download
    sources = []
    for d in selected:
        files = crawl_document_files({"@unid": d["unid"]})
        print(f"\n{d['title'][:55]} → {len(files)} file(s)")
        local_files = []
        for i, f in enumerate(files):
            url, ftype = f["file_url"], f["file_type"]
            ext = re.search(r"\.([a-z0-9]{2,5})(?:\?|$|%)", url.lower())
            ext = ext.group(1) if ext else ftype
            dest = FILES_DIR / f"{d['unid']}_{i}.{ext}"
            try:
                if not (dest.exists() and dest.stat().st_size > 0):
                    r = requests.get(url, timeout=120)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                local_files.append({"ext": ext, "url": url, "path": str(dest),
                                    "size": dest.stat().st_size})
                print(f"   ✓ .{ext:5} {dest.stat().st_size:>10,} bytes")
                time.sleep(0.3)
            except Exception as e:
                print(f"   ✗ {url[:60]} : {e}")
        sources.append({**d, "files": local_files})

    OUT.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(s["files"]) for s in sources)
    print(f"\nSaved {len(sources)} docs / {total} files → {OUT}")
    # quick ext tally
    exts = {}
    for s in sources:
        for f in s["files"]:
            exts[f["ext"]] = exts.get(f["ext"], 0) + 1
    print("file types:", exts)


if __name__ == "__main__":
    main()
