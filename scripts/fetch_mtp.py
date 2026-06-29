#!/usr/bin/env python3
"""Download the latest ZZZS MTP "Seznam s šifrantom, MK, postopki" Excel.

This is the authoritative medical-device (medicinsko-tehnični pripomočki) list: per device it
carries the prescribing **medical criteria/indications** (column "BOLEZEN / ZDRAVSTVENO STANJE in
DRUGI POGOJI"), who may prescribe, whether an odločba IOZ is needed, duration, naročilnica rules
and price standard. It is published on the e-gradiva page id=126&detail=DFDC… as dated .xlsx files;
the newest full list (SEZNAM S ŠIFRANTOM,MK,POSTOPKI_<d.m.Y>.xlsx) is what we want.

We scrape the detail page for the $FILE attachment links, pick the newest canonical full-list file,
and download it to data/mtp/. build_mtp_catalog.py then turns it into data/mtp_catalog.json.
"""
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import requests

DETAIL_URL = ("https://www.zzzs.si/?id=126&detail=DFDC914987E44E2AC1257353003EC73A")
HEADERS = {"User-Agent": "Mozilla/5.0 (Asclepius MCP MTP fetcher)"}
DATA_DIR = Path(__file__).parent.parent / "data" / "mtp"

# A canonical full-list filename: SEZNAM S ŠIFRANTOM,MK,POSTOPKI_<d.m.Y>.xlsx — no extra suffix
# (skip "spremembe zdr.", "novi poobl." and .002 partial-change variants).
_HREF_RE = re.compile(r'href=[\'"]([^\'"]+\$FILE/[^\'"]*SEZNAM[^\'"]*\.xlsx)', re.I)
_DATE_RE = re.compile(r"POSTOPKI_(\d{1,2})\.(\d{1,2})\.(\d{4})\.xlsx$", re.I)


def _candidates(html: str):
    """Yield (date_tuple, filename, url) for canonical full-list files on the detail page."""
    for url in _HREF_RE.findall(html):
        fname = unquote(url.rsplit("/", 1)[-1])
        m = _DATE_RE.search(fname)
        if not m:
            continue  # supplementary/partial file → skip
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        yield (y, mo, d), fname, url


def main():
    print("Fetching MTP detail page...")
    r = requests.get(DETAIL_URL, headers=HEADERS, timeout=90)
    r.raise_for_status()
    cands = sorted(set(_candidates(r.text)), reverse=True)
    if not cands:
        sys.exit("No canonical 'Seznam s šifrantom' .xlsx links found on the detail page")
    (y, mo, d), fname, url = cands[0]
    datum = f"{y:04d}-{mo:02d}-{d:02d}"
    print(f"Newest full list: {fname}  ({datum})")

    print(f"Downloading {url}")
    fr = requests.get(url, headers=HEADERS, timeout=180)
    fr.raise_for_status()
    if fr.content[:2] != b"PK" or len(fr.content) < 50_000:
        sys.exit(f"Downloaded file is not a valid xlsx ({len(fr.content)} bytes)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"seznam_mtp_{datum}.xlsx"
    out.write_bytes(fr.content)
    import json
    (DATA_DIR / "latest.json").write_text(json.dumps(
        {"datum": datum, "xlsx_file": out.name, "source_url": url,
         "source_filename": fname, "fetched": date.today().isoformat()},
        ensure_ascii=False, indent=2))
    print(f"Saved {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
