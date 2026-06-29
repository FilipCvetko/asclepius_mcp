#!/usr/bin/env python3
"""Download the latest ZZZS obračun "Vsi šifranti" XML from partner.zzzs.si.

ZZZS publishes the full current set of billing šifranti as a ZIP-wrapped XML for import
into provider apps ("Šifranti v obliki ZIP/XML"). Each publication ("objava") is keyed by
`leto_objave` (year) + `zap_st_objave` (sequence #); the file is fetched from a public,
no-auth TYPO3 endpoint. We discover the LATEST objava by probing the sequence for the
current year (then fall back a year), download "Vsi šifranti" (vrsta_datoteke=1), and
extract the single XML into data/sifranti/.

The build step (build_sifrant_catalog.py) turns that XML into data/sifrant_catalog.json.
"""
import argparse
import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import requests

BASE = "https://partner.zzzs.si/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Asclepius MCP sifrant fetcher)"}
DATA_DIR = Path(__file__).parent.parent / "data" / "sifranti"


def _params(leto: int, zap: int) -> dict:
    """Download params for the 'Vsi šifranti' XML zip of one objava."""
    return {
        "id": "742",
        "tx_agzzzsapi_zzzs[action]": "downloadfilesifranti",
        "tx_agzzzsapi_zzzs[controller]": "Izvajalci",
        "type": "1249058994",
        "leto_objave": str(leto),
        "zap_st_objave": str(zap),
        "oznaka_sifranta": " ",
        "vrsta_datoteke": "1",              # 1 = Vsi šifranti (full set)
        "naziv_datoteke": "SifrantiXML_Vsi_objava",
        "tip_datoteke": "zip",
    }


def _exists(leto: int, zap: int) -> bool:
    """Lightweight existence check — stream, confirm it's a real zip, don't download it all."""
    try:
        with requests.get(BASE, params=_params(leto, zap), headers=HEADERS,
                          timeout=60, stream=True) as r:
            if r.status_code != 200:
                return False
            head = next(r.iter_content(4), b"")
            return head[:2] == b"PK"
    except requests.RequestException as e:
        print(f"  {leto}/{zap}: request error {e}")
        return False


def _download(leto: int, zap: int):
    """Fully download this objava's 'Vsi šifranti' zip bytes (or None)."""
    r = requests.get(BASE, params=_params(leto, zap), headers=HEADERS, timeout=180)
    if r.status_code != 200 or r.content[:2] != b"PK" or len(r.content) < 10_000:
        return None
    return r.content


def find_latest(leto: int | None = None, max_seq: int = 60):
    """Find the highest existing objava (probe lightly), then download just that one."""
    years = [leto] if leto else [date.today().year, date.today().year - 1]
    for yr in years:
        last = None
        misses = 0
        for zap in range(1, max_seq + 1):
            if _exists(yr, zap):
                print(f"  {yr}/{zap}: exists")
                last, misses = zap, 0
            else:
                misses += 1
                # stop the year once we've seen a run of misses past the last hit
                if last and misses > 3:
                    break
        if last:
            print(f"Downloading {yr}/{last}...")
            content = _download(yr, last)
            if content:
                return yr, last, content
    return None, None, None


def extract_and_save(leto: int, zap: int, zip_bytes: bytes) -> Path:
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    xmls = [n for n in z.namelist() if n.lower().endswith(".xml")]
    if not xmls:
        raise SystemExit(f"No XML inside zip: {z.namelist()}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"vsi_sifranti_{leto}_{zap}.xml"
    out.write_bytes(z.read(xmls[0]))
    # 'latest' pointer + metadata for the build step
    meta = {"leto": leto, "zap_st": zap, "xml_file": out.name,
            "fetched": date.today().isoformat(), "zip_member": xmls[0]}
    (DATA_DIR / "latest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved {out} ({out.stat().st_size:,} bytes); zip member {xmls[0]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leto", type=int, help="force a publish year (else probe current/prev)")
    ap.add_argument("--zap", type=int, help="force a sequence # (requires --leto)")
    args = ap.parse_args()

    if args.leto and args.zap:
        content = _download(args.leto, args.zap)
        if content is None:
            sys.exit(f"No 'Vsi šifranti' for {args.leto}/{args.zap}")
        leto, zap = args.leto, args.zap
    else:
        print("Probing for latest objava...")
        leto, zap, content = find_latest(args.leto)
        if content is None:
            sys.exit("Could not find any 'Vsi šifranti' publication")

    print(f"Latest 'Vsi šifranti': objava {leto}/{zap}")
    extract_and_save(leto, zap, content)


if __name__ == "__main__":
    main()
