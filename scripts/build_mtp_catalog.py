#!/usr/bin/env python3
"""Build data/mtp_catalog.json from the ZZZS MTP "Seznam s šifrantom, MK, postopki" Excel.

The `seznam MP` sheet is the authoritative medical-device (medicinsko-tehnični pripomočki) list.
Per device it carries the prescribing rules a clinician needs — most importantly the **indications**
("BOLEZEN / ZDRAVSTVENO STANJE in DRUGI POGOJI"), plus who may prescribe, whether an odločba IOZ is
required, duration, naročilnica rules and the price standard. (The previous version of this script
dropped exactly the indications/naročilnica/price columns — they are the whole point.)

Source = the latest file fetched by scripts/fetch_mtp.py (data/mtp/latest.json); falls back to the
newest inner sheet of the e-gradiva-crawled zip if that's absent. Output is loaded by mtp.py.

The sheet is hierarchical: category/section rows (ŠIFRA holds descriptive text, IME empty) introduce
groups of device rows; we carry the current category path onto each device.
"""
import io
import json
import re
import zipfile
from pathlib import Path

import openpyxl

DATA = Path(__file__).parent.parent / "data"
MTP_DIR = DATA / "mtp"
ZIP_FALLBACK = DATA / "egradiva_files" / "DFDC914987E44E2AC1257353003EC73A.other"
FALLBACK_URL = ("https://www.zzzs.si/?id=126&detail=DFDC914987E44E2AC1257353003EC73A")
OUT = DATA / "mtp_catalog.json"

# Companion structured files (crawled snapshots in egradiva_files):
#  - mutual-exclusion list: devices that may NOT be prescribed together (Pravila OZZ).
#  - timska obravnava: devices requiring a team assessment before prescription.
# Companion files are baked into the image at data/mtp_companions/ (small; rarely change) so the
# periodic refresh can rebuild MTP with exclusions without the dockerignored egradiva_files/.
# Override with MTP_EXCL_FILE / MTP_TIMSKA_FILE to point at fresher copies on the volume.
import os as _os
EXCL_FILE = Path(_os.environ.get("MTP_EXCL_FILE", DATA / "mtp_companions" / "exclusions.xlsx"))
EXCL_URL = ("https://www.zzzs.si/?id=126")  # Zbirka podatkov za MP (e-gradiva)
TIMSKA_FILE = Path(_os.environ.get("MTP_TIMSKA_FILE", DATA / "mtp_companions" / "timska.xlsx"))


def _norm_code(c) -> str:
    """Normalize an MP code so the exclusion list ('131') matches the catalog ('0131')."""
    s = str(c or "").strip()
    return s.zfill(4) if s.isdigit() else s


def load_exclusions():
    """Return {norm_code: [{sifra, naziv}, …]} of mutually-exclusive devices (symmetric)."""
    if not EXCL_FILE.exists():
        print("WARNING: MP exclusion file missing; skipping izkljucuje")
        return {}
    import io
    wb = openpyxl.load_workbook(io.BytesIO(EXCL_FILE.read_bytes()), read_only=True, data_only=True)
    ws = wb["List1"] if "List1" in wb.sheetnames else wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    hdr_i = next((i for i, r in enumerate(rows)
                  if r and str(r[0] or "").strip().upper() == "ŠIFRA MP"), 2)
    pairs, naziv = {}, {}
    for r in rows[hdr_i + 1:]:
        if not r or not str(r[0] or "").strip().isdigit() or len(r) < 4:
            continue
        a, b = _norm_code(r[0]), _norm_code(r[2])
        if a == b:                                   # ignore self-references
            continue
        naziv[a] = _clean(r[1]); naziv[b] = _clean(r[3])
        pairs.setdefault(a, set()).add(b)
        pairs.setdefault(b, set()).add(a)            # symmetric
    return {c: [{"sifra": x, "naziv": naziv.get(x, "")} for x in sorted(s)]
            for c, s in pairs.items()}


def load_timska():
    """Return the set of norm codes requiring timska obravnava (team assessment)."""
    if not TIMSKA_FILE.exists():
        return set()
    import io
    wb = openpyxl.load_workbook(io.BytesIO(TIMSKA_FILE.read_bytes()), read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    return {_norm_code(r[0]) for r in ws.iter_rows(values_only=True)
            if r and str(r[0] or "").strip() and str(r[0]).strip()[0].isdigit()}

# Ordered (header-substring, field). First match wins per column, so specific/disambiguating
# patterns come BEFORE generic ones (e.g. "PODALJŠANJE" before "IZPOSOJA"/"OBNOVLJIVA", and
# "PREDPIS NAROČILNICE" before the bare naročilnica columns).
COLS = [
    ("ŠIFRA", "sifra"),
    ("IME PRIPOMOČKA", "ime"),
    ("PRISTOJNOST", "pristojnost"),
    ("ODLOČBA", "odlocba"),
    ("MATERIALNI STROŠEK", "materialni_strosek"),
    ("PODALJŠANJE", "podaljsanje_izposoje"),
    ("OBNOVLJIVA NAROČILNICA", "obnovljiva_narocilnica"),
    ("PREDPIS NAROČILNICE", "predpis_narocilnice"),
    ("IZPOSOJA", "izposoja"),
    ("POPRAVILA", "popravila"),
    ("PRILAGODITVE", "prilagoditve"),
    ("DOBA TRAJANJA", "doba_trajanja"),
    ("BOLEZEN", "indikacije"),
    ("ZDRAVSTVENO STANJE", "indikacije"),
    ("CENOVNI", "cenovni_standard"),
]


def _clean(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip() if v not in (None, "") else ""


def _load_source():
    """Return (workbook_bytes, source_filename, datum, source_url) for the newest list."""
    latest = MTP_DIR / "latest.json"
    if latest.exists():
        meta = json.loads(latest.read_text())
        xlsx = MTP_DIR / meta["xlsx_file"]
        if xlsx.exists():
            return xlsx.read_bytes(), meta["xlsx_file"], meta["datum"], meta.get("source_url", "")
    # fallback: newest inner xlsx of the crawled zip (stale snapshot)
    print("WARNING: data/mtp/latest.json missing — using bundled zip fallback "
          "(run scripts/fetch_mtp.py for the current file)")
    zf = zipfile.ZipFile(ZIP_FALLBACK)

    def dt(n):
        m = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", n)
        return (int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else (0, 0, 0)
    inner = sorted((n for n in zf.namelist() if n.lower().endswith(".xlsx")), key=dt)[-1]
    y, mo, d = dt(inner)
    return zf.read(inner), inner, f"{y:04d}-{mo:02d}-{d:02d}", FALLBACK_URL


def _map_columns(header):
    field_at = {}
    for idx, cell in enumerate(header):
        h = (str(cell).upper() if cell else "")
        for pat, fld in COLS:
            if pat in h:
                field_at[idx] = fld
                break
    return field_at


def main():
    blob, src_name, datum, src_url = _load_source()
    wb = openpyxl.load_workbook(io.BytesIO(blob), read_only=True, data_only=True)
    ws = wb["seznam MP"] if "seznam MP" in wb.sheetnames else wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))

    hdr_i = next(i for i, r in enumerate(rows)
                 if r and any(str(c).strip().upper() == "ŠIFRA" for c in r if c))
    field_at = _map_columns(rows[hdr_i])

    devices, categories, cat_path = [], 0, []
    for r in rows[hdr_i + 1:]:
        if not r or r[0] is None or not str(r[0]).strip():
            continue
        vals = {field_at[i]: _clean(r[i]) for i in field_at if i < len(r)}
        sifra, ime = vals.get("sifra", ""), vals.get("ime", "")
        has_data = any(vals.get(f) for f in ("ime", "pristojnost", "indikacije", "materialni_strosek"))
        if ime and has_data:                      # a real device row
            devices.append({**vals, "kategorija": "; ".join(cat_path),
                            "source": src_name, "source_url": src_url, "datum": datum})
        else:                                     # a category / section header
            if re.match(r"^\d+\.", sifra):
                cat_path = [sifra]
            elif cat_path:
                cat_path = cat_path[:1] + [sifra]
            else:
                cat_path = [sifra]
            categories += 1

    # Attach mutually-exclusive devices + timska-obravnava flag (companion structured files)
    exclusions = load_exclusions()
    timska = load_timska()
    n_excl = 0
    for d in devices:
        nc = _norm_code(d.get("sifra"))
        if nc in exclusions:
            d["izkljucuje"] = exclusions[nc]
            n_excl += 1
        if nc in timska:
            d["timska_obravnava"] = "DA"

    out = {"source": src_name, "source_url": src_url, "datum": datum,
           "count": len(devices), "fields": [f for _, f in COLS],
           "exclusion_pairs": sum(len(v) for v in exclusions.values()) // 2,
           "devices": devices}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"  exclusions: {n_excl} devices flagged; timska obravnava: "
          f"{sum(1 for d in devices if d.get('timska_obravnava'))} devices")
    with_ind = sum(1 for d in devices if d.get("indikacije"))
    print(f"MTP catalog: {len(devices)} devices ({with_ind} with indications), "
          f"{categories} category rows, datum {datum}")
    print(f"  source: {src_name}")
    print(f"  -> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    for d in devices:
        if "BERGLA" in d.get("ime", "").upper():
            print(f"  e.g. {d['sifra']} {d['ime']}: predpiše={d.get('pristojnost')} | "
                  f"odločba={d.get('odlocba')} | indik={d.get('indikacije','')[:60]}…")
            break


if __name__ == "__main__":
    main()
