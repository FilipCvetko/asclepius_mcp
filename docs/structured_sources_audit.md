# Audit: structured ZZZS sources missed by the MCP

**Date:** 2026-06-23 · **Source:** `data/egradiva_manifest.json` (3392 crawled e-gradiva docs).

## The general rule (why files are "in" but not usable)

The e-gradiva crawl ingested everything as **prose text for semantic search** (PDF/Word →
`zzzs_egradiva`, 114k chunks). But a whole class of ZZZS sources encodes **operational rules as
structured spreadsheets / zipped data** (`.xlsx`/`.xls`/`.zip`). Those are downloaded and present in
the manifest, yet the text index **cannot surface their rows as answers** — a multi-sheet table
doesn't chunk into meaningful prose, so a question like "what justifies prescribing a bergla" finds
nothing even though the answer is in a cell. They need a **structured parser + a dedicated tool**,
which is exactly what was done for the MTP device list (`get_mtp`).

**Detection heuristic** (regenerate with the snippet at the bottom):
> A manifest entry is a *structured rule source* if its real file extension (from `file_url`, NOT
> `file_type` — xlsx/zip are mislabeled `other`) is `xlsx/xls/zip/csv/xlsm` **and** its `title`
> matches `šifrant|kriterij|seznam|pooblastil|cenovn|postopk|pravila|navodil|omejit|indikac|
> standard|zagotavljanj`.

**Counts:** 176 distinct structured docs → 107 rule-bearing → **79 are `Okrožnica` change-circulars
now SUPERSEDED** by the deployed "Vsi šifranti" XML (`get_sifra`) → **28 substantive** remain.

## Tier 1 — medical-device prescribing cluster (complements `get_mtp`)

These directly help a doctor prescribe devices; high value. `DFDC…` is the one just shipped.

| File (unid) | Type | What it adds | Status |
|---|---|---|---|
| **Seznam MP s šifrantom, MK, postopki** `DFDC914987E44E2AC1257353003EC73A` | zip/xlsx | device indications, who-may-prescribe, odločba, duration, naročilnica, price | ✅ **shipped** (`get_mtp`, refreshed to 28.5.2026) |
| **Seznam MP, ki se medsebojno izključujejo** (Pravila OZZ) `268BDE43546B9383C1257C13002468FB` | xlsx | which devices may NOT be prescribed together | **recommend next** — pure prescribing rule, not in `get_mtp` |
| Sklep – seznam MP, kjer **izbrani osebni zdravnik** lahko predpiše `025200E09CC1A457C12588A900340AFA` | xlsx | which devices a personal physician (not just IOZ) may prescribe | partly in MTP `pristojnost`; xlsx is authoritative |
| Sklep – seznam **pooblaščenih zdravnikov** za predpisovanje MP `E8C27AD99A9C6F8CC1258329002D724B` | zip | named authorized prescribers per device | partly in MTP `Pooblaščeni zdravniki` sheets |
| Sklep – seznam MP, kjer je potrebna **tkivna banka / poseben pogoj** `E6FAB9723453F34FC12583240027152D` | xlsx | extra preconditions for certain devices | additive |
| Sklep – **cenovni standardi** MP `0F008D0D2300D954C1257DA50034A517` | zip | binding price standards | overlaps MTP `cenovni_standard` + šifrant CenovniStandardMTP |
| **Navodilo o zagotavljanju slušnih aparatov** `41F8EC0A3CA18217C1257AC9003426C0` | xlsx | device-class provisioning rules (hearing aids) | additive (class-specific detail) |
| **Navodilo o zagotavljanju MP za inkontinenco** (soc. zavodi) `15CA03764F460957C1257457003007A5` | xls | device-class provisioning rules (incontinence) | additive |
| Šifranti za izmenjavo podatkov o MP `08365D5741AC0693C1257D8F004AB2BB` | xls | MP data-exchange code lists | low (integration) |

**Recommendation:** add the **mutually-exclusive MP list** (`268B…`) to `get_mtp` next (a clear
prescribing constraint), and optionally fold the two device-class *Navodila* (sluš./inkontinenca) in
as extra notes. The Sklepi about prescribers/preconditions are largely already represented in the
MTP catalog + šifrant VrsteMTP fields — include only if the clinician wants the verbatim Sklep.

## Tier 2 — substantive but lower clinical-prescribing priority

- **Pravni akti (xlsx):** Navodilo za uveljavljanje pravic do **fizioterapije**
  `762B5D404B10C0CAC1257C4300359F74`; Navodila za **dolgotrajno oskrbo (DO)**
  `98441E397F8774BEC1258CCF004842C8`, `79A5F2ACB5BB4990C1258D650034819F`; Pravilnik o KZZ kartici
  `5F83195A36BFD240C1256B3B0051437D`. Operational/administrative — wire only on request.
- **Drug list changes:** Spremembe **Seznama A** `203748C4EDFCE033C1257B2D004706A2` / **Seznama B**
  `F641B38F0189A2BFC1257B2D0047239C` (zip). Drug prescribing is already covered by `drugs.py`/
  `get_prescription_limitations`/`zzzs.py`; these are the authoritative A/B + NPV deltas — medium.
- **Provider/person lists (Dogovor–Pogodbe):** farmacevtski svetovalci, fizioterapevti, pregled
  uporabe zdravil. Low clinical value (directory data).

## Already covered / superseded — do NOT re-index

- **79 `Okrožnica ZAE …` šifrant-change xlsx** → superseded by the live "Vsi šifranti" XML (`get_sifra`).
- **Navodilo o beleženju in obračunavanju** `8B39572A…` → in `obracun_rules` (`get_billing_rules`).
- **Tehnična navodila / XML structure / e-exchange specs** (9 zip, incl. `D2A24C1D…` =
  the "Vsi šifranti" XML schema) → IT-integration docs, no clinical value.

## Regenerate this audit

```python
import json, re
from urllib.parse import unquote
m = json.load(open("data/egradiva_manifest.json"))
ext = lambda e: (re.search(r"\.([A-Za-z0-9]{2,4})$",
        unquote(e.get("file_url") or "").split("/$FILE/")[-1]) or [None,""])[1].lower()
KW = re.compile(r"šifrant|kriterij|seznam|pooblastil|cenovn|postopk|pravila|navodil|omejit|indikac|standard|zagotavljanj", re.I)
seen = {}
for e in m:
    if ext(e) in {"xlsx","xls","zip","csv","xlsm"}: seen.setdefault(e["unid"], e)
rule = [e for e in seen.values() if KW.search(e.get("title") or "")]
subst = [e for e in rule if not re.search(r"okrožnic", (e.get("title") or "")+(e.get("category_name") or ""), re.I)]
# -> 176 structured, 107 rule-bearing, 28 substantive
```
Download any candidate via its `file_url`; parse with the `build_mtp_catalog.py` pattern (map the
columns, capture the rule/criteria column, emit a small JSON + a `get_*` lookup tool).

---

## Worked-through structured files (2026-06-24)

Opened and inspected all **70 distinct clinically-plausible** structured files (xls/xlsx/zip) by
content (openpyxl / xlrd / zip-by-magic-bytes), not by title, to see if any others are MTP-like
"present but unanswerable" tables worth a tool. Conclusion: **the MTP seznam was nearly unique.**

**Already covered elsewhere → no tool needed:**
- **Drug NPV / interchangeable-medicines list** (`1CF0EBC9…`, 1255 rows: national code, name, ATC,
  generic, therapeutic group, NPV) — already served live by `zzzs.py` `_fetch_zzzs_therapeutic_groups()`
  (`siftsz.csv`: group/class/**NPV**/regulated+agreed price) and per-drug NPV in `drugs.py` (CBZ). The
  xlsx is a staler snapshot. **Skipped.**
- **Uredba o programih storitev** xlsx (`4CF8AFB9…`, Priloga I/Ia service-program seznam) — overlaps
  `get_sifra` (SifrantStoritev) + `obracun` (Uredba text). MP group codes (`12C991AD…`) → actually a
  PDF; the group codes are in the šifrant (SkupineMTP/Podskupine).

**Not actually a table (PDF/prose/scan already in the text index):** `9458214546` (copay %),
`15CA0376` (inkontinenca), `9B063B0F`/`0F113F00` (criteria docx), `0F008D0D` (old `.tif` scans).

**Low clinical value:** `41F8EC0A` "slušni aparati" is a **supplier/dispensing-location directory**
(846 rows of vendors), not prescribing rules; Splošni-dogovor aneksi 2004-2022, komisija minutes,
tender results (psi vodiči), foreign-insured groups.

**Overlaps `get_mtp` columns → folded in, not standalone:** `025200E0` (IOZ-predpis flag),
`E3A95F6B` (obnovljiva naročilnica + pogoj).

**The one genuine, additive miss → implemented:** `268BDE43…` **"Seznam MP, ki se medsebojno
izključujejo na podlagi Pravil OZZ"** — a clean **1,898-row code→code table** (device šifra → device
šifra that may not be prescribed with it), in neither `get_mtp` nor the šifrant (whose exclusion
tables are for *services*). Now parsed in `build_mtp_catalog.py` and attached as `izkljucuje` on each
device (symmetric, 1024 pairs over 230 devices), surfaced by `get_mtp` as "Se medsebojno izključuje
(ne hkrati) z: …". Also folded in **timska obravnava** (`E6FAB972…`, 89 devices) as a flag.

**Niche, noted only:** `9D110A02…` NPV for special medical foods (172 rows) — small; revisit if
medical-nutrition prescribing becomes a frequent question.
