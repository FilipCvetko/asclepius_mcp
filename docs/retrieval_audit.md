# Retrieval-quality audit — Asclepius / ZZZS e-gradiva

**Date:** 2026-06-23 · **Scope:** how well the crawled ZZZS corpus (3392 docs,
`data/egradiva_manifest.json`) is actually served to a clinician, and where it isn't.

Companion to [`structured_sources_audit.md`](structured_sources_audit.md) (the structured-Excel
sub-problem). This doc is the broader picture: **what classes of source content are poorly served,
mapped to the questions a doctor actually asks.**

---

## 1. Failure modes (why "crawled" ≠ "answerable")

The broad index (`zzzs_egradiva`, tool `search_zzzs_documents`) is built by
`scripts/build_egradiva_index.py`: download → extract text → chunk → embed. Four classes of content
fall through:

| # | Failure mode | Cause (in the build) | Symptom |
|---|---|---|---|
| A | **Legacy `.doc` dropped** | `extract_text_from_file` only handled `pdf`/`docx`; python-docx can't read `.doc` (OLE) | 257 docs (Word 97-2003) contribute nothing |
| B | **Scanned PDFs / `.tif` invisible** | `extract_pdf_text` skipped pages <50 chars as "possibly scanned" — **no OCR**; `.tif` unhandled | image-only docs (incl. the Uredba originally) silently empty |
| C | **Structured spreadsheets/zip dropped** | `.xls/.xlsx/.zip` returned `[]` | 556 xls(x) + 304 zip not searchable (the MTP case) |
| D | **Table-in-PDF imprecision** | semantic chunking of dense tables | even *indexed* point/price/code tables answer vaguely |

**Quantified (pre-fix):** of 3392 docs, only **2042 (pdf+docx) were extractable**; **~1350 (40%)
were dropped** — by real extension: xlsx 438, zip 304, doc 257, xls 118, **tif 90**, none 48,
xsd 37, pptx 33, + binaries. Most dropped docs are low clinical value (IT specs, superseded
okrožnice, binaries), but **~100 are clinically relevant**.

---

## 2. Doctor-question → current coverage

| Clinical question | Best source | Served today? |
|---|---|---|
| May I prescribe device X / what justifies it (indikacije) | MTP seznam (xlsx) | ✅ `get_mtp` |
| What is billing/service code X; points/price | Vsi šifranti XML | ✅ `get_sifra` / `get_billing_rules` |
| Rights / coverage / reimbursement under OZZ | Pravila OZZ | ✅ `get_pravila_ozz` |
| Drug info, prescribing limitations, ZZZS list | CBZ / ZZZS | ✅ `drugs` / `get_prescription_limitations` / `get_zzzs_drug_limitation` |
| Spa / rehab eligibility | spa list | ✅ `get_spa_eligibility` |
| How to record/bill a service (obračun, kodiranje) | billing navodila (OCR'd) | ✅ `get_billing_rules` |
| **Which diagnoses give 100% coverage (no doplačilo)** | "Kode MKB …100% plačilo" xlsx (2014) | ⚠️ **gap** (dropped; currency unverified) |
| **Which devices are mutually exclusive** | "Seznam MP, ki se medsebojno izključujejo" xlsx | ⚠️ **gap** (dropped) |
| **Device-class provisioning (slušni aparati, inkontinenca)** | Navodila MP (docx/xls) | ◑ partial (docx indexed; xls dropped) |
| **Long-term care (DO) rules** | DO priročniki/navodila (xlsx, 2025) | ⚠️ **gap** (dropped) |
| **Older guidance only in `.doc`** | legacy Navodila/Dogovori | ⚠️ **gap** (format dropped) → fixed by this change |
| Free-text guidance in scanned circulars | scanned PDFs / `.tif` | ⚠️ **gap** → fixed by this change (OCR) |

---

## 3. This change — index recall fix (A + B)

`scripts/build_egradiva_index.py` patched to recover the two highest-leverage, broad classes:

- **OCR fallback (B):** scanned PDF pages (<50 chars) are rendered (`fitz` @200 DPI) and OCR'd in
  Slovenian (`pytesseract lang="slv"`, same as `ocr_uredba.py`); `.tif/.tiff` OCR'd directly.
- **Legacy `.doc` (A):** extracted via `wvText` (wvWare), with antiword/libreoffice fallbacks.
  Build-machine-only dep (`brew install wv`); not shipped in the runtime image.
- **Real-extension dispatch:** files are stored as `{unid}.{file_type}` where `file_type` collapses
  xls/zip/tif into `other`, so dispatch now uses the real extension from `file_url`.
- Guarded by `EGRADIVA_OCR=0` / `--no-ocr`; per-type extraction counts logged.

**Recovered (re-index of 2026-06-24):** chunk count **114,027 → 144,426 (+30,399, +27%)**.
Per-type `EXTRACT_STATS`:
- **257 legacy `.doc`** files — all recovered (via `wvText`).
- **4,335 scanned PDF pages** OCR'd; **80 `.tif`** scanned images OCR'd.
- **91 giant scanned PDFs skipped** by the `OCR_MAX_PAGES=40` cap (annual reports / finančni
  načrti — low clinical value; OCR'ing hundreds of pages each would stall the build for hours).
- Still skipped (structured-tool backlog): 438 xlsx, 304 zip, 118 xls; plus pptx/binaries.

Re-index: `KMP_DUPLICATE_LIB_OK=TRUE venv/bin/python scripts/build_egradiva_index.py --download
--index` (add `--no-ocr` or `EGRADIVA_OCR_MAX_PAGES=N` to tune); bump `data/chromadb/.seed_version`
→ deploy triggers a one-time volume reseed.

---

## 4. Backlog (not done here) — recommended next, by value

1. **Structured tools for the few clean, current tables (class C)** — same pattern as `get_mtp`:
   - `Seznam MP, ki se medsebojno izključujejo` (`268BDE43…`, xlsx) — mutually-exclusive devices;
     a hard prescribing constraint. **Highest-value, cleanest.**
   - Long-term-care (DO) priročniki (xlsx, 2025) — newer domain, likely high demand.
   - Diagnosis→coverage% ("Kode MKB 100% plačilo" `66F003B5…`) — **verify currency first** (the
     crawled copy is *januar 2014*; confirm there is no newer authoritative source before exposing).
2. **Precision (class D):** for dense point/price tables already in the index, prefer structured
   extraction over semantic search — extend `get_sifra`/`get_mtp` coverage rather than re-chunk.
3. **Skip:** `.pptx`, `.xsd/.xslt/.jar/.msi`, superseded `Okrožnice ZAE` (covered by `get_sifra`),
   historical aneksi/dogovori — low or zero clinical value.

## 5. Maintenance note
The recall fix re-runs offline; OCR + `.doc` need `tesseract`+`slv` and `wv` on the **build**
machine only (already installed). Re-run after any future re-crawl. Structured catalogs
(`get_sifra`, `get_mtp`) have their own `fetch_*`/`build_*` refresh scripts.
