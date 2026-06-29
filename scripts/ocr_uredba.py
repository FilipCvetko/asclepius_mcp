#!/usr/bin/env python3
"""OCR the scanned Uredba o programih storitev (2026) PDF → page-level text JSON.

The Uredba PDF has no text layer (fully scanned), so we rasterize each page and run
Slovenian tesseract OCR. Output: data/billing_files/<unid>_ocr.json = [{page, text}],
consumed by build_obracun_index.py. Run once locally; the result is committed/shipped.

Requires: tesseract + slv traineddata, pytesseract, pillow, PyMuPDF.
"""
import io
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

DATA = Path(__file__).parent.parent / "data"
SOURCES = DATA / "billing_sources.json"
DPI = 300


def find_uredba_pdf() -> dict | None:
    src = json.loads(SOURCES.read_text(encoding="utf-8"))
    for s in src:
        if s["role"] == "uredba_programi":
            for f in s["files"]:
                if f["ext"] == "pdf":
                    return {"doc": s, "file": f}
    return None


def main() -> None:
    hit = find_uredba_pdf()
    if not hit:
        print("No Uredba PDF found in billing_sources.json")
        sys.exit(1)
    path = hit["file"]["path"]
    out = DATA / "billing_files" / f"{hit['doc']['unid']}_ocr.json"

    doc = fitz.open(path)
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pages = []
    print(f"OCR {Path(path).name}: {len(doc)} pages @ {DPI}dpi (slv)...")
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="slv")
        pages.append({"page": i + 1, "text": text.strip()})
        if (i + 1) % 10 == 0 or i + 1 == len(doc):
            got = sum(1 for p in pages if len(p["text"]) > 40)
            print(f"  {i+1}/{len(doc)}  (pages with text: {got})")

    out.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(len(p["text"]) for p in pages)
    print(f"\nSaved OCR → {out}  ({total:,} chars, "
          f"{sum(1 for p in pages if len(p['text'])>40)}/{len(pages)} non-empty pages)")


if __name__ == "__main__":
    main()
