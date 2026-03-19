#!/usr/bin/env python3
"""Download Pravila OZZ NPB42 PDF and parse into article-level chunks for zzzs_rules.json."""

import json
import re
import sys
from pathlib import Path

import fitz  # pymupdf

PDF_URL = "https://api.zzzs.si/ZZZS/info/egradiva.nsf/0/5580d0555f5a1feac1256cfb003bb45c/$FILE/Pravila%20OZZ_NPB42.pdf"
OUTPUT = Path(__file__).parent.parent / "data" / "zzzs_rules.json"

# Section prefix -> category tag
CATEGORY_MAP = {
    "I": "splošne določbe",
    "II": "pridobitev zavarovanja",
    "III": "izkazovanje",
    "IV/1": "osnovno zdravstvo",
    "IV/2": "zobozdravstvo",
    "IV/3": "institucionalno",
    "IV/4": "specialistično",
    "IV/5": "zdravilišče",
    "IV/6": "rehabilitacija",
    "IV/7": "prevoz",
    "IV/8": "zdravila",
    "IV/9": "spremstvo",
    "V": "pripomočki",
    "V/1": "pripomočki",
    "V/2": "pripomočki",
    "V/3": "pripomočki",
    "V/4": "pripomočki",
    "V/5": "pripomočki",
    "V/6": "pripomočki",
    "VI": "nujna pomoč",
    "VII": "standardi",
    "VIII": "trajanje pripomočkov",
    "IX": "predhodno zavarovanje",
    "X": "tujina potovanje",
    "XI": "zdravljenje v tujini",
    "XII/1": "nadomestilo plače",
    "XII/2": "pogrebnina posmrtnina",
    "XII/3": "potni stroški",
    "XIII/1": "izbira zdravnika",
    "XIII/2": "zamenjava zdravnika",
    "XIII/3": "postopek osnovno",
    "XIII/4": "postopek prevoz",
    "XIII/5": "postopek zobozdravstvo",
    "XIII/6": "napotnice specialist",
    "XIII/7": "postopek zdravilišče",
    "XIII/8": "predpisovanje zdravil",
    "XIII/9": "postopek pripomočki",
    "XIII/10": "postopek tujina",
    "XIII/11": "bolniška",
    "XIII/12": "postopek potni stroški",
    "XIII/13": "drugo",
    "XIV": "organi izvedenci",
    "XV": "nadzor",
    "XVI": "prehodne določbe",
}


def download_pdf(url: str) -> bytes:
    """Download PDF content."""
    import requests
    print(f"Downloading PDF from {url[:80]}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    print(f"Downloaded {len(resp.content)} bytes")
    return resp.content


def extract_text(pdf_bytes: bytes) -> str:
    """Extract all text from PDF, stripping page headers/footers."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        # Strip common header/footer patterns
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip page numbers and common headers
            if re.match(r"^\d+$", stripped):
                continue
            if re.match(r"^Stran \d+ od \d+", stripped):
                continue
            if stripped.startswith("Pravila obveznega zdravstvenega zavarovanja"):
                continue
            cleaned.append(line)
        pages.append("\n".join(cleaned))
    doc.close()
    return "\n".join(pages)


def detect_category(section_key: str) -> str:
    """Map a section key to its category tag."""
    # Try exact match first, then prefix
    if section_key in CATEGORY_MAP:
        return CATEGORY_MAP[section_key]
    # Try parent (e.g., "V/3" -> "V")
    parts = section_key.split("/")
    if parts[0] in CATEGORY_MAP:
        return CATEGORY_MAP[parts[0]]
    return "drugo"


def parse_articles(full_text: str) -> list:
    """Parse text into article-level chunks with chapter/section context."""

    # Regex patterns
    # Chapter: Roman numeral at start of line followed by period and uppercase title
    chapter_pat = re.compile(
        r"^([IVX]+)\.\s+([A-ZŠŽČĆĐ].+)$", re.MULTILINE
    )
    # Section: Roman/number like IV/1. or XIII/10.
    section_pat = re.compile(
        r"^([IVX]+/\d+)\.\s+(.+)$", re.MULTILINE
    )
    # Article: "123. člen" or "57.a člen" or "135.a člen"
    article_pat = re.compile(
        r"^(\d+\.?[a-z]?)\s*[čc]len\b", re.MULTILINE | re.IGNORECASE
    )

    # First pass: find all chapter and section boundaries
    chapters = []
    for m in chapter_pat.finditer(full_text):
        chapters.append({
            "pos": m.start(),
            "numeral": m.group(1),
            "title": f"{m.group(1)}. {m.group(2).strip()}",
        })

    sections = []
    for m in section_pat.finditer(full_text):
        sections.append({
            "pos": m.start(),
            "key": m.group(1),
            "title": f"{m.group(1)}. {m.group(2).strip()}",
        })

    # Find all articles
    article_matches = list(article_pat.finditer(full_text))
    if not article_matches:
        print("WARNING: No articles found!")
        return []

    print(f"Found {len(chapters)} chapters, {len(sections)} sections, {len(article_matches)} articles")

    # Build articles
    rules = []
    for i, m in enumerate(article_matches):
        art_start = m.start()
        # Article text goes until next article start (or end of text)
        if i + 1 < len(article_matches):
            art_end = article_matches[i + 1].start()
        else:
            art_end = len(full_text)

        art_text = full_text[art_start:art_end].strip()

        # Parse article number
        raw_num = m.group(1)
        # Extract numeric part for sorting
        num_match = re.match(r"(\d+)", raw_num)
        art_num = int(num_match.group(1)) if num_match else 0
        art_title = f"{raw_num} člen"

        # Find parent chapter (last chapter before this position)
        current_chapter = ""
        for ch in chapters:
            if ch["pos"] <= art_start:
                current_chapter = ch["title"]
            else:
                break

        # Find parent section (last section before this position)
        current_section = ""
        current_section_key = ""
        for sec in sections:
            if sec["pos"] <= art_start:
                current_section = sec["title"]
                current_section_key = sec["key"]
            else:
                break

        # If no section found, derive key from chapter numeral
        if not current_section_key:
            for ch in chapters:
                if ch["pos"] <= art_start:
                    current_section_key = ch["numeral"]
                else:
                    break

        category = detect_category(current_section_key)

        rules.append({
            "title": art_title,
            "section": current_section or current_chapter,
            "chapter": current_chapter,
            "text": art_text,
            "category": category,
            "article_number": art_num,
            "source": "Pravila OZZ NPB42",
            "source_url": PDF_URL,
        })

    return rules


def main():
    # Download
    pdf_bytes = download_pdf(PDF_URL)

    # Extract text
    print("Extracting text from PDF...")
    full_text = extract_text(pdf_bytes)
    print(f"Extracted {len(full_text)} characters of text")

    # Parse articles
    print("Parsing articles...")
    rules = parse_articles(full_text)

    if not rules:
        print("ERROR: No articles parsed! Aborting.")
        sys.exit(1)

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(rules)} rules to {OUTPUT}")

    # Quick stats
    categories = {}
    for r in rules:
        cat = r["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Spot-check a few articles
    for target in [57, 135, 64, 1]:
        matches = [r for r in rules if r["article_number"] == target]
        if matches:
            r = matches[0]
            print(f"\nSpot-check: {r['title']} ({r['category']}) - {r['text'][:80]}...")


if __name__ == "__main__":
    main()
