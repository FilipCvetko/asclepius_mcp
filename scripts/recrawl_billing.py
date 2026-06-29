#!/usr/bin/env python3
"""Targeted re-crawl of the OUTPATIENT billing categories from the live ZZZS API.

Listings-only (fast): fetches document lists WITH dates (DATUM) — which the original
egradiva crawl dropped — so we can pick the genuinely current version of each source.
Does NOT download files yet; that happens after the current set is confirmed.

v1 scope: outpatient billing (hospital/SPP descoped). Codes come from the Uredba
Excel annexes, so the relevant categories are the Navodilo, the Dogovor (Uredba), and
the billing Q&A.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from build_egradiva_index import crawl_categories, crawl_documents_for_category  # noqa: E402

OUT = Path(__file__).parent.parent / "data" / "billing_recrawl.json"

# Category names we care about (substring match against live VrstaGradiva)
WANTED = [
    "Navodilo o beleženju in obračunavanju",
    "Dogovor",                       # contains the Uredba o programih storitev
    "Navodila za obračun",           # vprašanja in odgovori + STANDARDI KODIRANJA
]


def _date(doc: dict) -> str:
    return (doc.get("DATUM") or doc.get("datum") or "").strip()


def main() -> None:
    cats = crawl_categories()
    print(f"Live categories: {len(cats)}")
    targets = [c for c in cats if any(w.lower() in c["name"].lower() for w in WANTED)]
    print("Matched billing categories:")
    for c in targets:
        print(f"   [{c['code']}] {c['name']}")

    all_docs = []
    for c in targets:
        docs = crawl_documents_for_category(c)
        for d in docs:
            all_docs.append({
                "category": c["name"],
                "unid": d.get("@unid") or d.get("unid", ""),
                "title": (d.get("NASLOV") or d.get("naslov") or "").strip(),
                "date": _date(d),
            })

    OUT.write_text(json.dumps(all_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_docs)} listings → {OUT}")

    # Report the newest documents per role
    def newest(pred, n=6):
        items = [d for d in all_docs if pred(d["title"])]
        items.sort(key=lambda d: d["date"], reverse=True)
        return items[:n]

    print("\n=== NEWEST: Uredba o programih storitev (outpatient programs/points) ===")
    for d in newest(lambda t: re.search(r"uredba o programih storitev", t, re.I)):
        print(f"   {d['date']:12} {d['title'][:75]}")

    print("\n=== NEWEST: Navodilo o beleženju in obračunavanju ===")
    for d in newest(lambda t: re.search(r"navodilo o beleženju", t, re.I)):
        print(f"   {d['date']:12} {d['title'][:75]}")

    print("\n=== NEWEST: Navodilo za obračun (Q&A / standardi kodiranja) ===")
    for d in newest(lambda t: re.search(r"navodilo za obračun|vprašanja in odgovori|standardi kodiranja", t, re.I)):
        print(f"   {d['date']:12} {d['title'][:75]}")


if __name__ == "__main__":
    main()
