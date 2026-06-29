#!/usr/bin/env python3
"""Step 0 (discovery/curation) for the billing-rules tool.

Scans data/egradiva_manifest.json and proposes the CURRENT-year canonical billing
sources (outpatient + hospital) grouped by document. Writes a reviewable
data/billing_sources.json and prints a human summary. This does NOT ingest anything —
its output is meant for human sign-off before building the obracun_rules index.

Manifest rows have: unid, title, category_name, file_url, file_type, local_path.
There are multiple rows per unid (one per attached file / annex).
"""
import json
import re
import collections
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
MANIFEST = DATA / "egradiva_manifest.json"
OUT = DATA / "billing_sources.json"


def real_ext(url: str) -> str:
    m = re.search(r"\.([a-z0-9]{2,5})(?:\?|$|%)", (url or "").lower())
    return m.group(1) if m else "?"


def year_of(title: str):
    m = re.search(r"pogodbeno leto\s*(\d{4})", title, re.I) or re.search(r"\b(20\d{2})\b", title)
    return int(m.group(1)) if m else None


# role → (category predicate, title regex or None)
def role_of(cat: str, title: str) -> str | None:
    t = title.lower()
    if re.search(r"čistopis|cistopis", t) and re.search(r"šifrant|sifrant", t):
        return "cistopis_sifrantov"          # the code lists (outpatient)
    if cat.startswith("Navodilo o beleženju"):
        return "navodilo_obracun"            # billing/recording instruction
    if cat == "Dogovor – Dogovor" and re.search(r"uredba o programih storitev", t):
        return "uredba_programi"             # Splošni-dogovor successor (2023+)
    if cat == "Dogovor – Dogovor" and re.search(r"splošni dogovor|splosni dogovor", t):
        return "splosni_dogovor"             # older program/pricing agreement
    if cat == "Navodila za obračun - vprašanja in odgovori":
        return "billing_qa"                  # billing Q&A
    if re.search(r"\bSPP\b|skupine primerljivih primerov|uteži|utezi|"
                 r"akutn\w+ bolni[šs]ni[čc]n", title, re.I):
        return "spp_hospital"                # hospital case grouping (SPP/DRG)
    return None


OUTPATIENT = {"cistopis_sifrantov", "navodilo_obracun", "uredba_programi",
              "splosni_dogovor", "billing_qa"}


def main() -> None:
    man = json.load(open(MANIFEST, encoding="utf-8"))

    # group rows by (unid) → doc with files
    docs: dict[str, dict] = {}
    for m in man:
        role = role_of(m.get("category_name", ""), m.get("title", ""))
        if not role:
            continue
        unid = m.get("unid", "")
        d = docs.setdefault(unid, {
            "unid": unid,
            "title": m.get("title", ""),
            "category": m.get("category_name", ""),
            "role": role,
            "domain": "hospital" if role == "spp_hospital" else "outpatient",
            "year": year_of(m.get("title", "")),
            "files": [],
        })
        if m.get("file_url"):
            d["files"].append({"ext": real_ext(m["file_url"]),
                               "file_type": m.get("file_type"),
                               "url": m["file_url"]})

    by_role: dict[str, list] = collections.defaultdict(list)
    for d in docs.values():
        by_role[d["role"]].append(d)

    # Year-bearing roles → keep only the latest year (current contract year)
    selected: dict[str, list] = {}
    for role, items in by_role.items():
        if role in ("uredba_programi", "splosni_dogovor"):
            yrs = [d["year"] for d in items if d["year"]]
            latest = max(yrs) if yrs else None
            keep = [d for d in items if d["year"] == latest]
            for d in keep:
                d["status"] = "current-year (auto)" if latest else "needs_review (no year)"
            selected[role] = sorted(keep, key=lambda d: d["title"])
        else:
            for d in items:
                d["status"] = "needs_review (pick current)"
            selected[role] = sorted(items, key=lambda d: d["title"])

    OUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── human summary ──
    order = ["uredba_programi", "splosni_dogovor", "navodilo_obracun",
             "cistopis_sifrantov", "billing_qa", "spp_hospital"]
    print(f"Curated billing-source candidates → {OUT}\n")
    for role in order:
        items = selected.get(role, [])
        dom = "hospital" if role == "spp_hospital" else "outpatient"
        print(f"== {role}  [{dom}]  — {len(items)} doc(s) ==")
        for d in items[:8]:
            exts = ",".join(sorted({f['ext'] for f in d['files']})) or "—"
            yr = d["year"] or "?"
            print(f"   ({yr}) [{exts:9}] {d['title'][:72]}  — {d['status']}")
        if len(items) > 8:
            print(f"   … +{len(items)-8} more")
        if not items:
            print("   (none found)")
        print()

    n_files = sum(len(d["files"]) for items in selected.values() for d in items)
    print(f"Total candidate docs: {sum(len(v) for v in selected.values())}  "
          f"(files: {n_files}). REVIEW before ingest — esp. SPP (hospital) and which "
          f"Navodilo/Čistopis is current.")


if __name__ == "__main__":
    main()
