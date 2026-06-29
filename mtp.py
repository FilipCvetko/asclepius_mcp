"""Medical-device (MTP) prescribing lookup — authoritative, structured.

Backed by data/mtp_catalog.json — the ZZZS "Seznam medicinskih pripomočkov s šifrantom, MK,
postopki" (built by scripts/fetch_mtp.py + build_mtp_catalog.py). For each medicinsko-tehnični
pripomoček it returns the prescribing rules a clinician needs: the **indications** (bolezen /
zdravstveno stanje in drugi pogoji that justify the device), who may prescribe it, whether an
odločba IOZ is required, the duration (doba trajanja), naročilnica rules, izposoja and price.

This is the authoritative source for "may I prescribe device X / what justifies it" — the broad
text index can't surface these structured rows. Bidirectional: a code (e.g. 0501), a device name
(bergla), or a condition (amputacija) all resolve to the matching device(s).
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastmcp.tools.tool import ToolResult

from paths import data_path

CATALOG_FILE = Path(__file__).parent / "data" / "mtp_catalog.json"
SOURCE_LABEL = "ZZZS Seznam medicinskih pripomočkov s šifrantom (MK, postopki)"

DEVICES: List[Dict[str, Any]] = []
BY_CODE: Dict[str, Dict[str, Any]] = {}
DATUM: str = ""
SOURCE_URL: str = ""
_VEC = None
_MAT = None
_CODE_RE = re.compile(r"^\d{3,5}$")

# Display order + Slovenian labels for the prescribing fields.
_FIELDS = [
    ("indikacije", "Indikacije (bolezen / zdravstveno stanje in drugi pogoji)"),
    ("pristojnost", "Kdo predpiše (pristojnost)"),
    ("odlocba", "Odločba imenovanega zdravnika"),
    ("timska_obravnava", "Timska obravnava potrebna"),
    ("doba_trajanja", "Doba trajanja"),
    ("predpis_narocilnice", "Predpis naročilnice (dni)"),
    ("obnovljiva_narocilnica", "Obnovljiva naročilnica"),
    ("podaljsanje_izposoje", "Podaljšanje izposoje / nova obn. naročilnica"),
    ("izposoja", "Izposoja"),
    ("popravila", "Popravila / vzdrževanje"),
    ("prilagoditve", "Prilagoditve"),
    ("materialni_strosek", "Materialni strošek (SVZ)"),
    ("cenovni_standard", "Cenovni standard / pogodbena cena"),
    ("kategorija", "Kategorija"),
]


def initialize_mtp() -> None:
    """Load the MTP catalog, build a code index + TF-IDF over name/indications/category."""
    global DEVICES, BY_CODE, DATUM, SOURCE_URL, _VEC, _MAT
    catalog_file = data_path("mtp_catalog.json")       # volume copy wins, else baked
    if not catalog_file.exists():
        print("WARNING: MTP catalog not found at", catalog_file)
        return
    data = json.loads(catalog_file.read_text(encoding="utf-8"))
    DEVICES = data.get("devices", [])
    DATUM = data.get("datum", "")
    SOURCE_URL = data.get("source_url", "")
    BY_CODE = {d["sifra"].strip(): d for d in DEVICES if d.get("sifra")}
    if DEVICES:
        corpus = [f"{d.get('sifra','')} {d.get('ime','')} {d.get('kategorija','')} "
                  f"{d.get('indikacije','')}" for d in DEVICES]
        _VEC = TfidfVectorizer()
        _MAT = _VEC.fit_transform(corpus)
    print(f"Loaded MTP catalog: {len(DEVICES)} devices (datum {DATUM})")


def _citation() -> str:
    return f"{SOURCE_LABEL}, {DATUM}"


def _lookup_mtp(query: str, n_results: int = 6) -> ToolResult:
    """Find medical device(s) by code, name, or condition, with full prescribing rules."""
    if not DEVICES:
        err = {"found": False, "error": "MTP catalog not loaded"}
        return ToolResult(content=_format_md([], query, err), structured_content=err)
    query = (query or "").strip()
    if not query:
        err = {"found": False, "error": "Query must not be empty"}
        return ToolResult(content=_format_md([], query, err), structured_content=err)

    results, seen = [], set()

    # 1. exact code(s): the whole query or any token that is a code
    for tok in [query] + query.split():
        d = BY_CODE.get(tok.strip())
        if d and d["sifra"] not in seen:
            results.append({**d, "match": "exact-code"})
            seen.add(d["sifra"])

    # 2. name / condition search (TF-IDF) to fill remaining slots
    if len(results) < n_results and _VEC is not None:
        sims = cosine_similarity(_VEC.transform([query]), _MAT).flatten()
        for i in sims.argsort()[::-1]:
            if sims[i] <= 0.05 or len(results) >= n_results:
                break
            d = DEVICES[i]
            if d.get("sifra") in seen:
                continue
            results.append({**d, "match": "ime/indikacije", "relevance": round(float(sims[i]), 4)})
            seen.add(d.get("sifra"))

    if not results:
        err = {"found": False, "error": "No matching device", "query": query}
        return ToolResult(content=_format_md([], query, err), structured_content=err)
    data = {"found": True, "count": len(results), "query": query, "datum": DATUM,
            "source_url": SOURCE_URL, "results": results}
    return ToolResult(content=_format_md(results, query, data), structured_content=data)


def _format_md(devices: List[Dict[str, Any]], query: str, data: dict) -> str:
    if not data.get("found"):
        return (f"V seznamu medicinskih pripomočkov ni zadetka za \"{query}\". "
                f"{data.get('error','')}").strip()
    lines = [f"**Medicinsko-tehnični pripomočki (MTP)** | {len(devices)} zadetkov za "
             f"\"{query}\"  ·  stanje {DATUM}", ""]
    for d in devices:
        lines += ["---", "", f"### Šifra {d.get('sifra','')} — {d.get('ime','')}", ""]
        for key, label in _FIELDS:
            val = d.get(key, "")
            if val:
                lines.append(f"- **{label}:** {val}")
        excl = d.get("izkljucuje") or []
        if excl:
            items = ", ".join(f"{e['sifra']} {e['naziv']}" for e in excl)
            lines.append(f"- **Se medsebojno izključuje (ne hkrati) z:** {items}")
        if d.get("relevance") is not None:
            lines.append(f"- *relevantnost:* {d['relevance']}")
        lines.append("")
        url = d.get("source_url") or SOURCE_URL
        lines.append(f"**Vir:** [{_citation()}]({url})" if url else f"**Vir:** {_citation()}")
        lines.append("")
    lines += ["---", "",
              "_Vir je ZZZS Seznam medicinskih pripomočkov s šifrantom (medicinski kriteriji, "
              "pooblastila, postopki). Avtoritativen za upravičenost/pogoje predpisa MTP — "
              "preveri morebitno novejšo objavo. Kratice (npr. IOZ, DMS, *TD) so pojasnjene v viru._"]
    return "\n".join(lines)


if __name__ == "__main__":
    initialize_mtp()
    for q in ["bergla", "0501", "voziček", "amputacija", "slušni aparat"]:
        print("\n" + "=" * 70, f"\nQUERY {q!r}")
        sc = _lookup_mtp(q, 3).structured_content
        print("found:", sc.get("found"), "count:", sc.get("count"))
        for r in sc.get("results", []):
            print(f"  [{r['match']}] {r.get('sifra')} {r.get('ime','')[:40]:40s} "
                  f"indik={r.get('indikacije','')[:45]}")
