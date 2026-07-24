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
_LIST_CAP = 40                   # max article names rendered in the "Sistem / sestavni artikli" list

DEVICES: List[Dict[str, Any]] = []
BY_CODE: Dict[str, Dict[str, Any]] = {}
DATUM: str = ""
SOURCE_URL: str = ""
ENRICHED: int = 0                # devices carrying "Vsi šifranti" prescribing enrichment
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
    global DEVICES, BY_CODE, DATUM, SOURCE_URL, ENRICHED, _VEC, _MAT
    catalog_file = data_path("mtp_catalog.json")       # volume copy wins, else baked
    if not catalog_file.exists():
        print("WARNING: MTP catalog not found at", catalog_file)
        return
    data = json.loads(catalog_file.read_text(encoding="utf-8"))
    DEVICES = data.get("devices", [])
    DATUM = data.get("datum", "")
    SOURCE_URL = data.get("source_url", "")
    ENRICHED = data.get("enriched", 0)
    BY_CODE = {d["sifra"].strip(): d for d in DEVICES if d.get("sifra")}
    if DEVICES:
        corpus = [f"{d.get('sifra','')} {d.get('ime','')} {d.get('kategorija','')} "
                  f"{d.get('indikacije','')}" for d in DEVICES]
        _VEC = TfidfVectorizer()
        _MAT = _VEC.fit_transform(corpus)
    print(f"Loaded MTP catalog: {len(DEVICES)} devices (datum {DATUM})")


def _citation() -> str:
    return f"{SOURCE_LABEL}, {DATUM}"


def _resolve_devices(query: str, n_results: int) -> List[Dict[str, Any]]:
    """Resolve device(s) by exact code first, then TF-IDF name/condition search — best first.
    Shared by _lookup_mtp and _lookup_mtp_prescribing so both entry points match identically."""
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
    return results


def _lookup_mtp(query: str, n_results: int = 6) -> ToolResult:
    """Find medical device(s) by code, name, or condition, with full prescribing rules."""
    if not DEVICES:
        err = {"found": False, "error": "MTP catalog not loaded"}
        return ToolResult(content=_format_md([], query, err), structured_content=err)
    query = (query or "").strip()
    if not query:
        err = {"found": False, "error": "Query must not be empty"}
        return ToolResult(content=_format_md([], query, err), structured_content=err)

    results = _resolve_devices(query, n_results)
    if not results:
        err = {"found": False, "error": "No matching device", "query": query}
        return ToolResult(content=_format_md([], query, err), structured_content=err)
    data = {"found": True, "count": len(results), "query": query, "datum": DATUM,
            "source_url": SOURCE_URL, "results": results}
    return ToolResult(content=_format_md(results, query, data), structured_content=data)


def _lookup_mtp_prescribing(query: str) -> ToolResult:
    """Focused single-device answer to the three prescribing questions: WHO may prescribe, the
    renewal PERIOD + max QUANTITY (per period / per day), and same-group devices prescribable
    alongside it (mutually-exclusive ones flagged). All from the ZZZS "Vsi šifranti" MP tables."""
    if not DEVICES:
        err = {"found": False, "error": "MTP catalog not loaded"}
        return ToolResult(content=_format_prescribing_md(None, query, err), structured_content=err)
    query = (query or "").strip()
    if not query:
        err = {"found": False, "error": "Query must not be empty"}
        return ToolResult(content=_format_prescribing_md(None, query, err), structured_content=err)

    results = _resolve_devices(query, 1)
    if not results:
        err = {"found": False, "error": "No matching device", "query": query}
        return ToolResult(content=_format_prescribing_md(None, query, err), structured_content=err)
    d = results[0]
    data = {"found": True, "query": query, "sifra": d.get("sifra"), "ime": d.get("ime"),
            "prescriber": d.get("prescriber"), "quantity_periods": d.get("quantity_periods"),
            "nacin_dobe": d.get("nacin_dobe"), "doba_mode": d.get("doba_mode"),
            "group": d.get("group"), "group_peers": d.get("group_peers"),
            "group_peers_overflow": d.get("group_peers_overflow"), "systems": d.get("systems"),
            "datum": DATUM, "source_url": d.get("source_url") or SOURCE_URL}
    return ToolResult(content=_format_prescribing_md(d, query, data), structured_content=data)


def _render_prescribing(d: Dict[str, Any]) -> List[str]:
    """Render the three enrichment sections (Kdo predpiše / Obdobje in količina / Ista skupina),
    from the build-time šifrant join. Returns [] when the device carries no enrichment (e.g. an
    older catalog built before the join) so callers can guard cleanly."""
    lines: List[str] = []

    # (1) Who may prescribe — verbatim category text is authoritative; chips are the derived axes.
    p = d.get("prescriber")
    if p:
        lines.append("**Kdo predpiše:**")
        if p.get("opis"):
            lines.append(f"> {p['opis']}")
        cats = []
        if p.get("personal"):
            cats.append("osebni zdravnik")
        if p.get("specialist"):
            cats.append("specialist")
        if p.get("nurse"):
            cats.append("medicinska sestra (DMS)")
        if cats:
            lines.append(f"- Kategorije predpisovalca: {', '.join(cats)}")
        if p.get("odlocba_ioz"):
            lines.append(f"- Odločba imenovanega zdravnika (IOZ): {p['odlocba_ioz']}")
        lines.append("")

    # (2) Period + max quantity — only what the šifrant states; never fabricate a per-day count.
    qp = d.get("quantity_periods") or []
    if qp or d.get("nacin_dobe"):
        lines.append("**Obdobje in največja količina:**")
        if d.get("nacin_dobe"):
            lines.append(f"- Način določitve dobe: {d['nacin_dobe']}")
        for q in qp:
            extra = []
            if q.get("trajanje"):
                extra.append(f"obdobje {q['trajanje']} {q.get('trajanje_enota', '').lower()}".strip())
            if q.get("max_kolicina"):
                extra.append(f"do {q['max_kolicina']} {q.get('kolicina_enota', '').lower()}".strip())
            if q.get("na_dan_max"):
                lo, hi = q.get("na_dan_min", ""), q["na_dan_max"]
                nd = hi if (lo in ("", hi)) else f"{lo}–{hi}"
                extra.append(f"{nd} na dan")
            if q.get("starost_od") or q.get("starost_do"):
                extra.append(f"starost {q.get('starost_od', '')}–{q.get('starost_do', '')} let")
            seg = q.get("opis") or ""
            tail = f" ({'; '.join(extra)})" if extra else ""
            lines.append(f"- {seg}{tail}".rstrip() if (seg or tail) else "- (ni podatka)")
        if not qp and d.get("doba_mode") == "durable":
            lines.append("- Trajnostna doba (življenjska) — količina praviloma 1; "
                         "posebna količina na obdobje ni določena.")
        if d.get("st_kosov_pakiranje"):
            lines.append(f"- Kosov v pakiranju: {d['st_kosov_pakiranje']}")
        if d.get("dvojna_kolicina"):
            lines.append(f"- Dvojna količina možna: {d['dvojna_kolicina']}")
        lines.append("")

    # (3) Same group — devices prescribable alongside (same group MINUS mutually-exclusive).
    g = d.get("group")
    peers = d.get("group_peers") or []
    if g or peers:
        lines.append("**Ista skupina — kaj se lahko predpiše skupaj:**")
        gg = g or {}
        path = [f"{s['sifra']} {s['naziv']}" for s in gg.get("skupine", [])]
        if gg.get("podgruca_naziv"):
            # Finest EHR family — prefer it over the (often repetitive) formal podskupine list.
            path.append(f"{gg.get('gruca_naziv', '')} › {gg['podgruca_naziv']}".strip(" ›"))
        else:
            # Distinct leaf podskupine (dedup — a device is often filed under several pod1 codes).
            for seg in dict.fromkeys(s.get("pod2_naziv") or s.get("pod1_naziv")
                                     for s in gg.get("podskupine", [])):
                if seg:
                    path.append(seg)
        if path:
            lines.append(f"- Skupina: {' | '.join(dict.fromkeys(path))}")
        if peers:
            lines.append("- Drugi pripomočki v isti skupini "
                         "(⚠ = se z izbranim medsebojno izključuje, ne hkrati):")
            for pr in peers:
                mark = "  ⚠ se izključuje" if pr.get("excluded") else ""
                lines.append(f"    - {pr['sifra']} {pr.get('ime', '')}{mark}")
            if d.get("group_peers_overflow"):
                lines.append(f"    - _… in še {d['group_peers_overflow']} drugih pripomočkov_")
        else:
            lines.append("- V isti (pod)skupini ni drugih pripomočkov.")
        lines.append("")

    # (optional) article system components — distinct names, capped (product lists can be long).
    sysrows = d.get("systems") or []
    if sysrows:
        names = list(dict.fromkeys(
            (s.get("sistem_naziv") or s.get("artikel_naziv") or "").strip()
            for s in sysrows))
        names = [n for n in names if n]
        if names:
            lines.append("**Sistem / sestavni artikli:**")
            for n in names[:_LIST_CAP]:
                lines.append(f"- {n}")
            if len(names) > _LIST_CAP:
                lines.append(f"- _… in še {len(names) - _LIST_CAP} artiklov_")
            lines.append("")

    return lines


def _format_prescribing_md(device, query: str, data: dict) -> str:
    """Markdown for the focused prescribing tool: the three sections for a single device."""
    if not data.get("found"):
        return (f"V seznamu medicinskih pripomočkov ni zadetka za \"{query}\". "
                f"{data.get('error', '')}").strip()
    lines = [f"**Predpis pripomočka — {device.get('sifra', '')} {device.get('ime', '')}**"
             f"  ·  stanje {DATUM}", ""]
    body = _render_prescribing(device)
    if body:
        lines += body
    else:
        lines.append("_Za ta pripomoček ni strukturiranih podatkov o predpisu v šifrantu VrsteMTP. "
                     "Uporabite `get_mtp` za osnovne podatke._")
        lines.append("")
    url = device.get("source_url") or SOURCE_URL
    lines.append(f"**Vir:** [{_citation()}]({url})" if url else f"**Vir:** {_citation()}")
    lines += ["",
              "_Strukturirani podatki o predpisu (predpisovalec, obdobje/količina, skupina) so iz "
              "ZZZS \"Vsi šifranti\" (VrsteMTP, TrajnostneDobeMTP, SkupineMTP …); osnovni opis iz "
              "Seznama MP. Kratice (IOZ, DMS, *TD) so pojasnjene v viru. Preveri morebitno novejšo "
              "objavo._"]
    return "\n".join(lines)


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
        # Structured prescribing enrichment (who / period+quantity / same-group peers). Returns []
        # for pre-enrichment catalogs, so this is a no-op there.
        pres = _render_prescribing(d)
        if pres:
            lines += pres
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
