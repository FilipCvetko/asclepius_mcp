"""Structured ZZZS obračun ŠIFRANTI lookup (authoritative, exact).

Backed by data/sifrant_catalog.json — the full current set of billing šifranti parsed from
the ZZZS "Vsi šifranti" XML (built by scripts/fetch_sifranti.py + build_sifrant_catalog.py).
Unlike the text-RAG tools (egradiva/obracun), this returns the EXACT catalog row for a code
(SiSto / SiDiag / SifraVrsMTP / …) or finds the code from a description — across ~96 code-list
šifranti (services, MKB, MTP, therapeutic/diagnostic procedures, document types, …).

This is the priority/authoritative source when reporting obračun šifre. It also exposes the
publication's change summary (KomVsebSprememb) as a changelog.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastmcp.tools.tool import ToolResult

from paths import data_path

CATALOG_FILE = Path(__file__).parent / "data" / "sifrant_catalog.json"
SOURCE_LABEL = "ZZZS Vsi šifranti (obračun)"

# Module state
OBJAVA: Dict[str, Any] = {}
SIFRANTI: Dict[str, Any] = {}                 # oznaka -> šifrant meta (+ zapisi for code-lists)
_ROWS: List[Dict[str, Any]] = []              # flat: {key, oznaka, naziv, sifra, opis, row}
_BY_CODE: Dict[str, List[int]] = {}           # normalized code -> indices into _ROWS
_VEC: Optional[TfidfVectorizer] = None
_MAT = None

# Dejavnost (specialty/practice) billability index — derived in build_sifrant_catalog.py.
# Lets a clinician/LLM scope šifra answers to what a given dejavnost may actually bill.
DEJAVNOSTI: Dict[str, Dict[str, str]] = {}        # vrsDej -> {naziv, ozn}
PODVRSTE: Dict[str, str] = {}                      # podDej -> opis
PODVRSTA_TO_VRSTA: Dict[str, List[str]] = {}       # podDej -> [vrsDej]
ALLOWED_BY_DEJAVNOST: Dict[str, List[str]] = {}    # vrsDej -> [SiSto] (current, sorted)
SEZNAMI_BY_DEJAVNOST: Dict[str, List[list]] = {}   # vrsDej -> [[sezCode, sezName], …]
DEJAVNOSTI_BY_SIFRA: Dict[str, List[str]] = {}     # SiSto -> [vrsDej] (reverse)
DEJAVNOST_CONTEXT: Dict[str, Dict[str, str]] = {}  # vrsDej -> {title, text} curated orientation
SERVICE_OZNAKA = "15"                              # šifrant 15 = storitve (the dejavnost-scoped set)
CONTEXT_DIR = Path(__file__).parent / "data" / "dejavnost_context"

_CODE_TOKEN = re.compile(r"^[A-Za-z0-9]{1,7}(?:\.[A-Za-z0-9]+)?$")


def _norm(code: str) -> str:
    return (code or "").strip().upper()


def _validity_rank(row: Dict[str, Any]) -> tuple:
    """Rank a row by currency: still-open (no VeljaDo) beats closed, then later VeljaOd."""
    return (0 if row.get("VeljaDo") else 1, row.get("VeljaOd", ""))


def initialize_sifranti() -> None:
    """Load the šifrant catalog, flatten code-list rows, build code + TF-IDF indexes."""
    global OBJAVA, SIFRANTI, _ROWS, _BY_CODE, _VEC, _MAT
    global DEJAVNOSTI, PODVRSTE, PODVRSTA_TO_VRSTA, ALLOWED_BY_DEJAVNOST
    global SEZNAMI_BY_DEJAVNOST, DEJAVNOSTI_BY_SIFRA
    catalog_file = data_path("sifrant_catalog.json")   # volume copy wins, else baked
    if not catalog_file.exists():
        print("WARNING: šifrant catalog not found at", catalog_file)
        return
    data = json.loads(catalog_file.read_text(encoding="utf-8"))
    OBJAVA = data.get("objava", {})
    SIFRANTI = data.get("sifranti", {})

    # Dejavnost billability index (absent in older catalogs → feature simply off)
    di = data.get("dejavnost_index", {})
    DEJAVNOSTI = di.get("dejavnosti", {})
    PODVRSTE = di.get("podvrste", {})
    PODVRSTA_TO_VRSTA = di.get("podvrsta_to_vrsta", {})
    ALLOWED_BY_DEJAVNOST = di.get("allowed_by_dejavnost", {})
    SEZNAMI_BY_DEJAVNOST = di.get("seznami_by_dejavnost", {})
    DEJAVNOSTI_BY_SIFRA = di.get("dejavnosti_by_sifra", {})
    _load_dejavnost_context()

    _ROWS = []
    _BY_CODE = {}
    corpus = []
    for key, sif in SIFRANTI.items():
        if not sif.get("is_code_list"):
            continue
        sfield, ofield = sif.get("sifra_field"), sif.get("opis_field")
        # A šifra can recur across validity periods (VeljaOd/VeljaDo) — keep the most
        # current row per (šifrant, šifra) so lookups don't return duplicates.
        canonical: Dict[str, Dict[str, Any]] = {}
        no_code: List[Dict[str, Any]] = []
        for row in sif.get("zapisi", []):
            sifra = row.get(sfield, "") if sfield else ""
            if not sifra:
                no_code.append(row)
                continue
            nc = _norm(sifra)
            if nc not in canonical or _validity_rank(row) > _validity_rank(canonical[nc]):
                canonical[nc] = row
        for row in list(canonical.values()) + no_code:
            sifra = row.get(sfield, "") if sfield else ""
            opis = row.get(ofield, "") if ofield else ""
            idx = len(_ROWS)
            _ROWS.append({"key": key, "oznaka": sif.get("oznaka"),
                          "naziv": sif.get("naziv"), "naziv_human": sif.get("naziv_human"),
                          "sifra": sifra, "opis": opis, "row": row})
            if sifra:
                _BY_CODE.setdefault(_norm(sifra), []).append(idx)
            # description corpus: code + every textual field value
            corpus.append(f"{sifra} {' '.join(str(v) for v in row.values())}")

    if corpus:
        _VEC = TfidfVectorizer()
        _MAT = _VEC.fit_transform(corpus)
    print(f"Loaded šifrant catalog: objava {OBJAVA.get('leto')}/{OBJAVA.get('zap_st')} "
          f"({OBJAVA.get('datum')}), {len(SIFRANTI)} šifrantov, {len(_ROWS):,} code rows; "
          f"dejavnost index: {len(DEJAVNOSTI)} dejavnosti, "
          f"{len(DEJAVNOSTI_BY_SIFRA):,} storitev mapiranih")


# ── šifrant resolution ────────────────────────────────────────────────────────

def _resolve_sifrant(spec: str) -> Optional[set]:
    """Map a šifrant filter (oznaka like '15', or a name substring like 'storitev')
    to the set of catalog keys it selects, or None for no filter."""
    if not spec:
        return None
    spec = spec.strip()
    keys = set()
    if spec in SIFRANTI:
        keys.add(spec)
    low = spec.lower()
    for key, sif in SIFRANTI.items():
        if str(sif.get("oznaka")) == spec or low in (sif.get("naziv", "") or "").lower() \
                or low in (sif.get("naziv_human", "") or "").lower():
            keys.add(key)
    return keys or None


# ── Dejavnost (specialty/practice) resolution ─────────────────────────────────

def _load_dejavnost_context() -> None:
    """Load curated per-dejavnost orientation notes (unstructured) from data/dejavnost_context."""
    global DEJAVNOST_CONTEXT
    DEJAVNOST_CONTEXT = {}
    idx = CONTEXT_DIR / "index.json"
    if not idx.exists():
        return
    try:
        meta = json.loads(idx.read_text(encoding="utf-8")).get("notes", {})
    except Exception as e:
        print(f"WARNING: dejavnost context index unreadable: {e}")
        return
    for vrs, info in meta.items():
        note = CONTEXT_DIR / info.get("note", "")
        if note.exists():
            DEJAVNOST_CONTEXT[vrs] = {"title": info.get("title", ""),
                                      "text": note.read_text(encoding="utf-8").strip()}
    if DEJAVNOST_CONTEXT:
        print(f"Loaded dejavnost context notes: {len(DEJAVNOST_CONTEXT)}")


def _resolve_dejavnost(spec: str) -> Dict[str, Any]:
    """Resolve a dejavnost spec to the vrsta-dejavnosti it selects, generically.

    `spec` may be a vrsta code ('302'), a podvrsta code ('001'), or a name substring
    ('družinska', 'kardiolog', 'zobozdrav', 'pediatr'). Returns:
      {"matched": [(vrsDej, naziv), …], "allowed": set(SiSto), "labels": "…"} or
      {"matched": [], "suggestions": [(code, naziv), …]} when nothing resolves.
    """
    spec = (spec or "").strip()
    if not spec or not DEJAVNOSTI:
        return {"matched": [], "allowed": set(), "labels": ""}
    low = spec.lower()
    matched: Dict[str, str] = {}

    # 1. exact vrsta code
    if spec in DEJAVNOSTI:
        matched[spec] = DEJAVNOSTI[spec].get("naziv", "")
    # 2. exact podvrsta code → its vrsta(s)
    for vrs in PODVRSTA_TO_VRSTA.get(spec, []):
        if vrs in DEJAVNOSTI:
            matched.setdefault(vrs, DEJAVNOSTI[vrs].get("naziv", ""))
    # 3. name substring over vrsta naziv + ozn
    if not matched:
        for code, meta in DEJAVNOSTI.items():
            if low in (meta.get("naziv", "") or "").lower() \
                    or low in (meta.get("ozn", "") or "").lower():
                matched[code] = meta.get("naziv", "")
    # 4. name substring over podvrsta names → their vrste
    if not matched:
        for pod, opis in PODVRSTE.items():
            if low in (opis or "").lower():
                for vrs in PODVRSTA_TO_VRSTA.get(pod, []):
                    if vrs in DEJAVNOSTI:
                        matched.setdefault(vrs, DEJAVNOSTI[vrs].get("naziv", ""))

    if not matched:
        # offer a few close-ish suggestions (any token overlap) for the formatter
        sugg = [(c, m.get("naziv", "")) for c, m in DEJAVNOSTI.items()
                if any(t in (m.get("naziv", "") or "").lower() for t in low.split())][:8]
        return {"matched": [], "allowed": set(), "labels": "", "suggestions": sugg}

    allowed: set = set()
    for vrs in matched:
        allowed.update(ALLOWED_BY_DEJAVNOST.get(vrs, ()))
    labels = "; ".join(f"{c} {n}" for c, n in matched.items())
    return {"matched": list(matched.items()), "allowed": allowed, "labels": labels}


def _billing_dejavnosti(sifra: str) -> List[List[str]]:
    """[(vrsDej, naziv), …] of dejavnosti that may bill this service code (reverse index)."""
    return [[c, DEJAVNOSTI.get(c, {}).get("naziv", "")]
            for c in DEJAVNOSTI_BY_SIFRA.get(_norm(sifra), [])]


# ── Lookup ────────────────────────────────────────────────────────────────────

_LIST_CAP = 60   # max service rows returned in dejavnost "list mode"


def _lookup_sifra(query: str, sifrant: Optional[str] = None, n_results: int = 10,
                  dejavnost: Optional[str] = None) -> ToolResult:
    """Bidirectional šifrant lookup: code → exact row(s), or description → matching codes.

    `sifrant` optionally restricts to one šifrant (by oznaka '15' or a name substring).
    `dejavnost` optionally scopes service answers to what a given specialty/practice may bill
    (vrsta code like '302', podvrsta code, or a name substring like 'družinska'/'kardiolog')."""
    if not _ROWS:
        err = {"found": False, "error": "Šifrant catalog not loaded"}
        return ToolResult(content=_format_md(err), structured_content=err)
    query = (query or "").strip()

    # Resolve the optional dejavnost filter up front
    dej = _resolve_dejavnost(dejavnost) if dejavnost else None
    if dejavnost and not dej.get("matched"):
        err = {"found": False, "error": "Neznana dejavnost", "query": query,
               "dejavnost_query": dejavnost, "suggestions": dej.get("suggestions", [])}
        return ToolResult(content=_format_md(err), structured_content=err)
    dej_allowed: Optional[set] = dej["allowed"] if dej else None

    # List mode: dejavnost given + no real query → list its billable services
    if dej and query in ("", "*"):
        return _list_dejavnost_storitve(dej)

    if not query:
        err = {"found": False, "error": "Query must not be empty"}
        return ToolResult(content=_format_md(err), structured_content=err)

    allow = _resolve_sifrant(sifrant) if sifrant else None
    # A pure code query (e.g. "AA001") is answered even when filtering, with a billable ✓/✗;
    # a description/list query is filtered down to the dejavnost's billable services.
    q_is_code = bool(_BY_CODE.get(_norm(query)))

    def _keep(r: Dict[str, Any]) -> bool:
        if allow and r["key"] not in allow:
            return False
        # filter only service rows on a description query; keep codes & non-services
        if dej_allowed is not None and not q_is_code \
                and r["oznaka"] == SERVICE_OZNAKA and r["sifra"] not in dej_allowed:
            return False
        return True

    results: List[Dict[str, Any]] = []
    seen: set = set()

    # 1. Exact code: the whole query, then each whitespace token
    candidates = [query] + query.split()
    for tok in candidates:
        for idx in _BY_CODE.get(_norm(tok), []):
            if idx in seen:
                continue
            r = _ROWS[idx]
            if not _keep(r):
                continue
            results.append(_mk_result(r, match="exact-code", dej_allowed=dej_allowed))
            seen.add(idx)

    # 2. Prefix code match (e.g. "AA0" or an MKB stem "C50") if room
    if len(results) < n_results:
        q = _norm(query)
        if 1 <= len(q) <= 7:
            for code, idxs in _BY_CODE.items():
                if code.startswith(q) and code != q:
                    for idx in idxs:
                        if idx in seen or len(results) >= n_results:
                            break
                        r = _ROWS[idx]
                        if not _keep(r):
                            continue
                        results.append(_mk_result(r, match="prefix-code", dej_allowed=dej_allowed))
                        seen.add(idx)

    # 3. Description search (TF-IDF) to fill remaining slots
    if len(results) < n_results and _VEC is not None:
        sims = cosine_similarity(_VEC.transform([query]), _MAT).flatten()
        for idx in sims.argsort()[::-1]:
            if sims[idx] <= 0.05 or len(results) >= n_results:
                break
            if idx in seen:
                continue
            r = _ROWS[idx]
            if not _keep(r):
                continue
            results.append(_mk_result(r, match="opis", relevance=round(float(sims[idx]), 4),
                                      dej_allowed=dej_allowed))
            seen.add(idx)

    if not results:
        err = {"found": False, "error": "No matching šifra/opis", "query": query,
               "dejavnost": _dej_meta(dej) if dej else None}
        return ToolResult(content=_format_md(err), structured_content=err)
    data = {"found": True, "count": len(results), "query": query,
            "objava": OBJAVA, "results": results, "dejavnost": _dej_meta(dej) if dej else None}
    return ToolResult(content=_format_md(data), structured_content=data)


def _dej_meta(dej: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize the active dejavnost filter for the result payload."""
    codes = [c for c, _ in dej["matched"]]
    seznami: List[list] = []
    context: List[Dict[str, str]] = []
    for c in codes:
        seznami.extend(SEZNAMI_BY_DEJAVNOST.get(c, []))
        if c in DEJAVNOST_CONTEXT:
            context.append({"vrsDej": c, **DEJAVNOST_CONTEXT[c]})
    return {"matched": dej["matched"], "labels": dej["labels"],
            "allowed_count": len(dej["allowed"]), "seznami": seznami, "context": context}


def _list_dejavnost_storitve(dej: Dict[str, Any]) -> ToolResult:
    """List the services a dejavnost may bill (its allowed SiSto set), capped, with totals."""
    codes = sorted(dej["allowed"])
    results = []
    for c in codes[:_LIST_CAP]:
        for idx in _BY_CODE.get(_norm(c), []):
            r = _ROWS[idx]
            if r["oznaka"] == SERVICE_OZNAKA:
                results.append(_mk_result(r, match="dejavnost-list", dej_allowed=dej["allowed"]))
                break
    data = {"found": bool(results), "count": len(results), "total": len(codes),
            "query": "", "objava": OBJAVA, "results": results,
            "dejavnost": _dej_meta(dej), "list_mode": True}
    if not results:
        data["error"] = "Za to dejavnost ni mapiranih obračunskih storitev"
    return ToolResult(content=_format_md(data), structured_content=data)


def _mk_result(r: Dict[str, Any], match: str, relevance: Optional[float] = None,
               dej_allowed: Optional[set] = None) -> Dict[str, Any]:
    is_service = r["oznaka"] == SERVICE_OZNAKA
    res = {
        "sifrant_oznaka": r["oznaka"],
        "sifrant_naziv": r["naziv"],
        "sifrant_naziv_human": r["naziv_human"],
        "sifra": r["sifra"],
        "opis": r["opis"],
        "polja": r["row"],
        "match": match,
        "relevance": relevance,
        "citation": _citation(),
        "source_url": OBJAVA.get("source_url", ""),
    }
    # Always annotate a service with the dejavnosti that may bill it (well-informed answer)
    if is_service:
        res["dejavnosti_billing"] = _billing_dejavnosti(r["sifra"])
        if dej_allowed is not None:
            res["billable_in_filter"] = r["sifra"] in dej_allowed
    return res


def _citation() -> str:
    o = OBJAVA
    return (f"{SOURCE_LABEL}, objava {o.get('leto')}/{o.get('zap_st')} "
            f"({o.get('datum')})")


# ── Changelog ─────────────────────────────────────────────────────────────────

def _sifrant_changelog() -> ToolResult:
    """Return the change summary (KomVsebSprememb) of the current šifrant publication."""
    if not OBJAVA:
        err = {"found": False, "error": "Šifrant catalog not loaded"}
        return ToolResult(content="Šifrant katalog ni naložen.", structured_content=err)
    o = OBJAVA
    lines = [
        f"**Spremembe šifrantov — objava {o.get('leto')}/{o.get('zap_st')}** "
        f"({o.get('datum')})", "",
        "> " + (o.get("komentar") or "(brez navedene vsebine sprememb)"), "",
        f"**Vir:** [{_citation()}]({o.get('source_url','')})", "",
        f"_Skupaj {len(SIFRANTI)} šifrantov v tej objavi. To je zadnja prevzeta objava "
        f"'Vsi šifranti'._",
    ]
    data = {"found": True, "leto": o.get("leto"), "zap_st": o.get("zap_st"),
            "datum": o.get("datum"), "komentar": o.get("komentar"),
            "stevilo_sifrantov": len(SIFRANTI), "source_url": o.get("source_url", "")}
    return ToolResult(content="\n".join(lines), structured_content=data)


# ── Formatting ────────────────────────────────────────────────────────────────

_PRIMARY = set()  # fields shown via the labeled Šifra/Opis header — skip in the detail list


def _fmt_dej_list(pairs: List[list], cap: int = 8) -> str:
    """Render a [(code, naziv), …] dejavnosti list, capped with a '+N' tail."""
    shown = [f"{c} {n}".strip() for c, n in pairs[:cap]]
    if len(pairs) > cap:
        shown.append(f"… (+{len(pairs) - cap})")
    return "; ".join(shown)


def _format_md(data: dict) -> str:
    if not data.get("found"):
        msg = (f"V šifrantih ni zadetka za \"{data.get('query','')}\". "
               f"{data.get('error','')}").strip()
        sugg = data.get("suggestions")
        if data.get("dejavnost_query") and sugg is not None:
            opts = ", ".join(f"{c} {n}" for c, n in sugg) or "—"
            msg += (f"\n\nDejavnosti \"{data['dejavnost_query']}\" ni bilo mogoče razrešiti. "
                    f"Morda ste mislili: {opts}.")
        return msg
    o = data.get("objava", {})
    header = f"**ZZZS šifranti (obračun)** | {data['count']} zadetkov"
    if data.get("list_mode"):
        header = (f"**ZZZS šifranti (obračun)** | {data['count']} prikazanih od "
                  f"{data.get('total', data['count'])} obračunskih storitev")
    else:
        header += f" za \"{data['query']}\""
    header += f"  ·  objava {o.get('leto')}/{o.get('zap_st')} ({o.get('datum')})"
    lines = [header, ""]

    dej = data.get("dejavnost")
    if dej:
        lines.append(f"**Dejavnost:** {dej.get('labels','')}  ·  "
                     f"{dej.get('allowed_count', 0)} obračunljivih storitev")
        seznami = dej.get("seznami") or []
        if seznami:
            sez = "; ".join(f"{c} {n}".strip() for c, n in seznami[:10])
            lines.append(f"- **Seznami storitev dejavnosti:** {sez}")
        lines.append("")
        for ctx in dej.get("context") or []:
            lines += [f"> **Kontekst dejavnosti — {ctx.get('title','')}**", ">"]
            lines += [("> " + ln) if ln else ">" for ln in ctx.get("text", "").splitlines()]
            lines.append("")

    for r in data["results"]:
        lines += ["---", ""]
        oz = r.get("sifrant_oznaka")
        naziv = r.get("sifrant_naziv_human") or r.get("sifrant_naziv") or ""
        lines.append(f"### Šifrant {oz} — {naziv}")
        lines.append("")
        lines.append(f"- **Šifra:** `{r.get('sifra','')}`")
        if r.get("opis"):
            lines.append(f"- **Opis:** {r['opis']}")
        # remaining exact fields (verbatim), skipping the code + opis already shown
        sfield_vals = r.get("polja", {})
        shown = {r.get("sifra"), r.get("opis")}
        extras = [f"**{k}:** {v}" for k, v in sfield_vals.items() if v not in shown]
        if extras:
            lines.append("- " + " · ".join(extras))
        # dejavnost annotations (services only)
        if r.get("billable_in_filter") is not None and dej:
            mark = "✓ obračunljivo" if r["billable_in_filter"] else "✗ ni na seznamu te dejavnosti"
            lines.append(f"- **V dejavnosti {dej.get('labels','')}:** {mark}")
        billing = r.get("dejavnosti_billing")
        if billing:
            lines.append(f"- **Obračunljivo v dejavnostih:** {_fmt_dej_list(billing)}")
        elif r.get("sifrant_oznaka") == SERVICE_OZNAKA:
            lines.append("- **Obračunljivo v dejavnostih:** (ni v mapiranju dejavnost↔storitev)")
        if r.get("relevance") is not None:
            lines.append(f"- *relevantnost:* {r['relevance']}")
        lines.append("")
        lines.append(f"**Vir:** [{r.get('citation')}]({r.get('source_url','')})")
        lines.append("")
    lines += ["---", "",
              "_Vir je ZZZS 'Vsi šifranti' (uradni obračunski šifranti). To je avtoritativni "
              "vir za šifre obračuna — pri sklicu preveri veljavnost (VeljaOd/VeljaDo). Mapiranje "
              "dejavnost↔storitev izhaja iz šifrantov K1.1/K1.2 (dovoljene storitve/seznami po "
              "vrstah dejavnosti)._"]
    return "\n".join(lines)


if __name__ == "__main__":
    initialize_sifranti()
    # (query, sifrant, dejavnost)
    cases = [
        ("AA001", None, None), ("prvi pregled", "15", None), ("B83B", None, None),
        ("voziček na ročni pogon", None, None), ("26", "VrsteDokumentov", None),
        ("laboratorij", None, "družinska medicina"),   # filter: only FM-billable services
        ("", None, "kardiologija"),                     # list mode for a specialty
        ("AA001", None, "zobozdravstvo"),               # code billable ✓/✗ in a dejavnost
        ("prvi pregled", None, "pediatrija"),           # filtered description query
    ]
    for q, sf, dj in cases:
        print("\n" + "=" * 70, f"\nQUERY {q!r} (sifrant={sf}, dejavnost={dj})")
        sc = _lookup_sifra(q, sifrant=sf, n_results=4, dejavnost=dj).structured_content
        print("found:", sc.get("found"), "count:", sc.get("count"),
              "total:", sc.get("total"))
        d = sc.get("dejavnost")
        if d:
            print("  dejavnost:", d.get("labels"), "| allowed:", d.get("allowed_count"))
        for r in sc.get("results", []):
            extra = ""
            if r.get("billable_in_filter") is not None:
                extra = " [✓]" if r["billable_in_filter"] else " [✗]"
            bill = r.get("dejavnosti_billing")
            bn = f" bill={len(bill)}" if bill else ""
            print(f"  [{r['match']}] šif {r['sifrant_oznaka']} {r['sifra']:8s} "
                  f"{(r['opis'] or '')[:48]}{extra}{bn}")
        if not sc.get("found"):
            print("   ->", sc.get("error"))
    print("\n--- changelog ---")
    print(_sifrant_changelog().content[:200])
