#!/usr/bin/env python3
"""Build data/sifrant_catalog.json from the ZZZS "Vsi šifranti" XML.

The XML (<VSI_SIFRANTI>, ns http://zzzs.si/b2b/Izdatki) holds the full current set of
obračun šifranti: `PodatkiObjave` (publish year/number/date + change summary) and ~190
šifrant "containers" grouped under SkupinaSifrantov_* sections. Each container carries an
`OznakaSifranta` (the šifrant number a provider references, e.g. 15 = storitve, 50.1 = MKB)
and many `<Zapis>` rows (the code entries).

Containers come in two shapes:
  • code lists  — a row has a code field + a description/name field (Opis*/Naziv*/SloNaziv*).
                  We store every row verbatim → these are the lookup targets.
  • relational/control tables (mostly the K* šifranti) — only foreign-key codes + validity
                  dates, no description. We keep their metadata (oznaka, fields, count) so the
                  full set is documented, but skip the bulk rows to bound size/RAM.

Stream-parsed with iterparse (the XML is ~60 MB). Output is loaded by sifranti.py.
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
SIFRANTI_DIR = DATA / "sifranti"
OUT = DATA / "sifrant_catalog.json"

# Public download URL pattern for citation (filled with leto/zap of the parsed objava).
DL_URL = (
    "https://partner.zzzs.si/?id=742"
    "&tx_agzzzsapi_zzzs%5Baction%5D=downloadfilesifranti"
    "&tx_agzzzsapi_zzzs%5Bcontroller%5D=Izvajalci&type=1249058994"
    "&leto_objave={leto}&zap_st_objave={zap}&oznaka_sifranta=%20"
    "&vrsta_datoteke=1&naziv_datoteke=SifrantiXML_Vsi_objava&tip_datoteke=zip"
)


def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


_DESC_RE = re.compile(
    r"^(Kratek|Kratki|Dolg|Dolgi|Krajsi|Krajši|Daljsi|Daljši|Slo)?(Opis|Naziv)(.*)$"
)


def _entity_tokens(sifra_field: str):
    """The entity suffix a description field should carry: the full code-field name and
    the name minus a leading qualifier (Sifra/Si/Oz). ZZZS uses both conventions
    (DolgOpis**SiSto** but SloNaziv**Diag** for SiDiag), so we accept either."""
    if not sifra_field:
        return {""}
    toks = {sifra_field}
    for p in ("Sifra", "Si", "Oz"):
        if sifra_field.startswith(p):
            toks.add(sifra_field[len(p):])
            break
    return toks


def _desc_score(field: str, tokens: set) -> int:
    """Score a field as THE entity's name/description (0 = a sub-attribute or not a desc).

    A real description is Opis/Naziv (optionally qualified Kratek/Dolg/Slo…) whose suffix
    is empty (bare 'Opis') or names this entity — so 'OpisKadrNorma' on a service row,
    whose suffix is 'KadrNorma', is correctly rejected.
    """
    m = _DESC_RE.match(field)
    if not m:
        return 0
    qual, kind, suffix = (m.group(1) or ""), m.group(2), m.group(3)
    if suffix and suffix not in tokens:
        return 0
    score = 3 if kind == "Naziv" else 2          # Naziv slightly preferred over Opis
    if qual in ("Dolg", "Dolgi", "Daljsi", "Daljši"):
        score += 2                                # longer description preferred
    elif qual in ("Kratek", "Kratki", "Krajsi", "Krajši"):
        score += 0
    else:
        score += 1                                # bare / Slo
    return score


def _pick_fields(field_order):
    """Choose the code field (first) and the best entity-description field, if any."""
    sifra_field = field_order[0] if field_order else None
    tokens = _entity_tokens(sifra_field)
    opis_field, best = None, 0
    for f in field_order:
        if f == sifra_field:
            continue
        s = _desc_score(f, tokens)
        if s > best:
            opis_field, best = f, s
    return sifra_field, opis_field


def _humanize(name: str) -> str:
    """CamelCase container name → spaced label (rough; oznaka is the authoritative id)."""
    s = re.sub(r"(?<=[a-zšžč])(?=[A-ZŠŽČ])", " ", name)
    s = re.sub(r"(?<=[A-ZŠŽČ])(?=[A-ZŠŽČ][a-zšžč])", " ", s)
    return s


# Relational mapping containers whose rows are normally dropped (is_code_list=False) but which
# we need to derive the dejavnost↔storitev billability index. Captured during the same parse.
WANT_MAPPING = {"K1.1", "K1.2", "S3", "2.3"}


def build(xml_path: Path, leto: int, zap: int) -> dict:
    objava = {"leto": leto, "zap_st": zap, "datum": "", "komentar": "",
              "source_url": DL_URL.format(leto=leto, zap=zap), "xml_file": xml_path.name}
    sifranti = {}
    mapping_rows: dict = {}            # oznaka (in WANT_MAPPING) -> its raw Zapis rows

    # iterparse with a tag stack. A šifrant container is, robustly, the element that has
    # an <OznakaSifranta> child — independent of how deep its group wrapper nests. We anchor
    # on that: when OznakaSifranta closes, its parent (stack top) IS the container; its
    # <Zapis>/<SifrantVeljaOd> siblings sit at the same depth; the container finalizes when
    # that element itself closes.
    stack = []
    cur = None            # the container dict currently being filled
    cur_depth = -1        # len(stack) while inside the current container's body
    in_objava = False

    ctx = ET.iterparse(str(xml_path), events=("start", "end"))
    for ev, el in ctx:
        lt = _local(el.tag)
        if ev == "start":
            if lt == "PodatkiObjave":
                in_objava = True
            stack.append(lt)
            continue

        # ── end event ──
        stack.pop()

        if lt == "PodatkiObjave":
            in_objava = False
            el.clear()
            continue
        if in_objava:
            txt = (el.text or "").strip()
            if lt == "DatumObjave":
                objava["datum"] = txt
            elif lt == "KomVsebSprememb":
                objava["komentar"] = txt
            continue

        if lt == "OznakaSifranta":
            # the container is now the stack top; open a fresh accumulator for it
            cur = {"oznaka": (el.text or "").strip(),
                   "naziv": stack[-1] if stack else "",
                   "naziv_human": _humanize(stack[-1] if stack else ""),
                   "valja_od": "", "field_order": [], "_seen": set(), "zapisi": []}
            cur_depth = len(stack)
        elif cur is not None and len(stack) == cur_depth and lt == "SifrantVeljaOd":
            cur["valja_od"] = (el.text or "").strip()
        elif cur is not None and len(stack) == cur_depth and lt == "Zapis":
            row = {}
            for child in el:
                cl = _local(child.tag)
                val = (child.text or "").strip()
                if cl not in cur["_seen"]:
                    cur["field_order"].append(cl)
                    cur["_seen"].add(cl)
                if val:
                    row[cl] = val
            cur["zapisi"].append(row)
            el.clear()
        elif cur is not None and len(stack) == cur_depth - 1:
            # the container element itself just closed → finalize
            if cur["oznaka"] in WANT_MAPPING:
                # stash the raw rows before _finalize drops them (relational tables keep none)
                mapping_rows[cur["oznaka"]] = cur["zapisi"]
            _finalize(cur, sifranti)
            cur = None
            cur_depth = -1
            el.clear()

    dejavnost_index = _build_dejavnost_index(sifranti, mapping_rows)
    return {"objava": objava, "sifranti": sifranti, "dejavnost_index": dejavnost_index}


def _by_oznaka(sifranti: dict, oznaka: str) -> dict:
    """Return the (first) šifrant entry with this oznaka, or {} — keys may be dup-suffixed."""
    for v in sifranti.values():
        if str(v.get("oznaka")) == oznaka:
            return v
    return {}


def _current(rows, kveljado="VeljaDo"):
    """Keep only rows that are still open (no end date) — the current/valid set."""
    return [r for r in rows if not r.get(kveljado)]


def _build_dejavnost_index(sifranti: dict, mapping_rows: dict) -> dict:
    """Derive the dejavnost↔storitev billability index from the mapping tables.

    allowed_by_dejavnost[vrsDej] = direct codes (K1.1) ∪ codes of every service-list the dejavnost
    may use (K1.2 expanded through S3). Keyed at the vrsta-dejavnosti level (SifraVrsZdrDej), which
    is what the K1.* tables key on and what a clinician means by "my specialty/practice".
    """
    # dejavnosti (vrste) + their human names, from code-list 2.1
    dejavnosti: dict = {}
    for r in _current(_by_oznaka(sifranti, "2.1").get("zapisi", [])):
        code = r.get("SifraVrsZdrDej")
        if code:
            dejavnosti[code] = {"naziv": r.get("OpisVrsZdrDej", ""),
                                "ozn": r.get("OznVrsZdrDej", "")}
    # podvrste names (2.2) + podvrsta→vrsta links (2.3) for name resolution
    podvrste = {r.get("SifraPodZdrDej"): r.get("OpisPodZdrDej", "")
                for r in _current(_by_oznaka(sifranti, "2.2").get("zapisi", []))
                if r.get("SifraPodZdrDej")}
    podvrsta_to_vrsta: dict = {}
    for r in _current(mapping_rows.get("2.3", [])):
        pod, vrs = r.get("SifraPodZdrDej"), r.get("SifraVrsZdrDej")
        if pod and vrs:
            podvrsta_to_vrsta.setdefault(pod, set()).add(vrs)

    # seznam → set(SiSto), from S3 (PripadnostStoritevSeznamom)
    sez_to_sisto: dict = {}
    for r in _current(mapping_rows.get("S3", [])):
        sez, sisto = r.get("SifraSezSto"), r.get("SiSto")
        if sez and sisto:
            sez_to_sisto.setdefault(sez, set()).add(sisto)
    # seznam names from code-list S1
    sez_naziv = {r.get("SifraSezSto"): r.get("OpisSezSto", "")
                 for r in _by_oznaka(sifranti, "S1").get("zapisi", [])
                 if r.get("SifraSezSto")}

    # direct codes per dejavnost (K1.1) and allowed seznami per dejavnost (K1.2)
    allowed: dict = {}                # vrsDej -> set(SiSto)
    seznami: dict = {}               # vrsDej -> set(sezCode)
    for r in _current(mapping_rows.get("K1.1", [])):
        vrs, sisto = r.get("SifraVrsZdrDej"), r.get("SiSto")
        if vrs and sisto:
            allowed.setdefault(vrs, set()).add(sisto)
    for r in _current(mapping_rows.get("K1.2", [])):
        vrs, sez = r.get("SifraVrsZdrDej"), r.get("SifraSezSto")
        if not (vrs and sez):
            continue
        seznami.setdefault(vrs, set()).add(sez)
        allowed.setdefault(vrs, set()).update(sez_to_sisto.get(sez, ()))

    # reverse index SiSto -> [vrsDej]
    by_sifra: dict = {}
    for vrs, codes in allowed.items():
        for c in codes:
            by_sifra.setdefault(c, set()).add(vrs)

    return {
        "dejavnosti": dejavnosti,
        "podvrste": podvrste,
        "podvrsta_to_vrsta": {k: sorted(v) for k, v in podvrsta_to_vrsta.items()},
        "allowed_by_dejavnost": {k: sorted(v) for k, v in allowed.items()},
        "seznami_by_dejavnost": {
            k: sorted(([s, sez_naziv.get(s, "")] for s in v), key=lambda x: x[0])
            for k, v in seznami.items()
        },
        "dejavnosti_by_sifra": {k: sorted(v) for k, v in by_sifra.items()},
    }


def _finalize(cur: dict, sifranti: dict) -> None:
    sifra_field, opis_field = _pick_fields(cur["field_order"])
    oznaka = cur["oznaka"] or cur["naziv"]
    key = oznaka
    n = 2
    while key in sifranti:                     # guard against duplicate oznake
        key = f"{oznaka}#{n}"
        n += 1
    entry = {
        "oznaka": cur["oznaka"],
        "naziv": cur["naziv"],
        "naziv_human": cur["naziv_human"],
        "valja_od": cur["valja_od"],
        "polja": cur["field_order"],
        "sifra_field": sifra_field,
        "opis_field": opis_field,
        "count": len(cur["zapisi"]),
        "is_code_list": opis_field is not None,
    }
    # Store rows only for code lists (description-bearing); document the rest by metadata.
    entry["zapisi"] = cur["zapisi"] if opis_field is not None else []
    sifranti[key] = entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", help="path to a specific vsi_sifranti XML (else use latest.json)")
    args = ap.parse_args()

    if args.xml:
        xml_path = Path(args.xml)
        m = re.search(r"vsi_sifranti_(\d+)_(\d+)", xml_path.name)
        leto, zap = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    else:
        meta = json.loads((SIFRANTI_DIR / "latest.json").read_text())
        xml_path = SIFRANTI_DIR / meta["xml_file"]
        leto, zap = meta["leto"], meta["zap_st"]

    print(f"Parsing {xml_path.name} (objava {leto}/{zap})...")
    cat = build(xml_path, leto, zap)

    sif = cat["sifranti"]
    code_lists = {k: v for k, v in sif.items() if v["is_code_list"]}
    total_rows = sum(v["count"] for v in code_lists.values())
    OUT.write_text(json.dumps(cat, ensure_ascii=False), encoding="utf-8")

    print(f"Objava {cat['objava']['leto']}/{cat['objava']['zap_st']} "
          f"({cat['objava']['datum']})")
    print(f"  komentar: {cat['objava']['komentar'][:90]}")
    print(f"  šifrantov skupaj: {len(sif)}  |  code-lists: {len(code_lists)}  "
          f"|  shranjenih zapisov: {total_rows:,}")
    di = cat.get("dejavnost_index", {})
    print(f"  dejavnost index: {len(di.get('dejavnosti', {}))} dejavnosti, "
          f"{len(di.get('allowed_by_dejavnost', {}))} z dovoljenimi storitvami, "
          f"{len(di.get('dejavnosti_by_sifra', {}))} storitev mapiranih")
    print(f"  -> {OUT}  ({OUT.stat().st_size/1_048_576:.1f} MB)")
    print("\n  Top code-lists by size:")
    for k, v in sorted(code_lists.items(), key=lambda kv: -kv[1]["count"])[:12]:
        print(f"    šifrant {k:8s} {v['naziv'][:42]:42s} {v['count']:>6}  "
              f"[{v['sifra_field']} / {v['opis_field']}]")


if __name__ == "__main__":
    main()
