"""Cross-source change digest — what changed across ZZZS sources in the last N days.

One actionable, summarized view over every data source the server tracks (obračunski
šifranti, medicinski pripomočki, navodila za obračun, e-gradiva dokumenti). The window is
defined by each item's own publication/document date (`doc_date`, objava `datum`, billing
`date`, MTP `datum`); it falls back to the ingestion date (`embedded_at`) only when a
document carries no publication date.

The tool returns concise metadata + source links only — never full document bodies. Long
sections are capped and point back to the source / search tools, so the connected model can
turn the digest into an actionable clinical summary (per the server instructions).

Reads only already-loaded module state (sifranti.OBJAVA, mtp.DATUM, obracun._METAS) and two
durable JSON files written by refresh.py (embedded_ledger.json, refresh_state.json). No
ChromaDB query, no embeddings, no init step of its own.
"""
import datetime as dt
import json
from typing import Any, Dict, List, Optional

from fastmcp.tools.tool import ToolResult

from paths import data_path

# Already-loaded state we read. None of these import `changes`, so there is no cycle; and
# importing them only runs their module top-level (globals), not their initialize_* loaders.
import sifranti
import mtp
import obracun

# Caps so a long window can't balloon the output. When a list is truncated we append a
# "… in še N drugih" pointer (mirrors refresh._MAX_LIST) so nothing is silently dropped.
_CAP_OBRACUN = 25
_CAP_EGRADIVA = 50
_CAP_PER_CATEGORY = 15

# Accept English + Slovenian source names for the optional `source` filter.
_SOURCE_ALIASES = {
    "sifranti": "sifranti", "šifranti": "sifranti", "sifrant": "sifranti", "šifrant": "sifranti",
    "mtp": "mtp", "pripomocki": "mtp", "pripomočki": "mtp", "pripomocek": "mtp", "pripomoček": "mtp",
    "obracun": "obracun", "obračun": "obracun", "dogovor": "obracun", "billing": "obracun",
    "egradiva": "egradiva", "e-gradiva": "egradiva", "dokumenti": "egradiva", "gradiva": "egradiva",
}


def _load_json(name: str, default):
    try:
        return json.loads(data_path(name).read_text(encoding="utf-8"))
    except Exception:
        return default


def _parse_date(s: Any) -> Optional[dt.date]:
    """Parse ISO 'YYYY-MM-DD' (or a full ISO timestamp) and Slovenian 'DD.MM.YYYY'.

    ZZZS values are calendar publication dates, so we compare them as dates (there is no
    Ljubljana timezone anywhere in the repo — 'today' is taken as the UTC calendar date)."""
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    try:
        return dt.date.fromisoformat(s[:10])          # 2026-06-23 / 2026-06-23T12:00:00
    except Exception:
        pass
    try:
        return dt.datetime.strptime(s[:10], "%d.%m.%Y").date()   # 20.04.2026
    except Exception:
        return None


def _summarize_changes(days: int = 45, source: Optional[str] = None) -> ToolResult:
    """Build the last-N-days cross-source change digest. Never raises — returns a ToolResult."""
    try:
        days = int(days)
    except Exception:
        days = 45
    if days <= 0:
        days = 45

    # Obtain the current date at call time, then derive the look-back window.
    today = dt.datetime.now(dt.timezone.utc).date()
    cutoff = today - dt.timedelta(days=days)

    want = _SOURCE_ALIASES.get((source or "").strip().lower()) if source else None

    def _in(d: Optional[dt.date]) -> bool:
        return d is not None and d >= cutoff

    sections: List = []                # (heading, [markdown lines])
    counts: Dict[str, int] = {}
    entries: Dict[str, Any] = {}

    # ── 1) Obračunski šifranti — snapshot changelog (latest publication only) ──────────
    if want in (None, "sifranti"):
        o = getattr(sifranti, "OBJAVA", None) or {}
        if o and _in(_parse_date(o.get("datum"))):
            try:
                cit = sifranti._citation()
            except Exception:
                cit = getattr(sifranti, "SOURCE_LABEL", "ZZZS Vsi šifranti (obračun)")
            lines = [
                f"**{o.get('datum')}** — nova objava šifrantov {o.get('leto')}/{o.get('zap_st')}",
                "",
                "> " + (o.get("komentar") or "(brez navedene vsebine sprememb)"),
                "",
                f"**Vir:** [{cit}]({o.get('source_url', '')})",
            ]
            sections.append(("Obračunski šifranti (ZZZS)", lines))
            counts["sifranti"] = 1
            entries["sifranti"] = [{"datum": o.get("datum"),
                                    "objava": f"{o.get('leto')}/{o.get('zap_st')}",
                                    "komentar": o.get("komentar"),
                                    "source_url": o.get("source_url", "")}]

    # ── 2) Medicinski pripomočki (MTP) — snapshot (latest catalog date only) ───────────
    if want in (None, "mtp"):
        mdatum = getattr(mtp, "DATUM", "") or ""
        if _in(_parse_date(mdatum)):
            label = getattr(mtp, "SOURCE_LABEL", "ZZZS Seznam medicinskih pripomočkov")
            url = getattr(mtp, "SOURCE_URL", "") or ""
            lines = [
                f"**{mdatum}** — posodobljen seznam medicinskih pripomočkov (MTP)",
                "",
                f"**Vir:** [{label}]({url})",
            ]
            sections.append(("Medicinski pripomočki (MTP)", lines))
            counts["mtp"] = 1
            entries["mtp"] = [{"datum": mdatum, "source_url": url}]

    # ── 3) Navodila za obračun / Splošni dogovor — per-document `date` ─────────────────
    if want in (None, "obracun"):
        docs: Dict[Any, Dict[str, Any]] = {}   # dedupe per-chunk metas down to one row per doc
        for m in (getattr(obracun, "_METAS", None) or []):
            if not isinstance(m, dict):
                continue
            key = (m.get("doc_title", ""), m.get("source_url", ""))
            if key in docs:
                continue
            d = _parse_date(m.get("date"))
            if not _in(d):
                continue
            docs[key] = {"title": m.get("doc_title") or "(brez naslova)",
                         "date": m.get("date", ""), "d": d,
                         "role": m.get("role", ""), "domain": m.get("domain", ""),
                         "source_url": m.get("source_url", "")}
        ordered = sorted(docs.values(), key=lambda x: x["d"], reverse=True)
        if ordered:
            lines = []
            for x in ordered[:_CAP_OBRACUN]:
                ctx = " · ".join(t for t in (x["role"], x["domain"]) if t)
                ctx = f" · _{ctx}_" if ctx else ""
                vir = f" — [vir]({x['source_url']})" if x["source_url"] else ""
                lines.append(f"- **{x['date']}** — {x['title']}{ctx}{vir}")
            if len(ordered) > _CAP_OBRACUN:
                lines.append(f"_… in še {len(ordered) - _CAP_OBRACUN} drugih — "
                             f"uporabite `get_billing_rules`._")
            sections.append(("Navodila za obračun / Splošni dogovor", lines))
            counts["obracun"] = len(ordered)
            entries["obracun"] = [{"date": x["date"], "title": x["title"], "role": x["role"],
                                   "source_url": x["source_url"]} for x in ordered]

    # ── 4) ZZZS e-gradiva — durable per-document ledger (the only 45-day history) ──────
    if want in (None, "egradiva"):
        kept: List[Dict[str, Any]] = []
        for _unid, e in (_load_json("embedded_ledger.json", {}) or {}).items():
            if not isinstance(e, dict):
                continue
            d = _parse_date(e.get("doc_date")) or _parse_date(e.get("embedded_at"))
            if not _in(d):
                continue
            kept.append({"title": e.get("title") or "(brez naslova)",
                         "date": e.get("doc_date") or (e.get("embedded_at", "") or "")[:10],
                         "d": d, "category": e.get("category") or "Ostalo",
                         "source_url": e.get("source_url", "")})

        # Removed docs survive only in the refresh history (ledger drops them on prune).
        removed: List[str] = []
        for run in (_load_json("refresh_state.json", {}) or {}).get("history") or []:
            if not _in(_parse_date((run.get("finished_at") or "")[:10])):
                continue
            for p in ((run.get("egradiva") or {}).get("pruned_docs") or []):
                removed.append(p.get("title") or p.get("unid") or "(neznan dokument)")

        if kept or removed:
            lines = []
            by_cat: Dict[str, List[Dict[str, Any]]] = {}
            for x in kept:
                by_cat.setdefault(x["category"], []).append(x)
            # Categories ordered by their most-recent document; docs newest-first within.
            cat_order = sorted(by_cat.items(),
                               key=lambda kv: max(x["d"] for x in kv[1]), reverse=True)
            shown = 0
            for cat, items in cat_order:
                if shown >= _CAP_EGRADIVA:
                    break
                items.sort(key=lambda x: x["d"], reverse=True)
                lines.append(f"**{cat}**")
                for x in items[:_CAP_PER_CATEGORY]:
                    if shown >= _CAP_EGRADIVA:
                        break
                    vir = f" — [vir]({x['source_url']})" if x["source_url"] else ""
                    lines.append(f"- **{x['date']}** — {x['title']}{vir}")
                    shown += 1
                if len(items) > _CAP_PER_CATEGORY:
                    lines.append(f"_… in še {len(items) - _CAP_PER_CATEGORY} v tej kategoriji._")
                lines.append("")
            if len(kept) > shown:
                lines.append(f"_Prikazanih {shown} od {len(kept)} dokumentov — zožite po kategoriji "
                             f"ali uporabite `search_zzzs_documents`._")
            if removed:
                lines.append("")
                lines.append("**Odstranjeni dokumenti:**")
                for t in removed[:_CAP_PER_CATEGORY]:
                    lines.append(f"- {t}")
                if len(removed) > _CAP_PER_CATEGORY:
                    lines.append(f"_… in še {len(removed) - _CAP_PER_CATEGORY} odstranjenih._")
            sections.append(("ZZZS e-gradiva (dokumenti)", lines))
            counts["egradiva"] = len(kept)
            counts["egradiva_removed"] = len(removed)
            entries["egradiva"] = [{"date": x["date"], "title": x["title"],
                                    "category": x["category"], "source_url": x["source_url"]}
                                   for x in kept]
            entries["egradiva_removed"] = removed

    # ── Assemble ──────────────────────────────────────────────────────────────────────
    total = sum(v for k, v in counts.items() if k != "egradiva_removed")
    out = [f"**Spremembe v zadnjih {days} dneh** — "
           f"{cutoff.strftime('%d.%m.%Y')} do {today.strftime('%d.%m.%Y')}", ""]
    if not sections:
        out.append("Brez zabeleženih sprememb v tem oknu.")
        if want:
            out.append(f"_(Filter vira: {want}.)_")
    else:
        for heading, lines in sections:
            out.append(f"### {heading}")
            out.extend(lines)
            out.append("")
    out.append(f"_Datum poizvedbe: {today.strftime('%d.%m.%Y')} (UTC) · okno {days} dni · "
               f"merilo: datum objave dokumenta._")

    data = {"days": days, "cutoff": cutoff.isoformat(), "today": today.isoformat(),
            "total_changes": total, "counts_per_source": counts,
            "source_filter": want, "entries": entries}
    return ToolResult(content="\n".join(out).rstrip(), structured_content=data)


if __name__ == "__main__":
    # Populate the snapshot sources (local JSON only) so the demo shows real data.
    # obracun/egradiva need ChromaDB/a runtime ledger, so they stay empty here — expected.
    for name, fn in (("sifranti", getattr(sifranti, "initialize_sifranti", None)),
                     ("mtp", getattr(mtp, "initialize_mtp", None))):
        try:
            fn and fn()
        except Exception as e:
            print(f"[demo] {name} init skipped: {e}")
    for d in (45, 1, 400):
        print("\n" + "=" * 72)
        r = _summarize_changes(days=d)
        print(r.content)
        print("--- counts:", r.structured_content.get("counts_per_source"))
