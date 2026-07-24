"""Periodic, incremental, atomic refresh of the ZZZS data layer — runs in-process on the Fly app.

Goals (see docs / plan):
  • Every REFRESH_INTERVAL_DAYS (default 2), pull only what's NEW/CHANGED from ZZZS — never
    re-embed the whole corpus. A persistent ledger (`embedded_ledger.json`) tracks, per document,
    its source URL, version date, embedded chunk ids and embed time.
  • Structured catalogs (šifranti, MTP) re-fetch the latest; the embedded change summary
    (KomVsebSprememb) is the changelog. Old snapshots of versioned docs are pruned (latest-only).
  • Staged + atomic: download → extract → chunk happens off to the side; the live ChromaDB
    collection / catalogs are only mutated in a commit phase, per document, after a successful
    retrieval. A failure of one document is skipped (retried next run), never crashes serving.
  • Emails a summary (Gmail SMTP) on any change, on failure, and on every manual run.

`refresh_all(embed, client, manual=False)` is called by the scheduler in main.py with the app's
shared embedding function + ChromaDB client (one client per path — never make a second).
"""
import datetime as dt
import json
import os
import smtplib
import sys
import time
import traceback
from email.message import EmailMessage
from pathlib import Path

# Reuse the crawl/extract/chunk logic from the offline builder (shipped to /app/scripts).
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import build_egradiva_index as bx  # noqa: E402

COLLECTION_NAME = "zzzs_egradiva"
STATE_DIR = Path(os.environ.get("ASCLEPIUS_DATA_DIR") or (Path(__file__).parent / "data"))
LEDGER_FILE = STATE_DIR / "embedded_ledger.json"
STATE_FILE = STATE_DIR / "refresh_state.json"
MANIFEST_FILE = STATE_DIR / "egradiva_manifest.json"
MAX_DOCS_PER_RUN = int(os.environ.get("REFRESH_MAX_DOCS", "300"))  # safety bound per run

# Downloads go to the volume on Fly (set EGRADIVA_FILES_DIR=/data/egradiva_files).
bx.FILES_DIR = Path(os.environ.get("EGRADIVA_FILES_DIR", str(bx.FILES_DIR)))


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _load_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(p: Path, data) -> None:
    """Atomic write (temp + rename) so a crash never leaves a half-written file."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


# ── Ledger ───────────────────────────────────────────────────────────────────

def _bootstrap_ledger(collection) -> dict:
    """Build the ledger from the already-embedded collection (first run / missing ledger),
    so we know what's indexed without re-embedding anything."""
    print("refresh: bootstrapping ledger from existing collection...")
    ledger, n, offset, batch = {}, 0, 0, 5000
    while True:                                    # paginate — get() can't return 144k rows at once
        got = collection.get(include=["metadatas"], limit=batch, offset=offset)
        ids = got["ids"]
        if not ids:
            break
        for cid, meta in zip(ids, got["metadatas"]):
            unid = (meta or {}).get("unid") or (meta or {}).get("doc_id")
            if not unid:
                continue
            e = ledger.setdefault(unid, {"unid": unid, "title": (meta or {}).get("doc_title", ""),
                                         "source_url": (meta or {}).get("file_url", ""),
                                         "doc_date": (meta or {}).get("doc_date", ""),
                                         "category": (meta or {}).get("category", ""),
                                         "chunk_ids": [], "embedded_at": (meta or {}).get("embedded_at", "")})
            e["chunk_ids"].append(cid)
        n += len(ids)
        offset += len(ids)
        if len(ids) < batch:
            break
    print(f"refresh: ledger bootstrapped — {len(ledger)} docs, {n} chunks")
    return ledger


def _get_ledger(collection) -> dict:
    led = _load_json(LEDGER_FILE, None)
    if led is None:
        led = _bootstrap_ledger(collection)
        _save_json(LEDGER_FILE, led)
    return led


# ── Crawl + diff ─────────────────────────────────────────────────────────────

def _crawl_light() -> dict:
    """Current ZZZS docs (unid -> {doc_date, title, category}) — categories only, no per-doc fetch."""
    current = {}
    for cat in bx.crawl_categories():
        for doc in bx.crawl_documents_for_category(cat):
            unid = doc.get("@unid") or doc.get("unid") or ""
            if not unid:
                continue
            current[unid] = {
                "doc_date": bx._norm_date(doc.get("DATUM") or doc.get("datum")),
                "title": (doc.get("NASLOV") or "").strip(),
                "category_name": doc.get("category_name", ""),
                "category_code": doc.get("category_code", ""),
                "_doc": doc,
            }
    return current


def _diff(ledger: dict, current: dict):
    new = [u for u in current if u not in ledger]
    # "changed" only when BOTH sides have a date and they differ — so a missing ledger date
    # (e.g. pre-provenance chunks) never triggers a needless full re-embed.
    changed = [u for u in current if u in ledger
               and current[u]["doc_date"] and ledger[u].get("doc_date")
               and current[u]["doc_date"] != ledger[u]["doc_date"]]
    removed = [u for u in ledger if u not in current]
    return new, changed, removed


# ── Stage one document (download → extract → chunk), no collection writes ─────

_TEXT_EXTS = {"pdf", "docx", "doc", "tif", "tiff"}


def _stage_doc(unid: str, info: dict):
    """Return (chunks, entry). chunks is [] for docs with no extractable text (those are ledgered
    as no_content so they aren't re-fetched every run). Only text docs are downloaded + chunked."""
    base = {"unid": unid, "title": info["title"], "doc_date": info["doc_date"],
            "category_name": info["category_name"], "file_url": "", "file_type": "", "local_path": None}
    files = bx.crawl_document_files(info["_doc"])
    if not files:
        return [], base
    f = files[0]
    base["file_url"], base["file_type"] = f["file_url"], f["file_type"]
    ext = bx._real_ext({"file_url": f["file_url"], "file_type": f["file_type"]})
    if ext not in _TEXT_EXTS:                      # xls/xlsx/zip/… → no text content; don't download
        return [], base
    import requests
    bx.FILES_DIR.mkdir(parents=True, exist_ok=True)
    local = bx.FILES_DIR / f"{unid}.{f.get('file_type') or 'bin'}"
    if not (local.exists() and local.stat().st_size > 0):
        r = requests.get(f["file_url"], timeout=180,
                         headers={"User-Agent": "Mozilla/5.0 (Asclepius refresh)"})
        if r.status_code != 200 or not r.content:
            raise RuntimeError(f"download {r.status_code}")
        local.write_bytes(r.content)
    base["local_path"] = str(local)
    return bx.chunk_document(base), base           # OCR/.doc-aware extractors


# ── e-gradiva incremental (staged, then atomic per-doc commit) ────────────────

def _refresh_egradiva(collection, report: dict) -> None:
    ledger = _get_ledger(collection)
    current = _crawl_light()
    # Cheap ledger-only date backfill: give pre-provenance docs their current date (no re-embed) so
    # a *future* date change is detectable. Doesn't touch the collection.
    backfilled = 0
    for u in current:
        if u in ledger and not ledger[u].get("doc_date") and current[u]["doc_date"]:
            ledger[u]["doc_date"] = current[u]["doc_date"]
            backfilled += 1
    new, changed, removed = _diff(ledger, current)
    report["egradiva"] = {"new": len(new), "changed": len(changed), "removed": len(removed),
                          "date_backfilled": backfilled, "added_docs": [], "pruned_docs": [], "errors": 0}

    # Stage new + changed (bounded). Per-doc resilient: a failure is skipped, not fatal.
    todo = (new + changed)[:MAX_DOCS_PER_RUN]
    staged = []   # (unid, entry, chunks, old_chunk_ids)
    for unid in todo:
        try:
            chunks, entry = _stage_doc(unid, current[unid])
            if chunks:
                old_ids = ledger.get(unid, {}).get("chunk_ids", [])  # for changed → prune old version
                staged.append((unid, entry, chunks, old_ids))
            else:
                # no extractable text → ledger it so it isn't re-fetched every run
                ledger[unid] = {"unid": unid, "title": current[unid]["title"],
                                "source_url": entry.get("file_url", ""), "doc_date": current[unid]["doc_date"],
                                "category": current[unid]["category_name"], "chunk_ids": [],
                                "embedded_at": _now(), "status": "no_content"}
        except Exception as e:
            report["egradiva"]["errors"] += 1
            print(f"refresh: stage failed for {unid}: {e}")

    # Commit phase: only now do we mutate the live collection (and the ledger), per document.
    for unid, entry, chunks, old_ids in staged:
        try:
            ids, docs, metas = [], [], []
            for i, ch in enumerate(chunks):
                ids.append(f"{unid}_chunk_{i}")
                docs.append(ch["text"])
                metas.append({k: ("" if v is None else v) for k, v in ch["metadata"].items()})
            if old_ids:
                collection.delete(ids=old_ids)         # prune the prior version (latest-only)
            for j in range(0, len(ids), 100):
                collection.upsert(ids=ids[j:j+100], documents=docs[j:j+100], metadatas=metas[j:j+100])
            ledger[unid] = {"unid": unid, "title": entry["title"], "source_url": entry["file_url"],
                            "doc_date": entry["doc_date"], "category": entry["category_name"],
                            "chunk_ids": ids, "embedded_at": _now()}
            report["egradiva"]["added_docs"].append({"unid": unid, "title": entry["title"],
                                                     "doc_date": entry["doc_date"]})
        except Exception as e:
            report["egradiva"]["errors"] += 1
            print(f"refresh: commit failed for {unid}: {e}")

    # Prune docs ZZZS removed from the listing (latest-only).
    for unid in removed:
        try:
            old_ids = ledger.get(unid, {}).get("chunk_ids", [])
            if old_ids:
                collection.delete(ids=old_ids)
            report["egradiva"]["pruned_docs"].append({"unid": unid,
                                                      "title": ledger.get(unid, {}).get("title", "")})
            ledger.pop(unid, None)
        except Exception as e:
            print(f"refresh: prune failed for {unid}: {e}")

    _save_json(LEDGER_FILE, ledger)
    # Persist a manifest snapshot for reference.
    _save_json(MANIFEST_FILE, [{"unid": u, **{k: v for k, v in info.items() if k != "_doc"}}
                               for u, info in current.items()])
    report["egradiva"]["total_indexed_docs"] = len(ledger)
    report["egradiva"]["chunk_count"] = collection.count()


# ── Structured catalogs (Tier A) ─────────────────────────────────────────────

def _refresh_sifranti(report: dict) -> None:
    import fetch_sifranti as fs
    import build_sifrant_catalog as bsc
    import sifranti
    cur = sifranti.OBJAVA or {}
    leto, zap, content = fs.find_latest()
    if not content:
        return
    newer = (leto, zap) > (cur.get("leto", 0), cur.get("zap_st", 0))
    # Also rebuild for the SAME objava when the loaded catalog is missing derived data — e.g. after
    # a code upgrade that adds the dejavnost index. paths.data_path prefers the writable volume copy
    # over the freshly-baked image copy, so without this a new derivation would never reach the
    # volume until ZZZS happens to publish a new objava. This makes the volume self-heal.
    # Also self-heal when the loaded catalog predates the MP enrichment (its link tables, e.g. K42,
    # were not retained by an older build_sifrant_catalog.py) — otherwise the MTP prescribing join
    # would stay empty on the volume until ZZZS happens to publish a new objava.
    mp_link_missing = not (sifranti.SIFRANTI.get("K42") or {}).get("zapisi")
    needs_rebuild = not getattr(sifranti, "DEJAVNOSTI", {}) or mp_link_missing
    if not newer and not needs_rebuild:
        report["sifranti"] = {"changed": False, "objava": f"{cur.get('leto')}/{cur.get('zap_st')}"}
        return
    fs.extract_and_save(leto, zap, content)        # writes data/sifranti/*.xml + latest.json
    cat = bsc.build(fs.DATA_DIR / f"vsi_sifranti_{leto}_{zap}.xml", leto, zap)
    out = STATE_DIR / "sifrant_catalog.json"
    _save_json(out, cat)
    sifranti.initialize_sifranti()                 # hot-reload from the volume copy
    di = cat.get("dejavnost_index", {})
    report["sifranti"] = {"changed": bool(newer),
                          "rebuilt_for_derivation": bool(needs_rebuild and not newer),
                          "from": f"{cur.get('leto')}/{cur.get('zap_st')}",
                          "to": f"{leto}/{zap}", "datum": cat["objava"].get("datum"),
                          "komentar": cat["objava"].get("komentar", ""),
                          "dejavnosti": len(di.get("dejavnosti", {}))}


def _refresh_mtp(report: dict) -> None:
    import importlib
    import mtp
    cur_datum = mtp.DATUM
    try:
        import subprocess
        # fetch newest + rebuild via the existing scripts (writes data/mtp_catalog.json)
        subprocess.run([sys.executable, str(Path(__file__).parent / "scripts" / "fetch_mtp.py")],
                       check=True, timeout=300)
        subprocess.run([sys.executable, str(Path(__file__).parent / "scripts" / "build_mtp_catalog.py")],
                       check=True, timeout=300)
    except Exception as e:
        report["mtp"] = {"changed": False, "error": str(e)}
        return
    built = _load_json(Path(__file__).parent / "data" / "mtp_catalog.json", {})
    new_datum = built.get("datum", "")
    datum_changed = bool(new_datum and new_datum != cur_datum)
    # Self-heal: also commit + reload when the freshly-built catalog now carries prescribing
    # enrichment but the loaded (volume) copy does not — otherwise a stale non-enriched volume
    # catalog would keep being served after a deploy until the MTP datum happens to change.
    enrich_added = bool(built.get("enriched")) and not getattr(mtp, "ENRICHED", 0)
    if datum_changed or enrich_added:
        _save_json(STATE_DIR / "mtp_catalog.json", built)
        mtp.initialize_mtp()
        report["mtp"] = {"changed": datum_changed, "enrichment_added": enrich_added,
                         "from": cur_datum, "to": new_datum or cur_datum}
    else:
        report["mtp"] = {"changed": False, "datum": cur_datum}


# ── Email ────────────────────────────────────────────────────────────────────

def _has_changes(report: dict) -> bool:
    return bool(report.get("sifranti", {}).get("changed")
                or report.get("mtp", {}).get("changed")
                or report.get("egradiva", {}).get("added_docs")
                or report.get("egradiva", {}).get("pruned_docs"))


def _num(n) -> str:
    """Thousands-separated (Slovenian style: 145.890)."""
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)


_MAX_LIST = 60   # cap the per-doc list so the email can't balloon


def _email_bodies(report: dict, manual: bool):
    """Build (subject, plain_text, html) for the refresh summary. Structured, no doc summaries."""
    import html as _html
    status = (report.get("status") or "ok").upper()
    nacin = "ročno" if manual else "samodejno"
    changed = _has_changes(report)
    subject = (f"[Asclepius MCP] ZZZS osvežitev: {status} — "
               f"{'spremembe' if changed else 'brez sprememb'}")

    sif = report.get("sifranti", {})
    mtp_r = report.get("mtp", {})
    eg = report.get("egradiva", {})
    added = eg.get("added_docs", []) or []
    pruned = eg.get("pruned_docs", []) or []
    # newest first for the listing
    added_sorted = sorted(added, key=lambda d: d.get("doc_date", ""), reverse=True)

    # ── plain text ──
    P = ["ASCLEPIUS MCP — OSVEŽITEV PODATKOV ZZZS",
         "=" * 40, "",
         f"Status: {status}   |   Trajanje: {report.get('duration_s','?')} s   |   Način: {nacin}",
         f"Čas: {report.get('finished_at','')}", ""]
    P += ["ŠIFRANTI (obračun)", "-" * 18]
    if sif.get("changed"):
        P += [f"Nova objava: {sif.get('from')} -> {sif.get('to')}   (datum {sif.get('datum')})",
              f"Vsebina sprememb: {sif.get('komentar') or '(ni navedena)'}"]
    else:
        P += [f"Brez sprememb (objava {sif.get('objava','?')})"]
    P += ["", "MEDICINSKI PRIPOMOČKI (MTP)", "-" * 27]
    P += [f"Sprememba: {mtp_r.get('from')} -> {mtp_r.get('to')}" if mtp_r.get("changed")
          else f"Brez sprememb (stanje {mtp_r.get('datum','?')})"]
    P += ["", "E-GRADIVA", "-" * 9,
          f"Novih: {len(added)}   |   Odstranjenih: {len(pruned)}   |   "
          f"Napak: {eg.get('errors', 0)}",
          f"Skupaj chunkov: {_num(eg.get('chunk_count','?'))}"]
    if added_sorted:
        P += ["", "Novi dokumenti:"]
        for d in added_sorted[:_MAX_LIST]:
            P.append(f"  {d.get('doc_date','—') or '—'}   {d.get('title','')}")
        if len(added_sorted) > _MAX_LIST:
            P.append(f"  … in še {len(added_sorted) - _MAX_LIST} drugih")
    if pruned:
        P += ["", "Odstranjeni dokumenti:"]
        for d in pruned[:_MAX_LIST]:
            P.append(f"  {d.get('title','')}")
    plain = "\n".join(P)

    # ── html ──
    def esc(s):
        return _html.escape(str(s or ""))
    h3 = "margin:18px 0 6px;border-bottom:1px solid #ccc;padding-bottom:3px;font-size:15px"
    H = [f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#222;line-height:1.5">',
         f'<h2 style="margin:0 0 4px">Asclepius MCP — osvežitev podatkov ZZZS</h2>',
         f'<p style="margin:0 0 12px;color:#555"><b>Status:</b> {esc(status)} &nbsp;|&nbsp; '
         f'<b>Trajanje:</b> {esc(report.get("duration_s","?"))} s &nbsp;|&nbsp; '
         f'<b>Način:</b> {esc(nacin)}<br><b>Čas:</b> {esc(report.get("finished_at",""))}</p>']
    H.append(f'<h3 style="{h3}"><u>Šifranti (obračun)</u></h3>')
    if sif.get("changed"):
        H.append(f'<p style="margin:0"><b>Nova objava:</b> {esc(sif.get("from"))} → '
                 f'<b>{esc(sif.get("to"))}</b> &nbsp;(datum {esc(sif.get("datum"))})<br>'
                 f'<b>Vsebina sprememb:</b> {esc(sif.get("komentar") or "(ni navedena)")}</p>')
    else:
        H.append(f'<p style="margin:0;color:#555">Brez sprememb (objava {esc(sif.get("objava","?"))})</p>')
    H.append(f'<h3 style="{h3}"><u>Medicinski pripomočki (MTP)</u></h3>')
    H.append(f'<p style="margin:0">{("<b>Sprememba:</b> "+esc(mtp_r.get("from"))+" → <b>"+esc(mtp_r.get("to"))+"</b>") if mtp_r.get("changed") else ("<span style=\"color:#555\">Brez sprememb (stanje "+esc(mtp_r.get("datum","?"))+")</span>")}</p>')
    H.append(f'<h3 style="{h3}"><u>E-gradiva</u></h3>')
    H.append(f'<p style="margin:0 0 6px"><b>Novih:</b> {len(added)} &nbsp;|&nbsp; '
             f'<b>Odstranjenih:</b> {len(pruned)} &nbsp;|&nbsp; <b>Napak:</b> {eg.get("errors",0)} '
             f'&nbsp;|&nbsp; <b>Skupaj chunkov:</b> {_num(eg.get("chunk_count","?"))}</p>')
    if added_sorted:
        H.append('<p style="margin:8px 0 2px"><u>Novi dokumenti</u></p><ul style="margin:0;padding-left:20px">')
        for d in added_sorted[:_MAX_LIST]:
            H.append(f'<li><b>{esc(d.get("doc_date","—") or "—")}</b> — {esc(d.get("title",""))}</li>')
        H.append("</ul>")
        if len(added_sorted) > _MAX_LIST:
            H.append(f'<p style="margin:2px 0;color:#555">… in še {len(added_sorted)-_MAX_LIST} drugih</p>')
    if pruned:
        H.append('<p style="margin:8px 0 2px"><u>Odstranjeni dokumenti</u></p><ul style="margin:0;padding-left:20px">')
        for d in pruned[:_MAX_LIST]:
            H.append(f'<li>{esc(d.get("title",""))}</li>')
        H.append("</ul>")
    H.append("</div>")
    return subject, plain, "\n".join(H)


def _email(report: dict, manual: bool) -> None:
    user, pw = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    to = os.environ.get("NOTIFY_EMAIL")
    if not (user and pw and to):
        print("refresh: SMTP not configured — skipping email")
        return
    if not (manual or report.get("status") == "error" or _has_changes(report)):
        return  # scheduled run with nothing to report
    subject, plain, html = _email_bodies(report, manual)
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(plain)                 # plain-text part
    msg.add_alternative(html, subtype="html")  # rendered (bold/underline/headers) in Gmail
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls(); s.login(user, pw); s.send_message(msg)
        print("refresh: notification email sent")
    except Exception as e:
        print(f"refresh: email failed: {e}")


# ── Orchestrator ─────────────────────────────────────────────────────────────

def refresh_all(embed=None, client=None, *, manual: bool = False) -> dict:
    """Run a full refresh. Never raises — returns a report dict and writes refresh_state.json."""
    t0 = time.time()
    report = {"started_at": _now(), "manual": manual, "status": "ok"}
    try:
        if client is None:
            import chromadb
            client = chromadb.PersistentClient(path=os.environ.get("CHROMADB_PATH",
                                               str(Path(__file__).parent / "data" / "chromadb")))
        collection = client.get_collection(COLLECTION_NAME, embedding_function=embed)
        for tier in (lambda: _refresh_sifranti(report),
                     lambda: _refresh_mtp(report),
                     lambda: _refresh_egradiva(collection, report)):
            try:
                tier()
            except Exception as e:
                report["status"] = "partial"
                report.setdefault("errors", []).append(str(e))
                print(f"refresh: tier failed: {e}\n{traceback.format_exc()}")
    except Exception as e:
        report["status"] = "error"
        report["fatal"] = str(e)
        print(f"refresh: fatal: {e}\n{traceback.format_exc()}")
    report["finished_at"] = _now()
    report["duration_s"] = round(time.time() - t0, 1)
    # Persist state (keep a short history) + notify.
    state = _load_json(STATE_FILE, {"history": []})
    state["last_refresh"] = report
    state["history"] = ([report] + state.get("history", []))[:10]
    _save_json(STATE_FILE, state)
    _email(report, manual)
    return report


def get_status() -> dict:
    return _load_json(STATE_FILE, {"last_refresh": None, "history": []})


if __name__ == "__main__":
    # Manual/local run (uses a fresh client + the bundled model).
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    e = SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    print(json.dumps(refresh_all(embed=e, manual=True), ensure_ascii=False, indent=2)[:2000])
