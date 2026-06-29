"""Outpatient billing-rules module (obračun storitev) — v1.

Authoritative retrieval over the current-year outpatient billing sources (Vsebinsko
navodilo čistopis, Uredba o programih storitev OCR, Standardi kodiranja, billing Q&A),
isolated in the `obracun_rules` ChromaDB collection — separate from the broad e-gradiva
index. Hybrid retrieval: dense embeddings ⊕ TF-IDF over the indexed chunks, RRF-fused.

The collection is prebuilt offline (scripts/build_obracun_index.py) because ingest needs
OCR + docx parsing; this module only loads it (no runtime self-build).
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastmcp.tools.tool import ToolResult

CHROMA_DIR = Path(os.environ.get("CHROMADB_PATH", Path(__file__).parent / "data" / "chromadb"))
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "obracun_rules"
CANDIDATES = 20
RRF_K = 60

COLLECTION = None
EMBED_FN = None
_IDS: List[str] = []
_DOCS: List[str] = []
_METAS: List[dict] = []
_TFIDF: Optional[TfidfVectorizer] = None
_MATRIX = None


def initialize_obracun(embed_fn=None, client=None) -> None:
    """Load the prebuilt obracun_rules collection and build an in-memory TF-IDF arm."""
    global COLLECTION, EMBED_FN, _IDS, _DOCS, _METAS, _TFIDF, _MATRIX
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        EMBED_FN = embed_fn or SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = client or chromadb.PersistentClient(path=str(CHROMA_DIR))
        COLLECTION = client.get_collection(name=COLLECTION_NAME, embedding_function=EMBED_FN)
        got = COLLECTION.get(include=["documents", "metadatas"])
        _IDS, _DOCS, _METAS = got["ids"], got["documents"], got["metadatas"]
        if _DOCS:
            _TFIDF = TfidfVectorizer()
            _MATRIX = _TFIDF.fit_transform(_DOCS)
        print(f"Loaded obracun_rules: {COLLECTION.count()} chunks")
    except Exception as e:
        print(f"WARNING: obracun_rules collection unavailable ({e}); "
              f"run scripts/build_obracun_index.py")
        COLLECTION = None


def _dense_ranked(query: str) -> List[str]:
    if COLLECTION is None:
        return []
    try:
        r = COLLECTION.query(query_texts=[query], n_results=CANDIDATES)
        return r["ids"][0] if r.get("ids") else []
    except Exception:
        return []


def _lexical_ranked(query: str) -> List[str]:
    if _TFIDF is None or _MATRIX is None:
        return []
    sims = cosine_similarity(_TFIDF.transform([query]), _MATRIX).flatten()
    order = sims.argsort()[::-1][:CANDIDATES]
    return [_IDS[i] for i in order if sims[i] > 0]


def _search_obracun(query: str, n_results: int = 8) -> ToolResult:
    if COLLECTION is None or not _IDS:
        err = {"found": False, "error": "Billing-rules index not built"}
        return ToolResult(content=_format_md(err), structured_content=err)
    query = (query or "").strip()
    if not query:
        err = {"found": False, "error": "Query must not be empty"}
        return ToolResult(content=_format_md(err), structured_content=err)

    # RRF fuse dense ⊕ lexical
    scores: Dict[str, float] = {}
    for ranked in (_dense_ranked(query), _lexical_ranked(query)):
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    ordered = sorted(scores, key=lambda c: -scores[c])[:n_results]

    idx = {cid: i for i, cid in enumerate(_IDS)}
    results = []
    for cid in ordered:
        i = idx.get(cid)
        if i is None:
            continue
        m = _METAS[i]
        title, page, src = m.get("doc_title", ""), m.get("page", 0), m.get("source_url", "")
        cite = title + (f", str. {page}" if page else "")
        results.append({
            "text": _strip_header(_DOCS[i]),
            "doc_title": title,
            "role": m.get("role", ""),
            "date": m.get("date", ""),
            "page": page,
            "source_url": src,
            "citation": f"{cite} — {src}" if src else cite,
            "relevance": round(scores[cid], 4),
        })

    if not results:
        err = {"found": False, "error": "No matching billing rules found", "query": query}
        return ToolResult(content=_format_md(err), structured_content=err)
    data = {"found": True, "count": len(results), "query": query, "results": results}
    return ToolResult(content=_format_md(data), structured_content=data)


_HEADER_RE = __import__("re").compile(r"^\[[^\]\n]*\|[^\]\n]*\]\s*\n")


def _strip_header(text: str) -> str:
    """Remove the leading '[doc | role | str. N]' context header from a chunk for display."""
    return _HEADER_RE.sub("", text, count=1).strip()


def _blockquote(text: str) -> str:
    return "\n".join(f"> {l}" for l in text.strip().splitlines()) if text else ""


def _format_md(data: dict) -> str:
    if not data.get("found"):
        return (f"Za \"{data.get('query','')}\" ni zadetkov v pravilih obračuna. "
                f"{data.get('error','')}").strip()
    lines = [f"**Pravila obračuna storitev** | {data['count']} zadetkov za \"{data['query']}\"", ""]
    for r in data["results"]:
        lines += ["---", ""]
        hdr = r.get("doc_title", "Vir")
        if r.get("date"):
            hdr += f" ({r['date']})"
        lines.append(f"### {hdr}")
        if r.get("page"):
            lines.append(f"*str. {r['page']}*")
        lines.append("")
        lines.append(_blockquote(r["text"]))
        lines.append("")
        if r.get("source_url"):
            label = r.get("doc_title") or "Vir"
            if r.get("page"):
                label += f", str. {r['page']}"
            lines.append(f"**Vir:** [{label}]({r['source_url']})")
            lines.append("")
    lines += ["---", "",
              "_Vir so aktualni dokumenti ZZZS za obračun ambulantnih storitev. "
              "Obračunska pravila se letno spreminjajo — preveri pogodbeno leto._"]
    return "\n".join(lines)


if __name__ == "__main__":
    initialize_obracun()
    for q in ["kako se obračuna prvi pregled pri specialistu",
              "ali se lahko obračunata dve storitvi na isti dan",
              "standardi kodiranja glavna diagnoza"]:
        print("\n" + "=" * 60, "\nQUERY:", q)
        sc = _search_obracun(q, 3).structured_content
        for r in sc.get("results", []):
            print(f"  [{r['role']}] str.{r['page']} rel={r['relevance']}  {r['text'][:80].strip()}")
        if not sc.get("found"):
            print("  ", sc.get("error"))
