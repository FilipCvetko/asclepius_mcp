"""ZZZS e-gradiva semantic search module — ChromaDB + sentence-transformers."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp.tools.tool import ToolResult

CHROMA_DIR = Path(os.environ.get("CHROMADB_PATH", Path(__file__).parent / "data" / "chromadb"))
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "zzzs_egradiva"

COLLECTION = None
EMBED_FN = None

# Recency-prioritised retrieval: pull a larger candidate pool, then float newer documents up so the
# most up-to-date material takes priority (the periodic refresh stamps each chunk with `doc_date`).
RECENCY_WEIGHT = float(os.environ.get("EGRADIVA_RECENCY_WEIGHT", "0.2"))
_POOL_FACTOR = 5


def _date_ord(s: str):
    import datetime
    try:
        return datetime.date.fromisoformat((s or "")[:10]).toordinal()
    except Exception:
        return None


def initialize_egradiva(embed_fn=None, client=None) -> None:
    """Load pre-built ChromaDB collection. Must run build_egradiva_index.py first.

    `embed_fn` and `client` may be shared singletons (one model + one PersistentClient
    for all ChromaDB-backed modules) — ChromaDB allows only one client per path, so
    parallel per-module clients race and fail.
    """
    global COLLECTION, EMBED_FN

    if not CHROMA_DIR.exists():
        print("WARNING: ChromaDB index not found at", CHROMA_DIR)
        print("  Run: python scripts/build_egradiva_index.py")
        return

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        EMBED_FN = embed_fn or SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = client or chromadb.PersistentClient(path=str(CHROMA_DIR))
        COLLECTION = client.get_collection(name=COLLECTION_NAME, embedding_function=EMBED_FN)
        print(f"Loaded e-gradiva index: {COLLECTION.count()} chunks")
    except Exception as e:
        print(f"WARNING: Failed to load e-gradiva index: {e}")
        print("  Run: python scripts/build_egradiva_index.py")
        COLLECTION = None


def _blockquote(text: str) -> str:
    if not text:
        return ""
    return "\n".join(f"> {line}" for line in text.strip().splitlines())


def _format_egradiva_md(data: dict) -> str:
    if not data.get("found"):
        return f"No results found for \"{data.get('query', '')}\". {data.get('error', '')}"

    lines = [f"**ZZZS e-gradiva** | {data['count']} results for \"{data['query']}\"", ""]

    for i, r in enumerate(data["results"], 1):
        lines.append("---")
        lines.append("")
        title = r.get("doc_title") or "Untitled"
        lines.append(f"### {i}. {title}")
        lines.append("")

        meta_parts = []
        if r.get("category"):
            meta_parts.append(f"**Category**: {r['category']}")
        if r.get("relevance") is not None:
            meta_parts.append(f"**Relevance**: {r['relevance']}")
        if meta_parts:
            lines.append(" | ".join(meta_parts))
            lines.append("")

        if r.get("text"):
            lines.append(_blockquote(r["text"]))
            lines.append("")

        # Source citation (inline, next to the chunk it supports)
        source_url = r.get("source_url", "")
        page = r.get("page_number", 0)
        if source_url:
            link_text = title
            if page:
                link_text += f", str. {page}"
            lines.append(f"**Vir:** [{link_text}]({source_url})")
            lines.append("")

    return "\n".join(lines)


def _search_egradiva(query: str, n_results: int = 8, category: Optional[str] = None) -> ToolResult:
    """Semantic search across all ZZZS e-gradiva documents."""
    if COLLECTION is None:
        err = {"found": False, "error": "E-gradiva index not built. Run: python scripts/build_egradiva_index.py"}
        return ToolResult(content=_format_egradiva_md(err), structured_content=err)

    query = query.strip()
    if not query:
        err = {"found": False, "error": "Query must not be empty"}
        return ToolResult(content=_format_egradiva_md(err), structured_content=err)

    where_filter = None
    if category:
        where_filter = {"category": {"$contains": category.strip()}}

    try:
        pool = min(max(n_results * _POOL_FACTOR, n_results), 50)   # larger pool to re-rank
        results = COLLECTION.query(query_texts=[query], n_results=pool, where=where_filter)
    except Exception as e:
        err = {"found": False, "error": f"Search failed: {e}"}
        return ToolResult(content=_format_egradiva_md(err), structured_content=err)

    if not results["documents"] or not results["documents"][0]:
        err = {"found": False, "error": "No matching documents found", "query": query}
        return ToolResult(content=_format_egradiva_md(err), structured_content=err)

    raw = []
    for i, doc_text in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = results["distances"][0][i] if results["distances"] else None
        relevance = round(1.0 - distance, 4) if distance is not None else 0.0
        source_url = meta.get("file_url", "")
        page_number = meta.get("page_number", 0)
        doc_title = meta.get("doc_title", "")
        ref_parts = [p for p in (doc_title, f"str. {page_number}" if page_number else "") if p]
        reference = (f"📄 {', '.join(ref_parts)} — {source_url}" if ref_parts else f"📄 {source_url}") if source_url else ""
        raw.append({
            "text": doc_text, "relevance": relevance, "doc_title": doc_title,
            "category": meta.get("category", ""), "section": meta.get("section", ""),
            "article_title": meta.get("article_title", ""), "source_url": source_url,
            "page_number": page_number, "doc_date": meta.get("doc_date", ""),
            "reference": reference, "source": meta.get("source", "ZZZS e-gradiva"),
        })

    # Blend similarity with recency (normalised over the pool); newer → higher. Undated = neutral.
    ords = [_date_ord(r["doc_date"]) for r in raw]
    valid = [o for o in ords if o is not None]
    lo, hi = (min(valid), max(valid)) if valid else (0, 0)
    for r, o in zip(raw, ords):
        rnorm = ((o - lo) / (hi - lo)) if (o is not None and hi > lo) else 0.0
        r["_score"] = r["relevance"] + RECENCY_WEIGHT * rnorm
    raw.sort(key=lambda r: -r["_score"])

    items, seen = [], set()                         # dedupe exact repeats (title+page), keep newest/best
    for r in raw:
        key = (r["doc_title"], r["page_number"])
        if key in seen:
            continue
        seen.add(key)
        r.pop("_score", None)
        items.append(r)
        if len(items) >= n_results:
            break

    data = {"found": True, "count": len(items), "query": query, "results": items}
    return ToolResult(content=_format_egradiva_md(data), structured_content=data)
