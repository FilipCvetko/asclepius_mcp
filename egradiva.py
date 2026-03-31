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


def initialize_egradiva() -> None:
    """Load pre-built ChromaDB collection. Must run build_egradiva_index.py first."""
    global COLLECTION, EMBED_FN

    if not CHROMA_DIR.exists():
        print("WARNING: ChromaDB index not found at", CHROMA_DIR)
        print("  Run: python scripts/build_egradiva_index.py")
        return

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        EMBED_FN = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
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

        # Source link
        source_url = r.get("source_url", "")
        page = r.get("page_number", 0)
        if source_url:
            link_text = title
            if page:
                link_text += f", str. {page}"
            lines.append(f"[{link_text}]({source_url})")
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
        results = COLLECTION.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )
    except Exception as e:
        err = {"found": False, "error": f"Search failed: {e}"}
        return ToolResult(content=_format_egradiva_md(err), structured_content=err)

    if not results["documents"] or not results["documents"][0]:
        err = {"found": False, "error": "No matching documents found", "query": query}
        return ToolResult(content=_format_egradiva_md(err), structured_content=err)

    items = []
    for i, doc_text in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = results["distances"][0][i] if results["distances"] else None
        relevance = round(1.0 - distance, 4) if distance is not None else None

        source_url = meta.get("file_url", "")
        page_number = meta.get("page_number", 0)
        doc_title = meta.get("doc_title", "")
        reference = ""
        if source_url:
            ref_parts = []
            if doc_title:
                ref_parts.append(doc_title)
            if page_number:
                ref_parts.append(f"str. {page_number}")
            reference = f"📄 {', '.join(ref_parts)} — {source_url}" if ref_parts else f"📄 {source_url}"

        items.append({
            "text": doc_text,
            "relevance": relevance,
            "doc_title": doc_title,
            "category": meta.get("category", ""),
            "section": meta.get("section", ""),
            "article_title": meta.get("article_title", ""),
            "source_url": source_url,
            "page_number": page_number,
            "reference": reference,
            "source": meta.get("source", "ZZZS e-gradiva"),
        })

    data = {
        "found": True,
        "count": len(items),
        "query": query,
        "results": items,
    }
    return ToolResult(content=_format_egradiva_md(data), structured_content=data)
