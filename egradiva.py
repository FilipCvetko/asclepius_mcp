"""ZZZS e-gradiva semantic search module — ChromaDB + sentence-transformers."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _search_egradiva(query: str, n_results: int = 8, category: Optional[str] = None) -> Dict[str, Any]:
    """Semantic search across all ZZZS e-gradiva documents."""
    if COLLECTION is None:
        return {
            "found": False,
            "error": "E-gradiva index not built. Run: python scripts/build_egradiva_index.py",
        }

    query = query.strip()
    if not query:
        return {"found": False, "error": "Query must not be empty"}

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
        return {"found": False, "error": f"Search failed: {e}"}

    if not results["documents"] or not results["documents"][0]:
        return {"found": False, "error": "No matching documents found", "query": query}

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

    return {
        "found": True,
        "count": len(items),
        "query": query,
        "results": items,
        "_citation_instruction": (
            "IMPORTANT: For each result, you MUST (1) quote the relevant portion of the 'text' field verbatim "
            "inside a Markdown blockquote (> ) so it is visually distinct as an unmodified source excerpt, "
            "(2) display the 'reference' field as a citation with clickable source_url, "
            "(3) show the relevance score. Present ALL results, not a summary."
        ),
    }
