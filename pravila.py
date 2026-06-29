"""Pravila OZZ module — authoritative search over the Rules of Mandatory Health Insurance.

Unlike the broad `egradiva` semantic search (which spans ~109k chunks of mixed ZZZS
material, most of it non-authoritative or superseded), this module searches ONLY the
307 atomic articles of the official consolidated Pravila OZZ (NPB42). It uses hybrid
retrieval — dense embeddings (ChromaDB) fused with lexical TF-IDF via Reciprocal Rank
Fusion — plus a deterministic exact-article shortcut, so a provider's reimbursement
question lands on the governing article with an official source citation.
"""

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastmcp.tools.tool import ToolResult

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_DIR = Path(os.environ.get("CHROMADB_PATH", Path(__file__).parent / "data" / "chromadb"))
RULES_FILE = Path(__file__).parent / "data" / "zzzs_rules.json"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "pravila_ozz"
SOURCE_LABEL = "Pravila OZZ NPB42"

# Retrieval params
CANDIDATES = 20          # per-arm candidate pool before fusion
RRF_K = 60               # Reciprocal Rank Fusion constant

# ── Module state ─────────────────────────────────────────────────────────────
COLLECTION = None
EMBED_FN = None
RULES: List[Dict[str, Any]] = []
BY_ARTICLE: Dict[int, Dict[str, Any]] = {}
TFIDF_VECTORIZER: Optional[TfidfVectorizer] = None
TFIDF_MATRIX = None

_ARTICLE_REF = re.compile(r"\b(\d+)\.?\s*člen\b", re.IGNORECASE)
_BARE_NUMBER = re.compile(r"^\s*(\d{1,3})\s*\.?\s*$")


def _doc_text(rule: Dict[str, Any]) -> str:
    """Embedding/representation text for one article, with a context header."""
    section = rule.get("section", "") or rule.get("category", "")
    return f"[{SOURCE_LABEL} | {section}]\n{rule.get('text', '')}"


def initialize_pravila(embed_fn=None, client=None) -> None:
    """Load the 307 Pravila OZZ articles, build the TF-IDF index, and ensure the
    dense ChromaDB collection exists (self-building it on first run / fresh volume).

    `embed_fn` may be passed in to share the already-loaded embedding function with
    the egradiva module and avoid loading the model twice.
    """
    global COLLECTION, EMBED_FN, RULES, BY_ARTICLE, TFIDF_VECTORIZER, TFIDF_MATRIX

    # 1. Load articles
    if not RULES_FILE.exists():
        print("WARNING: Pravila OZZ rules not found at", RULES_FILE)
        return
    with open(RULES_FILE, "r", encoding="utf-8") as f:
        RULES = json.load(f)
    BY_ARTICLE = {
        int(r["article_number"]): r
        for r in RULES
        if str(r.get("article_number", "")).isdigit()
    }
    print(f"Loaded {len(RULES)} Pravila OZZ articles")

    # 2. Lexical (TF-IDF) arm — built in-memory, cheap for 307 docs
    corpus = [
        f"{r.get('title','')} {r.get('section','')} {r.get('category','')} {r.get('text','')}"
        for r in RULES
    ]
    TFIDF_VECTORIZER = TfidfVectorizer()
    TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform(corpus)

    # 3. Dense (ChromaDB) arm — reuse shared embed_fn or load the model
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        EMBED_FN = embed_fn or SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = client or chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            COLLECTION = client.get_collection(name=COLLECTION_NAME, embedding_function=EMBED_FN)
            print(f"Loaded Pravila OZZ index: {COLLECTION.count()} articles")
        except Exception:
            COLLECTION = _build_collection(client)
    except Exception as e:
        print(f"WARNING: Pravila OZZ dense index unavailable ({e}); using lexical-only fallback")
        COLLECTION = None


def _build_collection(client):
    """Embed the 307 articles into a fresh `pravila_ozz` collection (cosine)."""
    print("Building Pravila OZZ dense index...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    coll = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )
    ids, docs, metas = [], [], []
    for r in RULES:
        art = r.get("article_number", 0)
        ids.append(f"pravila_ozz_{art}_{len(ids)}")
        docs.append(_doc_text(r))
        metas.append({
            "article_number": int(art) if str(art).isdigit() else 0,
            "title": r.get("title", ""),
            "section": r.get("section", ""),
            "category": r.get("category", ""),
            "source_url": r.get("source_url", ""),
            "source": SOURCE_LABEL,
        })
    # Batch to stay well within limits (307 fits in one, but batch defensively)
    for i in range(0, len(ids), 100):
        coll.add(ids=ids[i:i + 100], documents=docs[i:i + 100], metadatas=metas[i:i + 100])
    print(f"Pravila OZZ index built: {coll.count()} articles")
    return coll


# ── Retrieval ────────────────────────────────────────────────────────────────

def _rule_to_result(rule: Dict[str, Any], relevance: Optional[float] = None,
                    match: str = "") -> Dict[str, Any]:
    text = rule.get("text", "")
    cross = sorted({
        int(n) for n in _ARTICLE_REF.findall(text)
        if int(n) in BY_ARTICLE and int(n) != rule.get("article_number")
    })
    art = rule.get("article_number")
    src = rule.get("source_url", "")
    return {
        "article_number": art,
        "title": rule.get("title", ""),
        "section": rule.get("section", ""),
        "category": rule.get("category", ""),
        "text": text,
        "source_url": src,
        "citation": f"{SOURCE_LABEL}, {art}. člen — {src}" if (art and src) else (src or SOURCE_LABEL),
        "cross_refs": cross,
        "relevance": relevance,
        "match": match,
    }


def _dense_ranked(query: str) -> List[int]:
    """Return article_numbers ranked by dense similarity (best first)."""
    if COLLECTION is None:
        return []
    try:
        res = COLLECTION.query(query_texts=[query], n_results=CANDIDATES)
    except Exception:
        return []
    metas = res.get("metadatas") or [[]]
    return [m.get("article_number") for m in (metas[0] if metas else [])]


def _lexical_ranked(query: str) -> List[int]:
    """Return article_numbers ranked by TF-IDF cosine (best first)."""
    if TFIDF_VECTORIZER is None or TFIDF_MATRIX is None:
        return []
    qv = TFIDF_VECTORIZER.transform([query])
    sims = cosine_similarity(qv, TFIDF_MATRIX).flatten()
    order = sims.argsort()[::-1][:CANDIDATES]
    return [RULES[i].get("article_number") for i in order if sims[i] > 0]


def _rrf_fuse(*ranked_lists: List[int]) -> List[int]:
    """Reciprocal Rank Fusion across arms → article_numbers, best first."""
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, art in enumerate(ranked):
            if art is None:
                continue
            scores[art] = scores.get(art, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [a for a, _ in sorted(scores.items(), key=lambda x: -x[1])]


def _search_pravila(query: str, n_results: int = 8) -> ToolResult:
    """Hybrid (dense + TF-IDF, RRF-fused) search over Pravila OZZ, with an exact-article shortcut."""
    if not RULES:
        err = {"found": False, "error": "Pravila OZZ data not loaded"}
        return ToolResult(content=_format_md(err), structured_content=err)

    query = (query or "").strip()
    if not query:
        err = {"found": False, "error": "Query must not be empty"}
        return ToolResult(content=_format_md(err), structured_content=err)

    results: List[Dict[str, Any]] = []
    seen: set = set()

    # 1. Exact-article shortcut (deterministic): "23. člen", "člen 23", or a bare number
    exact_nums: List[int] = [int(n) for n in _ARTICLE_REF.findall(query)]
    bare = _BARE_NUMBER.match(query)
    if bare:
        exact_nums.append(int(bare.group(1)))
    for n in exact_nums:
        if n in BY_ARTICLE and n not in seen:
            results.append(_rule_to_result(BY_ARTICLE[n], relevance=1.0, match="exact-article"))
            seen.add(n)

    # 2. + 3. + 4. Hybrid: dense ⊕ lexical, fused by RRF
    fused = _rrf_fuse(_dense_ranked(query), _lexical_ranked(query))
    for art in fused:
        if art in seen or art not in BY_ARTICLE:
            continue
        results.append(_rule_to_result(BY_ARTICLE[art], match="hybrid"))
        seen.add(art)
        if len(results) >= n_results:
            break

    if not results:
        err = {"found": False, "error": "No matching articles found", "query": query}
        return ToolResult(content=_format_md(err), structured_content=err)

    data = {
        "found": True,
        "count": len(results),
        "query": query,
        "source": SOURCE_LABEL,
        "results": results,
    }
    return ToolResult(content=_format_md(data), structured_content=data)


# ── Formatting ───────────────────────────────────────────────────────────────

def _blockquote(text: str) -> str:
    if not text:
        return ""
    return "\n".join(f"> {line}" for line in text.strip().splitlines())


def _format_md(data: dict) -> str:
    if not data.get("found"):
        return (f"V Pravilih OZZ ni zadetkov za \"{data.get('query','')}\". "
                f"{data.get('error','')}").strip()

    lines = [
        f"**Pravila OZZ** (NPB42 — uradno prečiščeno besedilo) | "
        f"{data['count']} zadetkov za \"{data['query']}\"",
        "",
    ]
    for r in data["results"]:
        lines.append("---")
        lines.append("")
        art = r.get("article_number")
        title = r.get("title") or "Člen"
        lines.append(f"### Čl. {art}: {title}" if art else f"### {title}")
        lines.append("")

        meta = []
        if r.get("section"):
            meta.append(f"**Razdelek**: {r['section']}")
        if r.get("category"):
            meta.append(f"**Kategorija**: {r['category']}")
        if r.get("relevance") is not None:
            meta.append(f"**Relevantnost**: {round(r['relevance'], 4)}")
        if meta:
            lines.append(" | ".join(meta))
            lines.append("")

        if r.get("text"):
            lines.append(_blockquote(r["text"]))
            lines.append("")

        if r.get("cross_refs"):
            refs = ", ".join(f"{n}. člen" for n in r["cross_refs"])
            lines.append(f"*Sklicuje se na:* {refs}")
            lines.append("")

        if r.get("source_url"):
            art = r.get("article_number")
            label = f"{SOURCE_LABEL}, {art}. člen" if art else SOURCE_LABEL
            lines.append(f"**Vir:** [{label}]({r['source_url']})")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("_Vir je uradno prečiščeno besedilo Pravil OZZ (NPB42). "
                 "Pri sklicevanju preveri, ali obstaja novejše prečiščeno besedilo._")
    return "\n".join(lines)


if __name__ == "__main__":
    initialize_pravila()
    for q in ["veljavnost napotnice", "povračilo potnih stroškov",
              "doplačilo za zobni most", "23. člen"]:
        print("\n" + "=" * 60)
        print("QUERY:", q)
        out = _search_pravila(q, n_results=3)
        sc = out.structured_content
        if sc.get("found"):
            for r in sc["results"]:
                print(f"  Čl. {r['article_number']}: {r['title']}  "
                      f"[{r['match']}] ({r.get('category')})")
        else:
            print("  ", sc.get("error"))
