#!/usr/bin/env python3
"""Crawl ZZZS e-gradiva, download documents, extract text, chunk, and build ChromaDB index."""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
FILES_DIR = DATA_DIR / "egradiva_files"
MANIFEST_FILE = DATA_DIR / "egradiva_manifest.json"
RULES_FILE = DATA_DIR / "zzzs_rules.json"
CHROMA_DIR = DATA_DIR / "chromadb"
ERROR_LOG = DATA_DIR / "egradiva_errors.log"

BASE_API = "https://www.zzzs.si/zzzs-api/e-gradiva/vsa-gradiva/"
DOMINO_BASE = "https://api.zzzs.si/ZZZS/info/egradiva.nsf/0"
COLLECTION_NAME = "zzzs_egradiva"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
ARTICLE_MAX_CHARS = 2000
ARTICLE_PATTERN = re.compile(r"\d+\.?[a-z]?\s*člen\b", re.IGNORECASE)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Step 1: API Crawl ────────────────────────────────────────────────────────

def crawl_categories() -> list:
    """Fetch all e-gradiva categories from ZZZS API.
    Returns list of {code, name} dicts.
    API returns dict of {code: {SifraGradiva, VrstaGradiva}}.
    """
    url = f"{BASE_API}?ajax=1&act=get-vrste"
    log.info("Fetching categories from %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Response is {code: {SifraGradiva: ..., VrstaGradiva: ...}}
    categories = []
    if isinstance(data, dict):
        for code, info in data.items():
            if isinstance(info, dict):
                categories.append({
                    "code": info.get("SifraGradiva", code),
                    "name": info.get("VrstaGradiva", code),
                })
    log.info("Found %d categories", len(categories))
    return categories


def crawl_documents_for_category(category: dict) -> list:
    """Fetch all documents in a category.
    API returns list of {@unid, NASLOV, DATUM, VrstaGradiva}.
    """
    code = category["code"]
    name = category["name"]
    url = f"{BASE_API}?ajax=1&act=get-egradiva-by-vrsta&key={code}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        docs = data if isinstance(data, list) else []
        log.info("  Category '%s' (%s): %d documents", name, code, len(docs))
        return [{"category_code": code, "category_name": name, **d} for d in docs]
    except Exception as e:
        log.error("  Failed to fetch category '%s': %s", name, e)
        return []


def crawl_document_files(doc: dict) -> list:
    """Get file URLs for a document by parsing its Domino HTML page.
    Files are linked as /ZZZS/info/egradiva.nsf/0/{unid}/$FILE/{filename}.
    """
    unid = doc.get("@unid") or doc.get("unid") or ""
    if not unid:
        return []
    url = f"{DOMINO_BASE}/{unid}?OpenDocument"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return []
        # Extract $FILE links from HTML
        file_links = re.findall(
            r'href=["\']([^"\']*\$FILE[^"\']*)["\']',
            resp.text, re.IGNORECASE,
        )
        results = []
        seen = set()
        for link in file_links:
            if link.startswith("/"):
                full_url = f"https://api.zzzs.si{link}"
            elif link.startswith("http"):
                full_url = link
            else:
                full_url = f"https://api.zzzs.si/ZZZS/info/egradiva.nsf/0/{unid}/{link}"
            if full_url not in seen:
                seen.add(full_url)
                # Determine file type
                lower = full_url.lower()
                if ".pdf" in lower:
                    file_type = "pdf"
                elif ".docx" in lower:
                    file_type = "docx"
                elif ".doc" in lower:
                    file_type = "doc"
                else:
                    file_type = "other"
                results.append({
                    "file_url": full_url,
                    "file_type": file_type,
                })
        return results
    except Exception as e:
        log.warning("  Failed to get files for %s: %s", unid, e)
        return []


def build_manifest_entry(doc: dict, file_info: dict) -> dict:
    """Build a normalized manifest entry."""
    unid = doc.get("@unid") or doc.get("unid") or ""
    title = doc.get("NASLOV") or doc.get("naslov") or doc.get("title") or ""
    return {
        "unid": unid,
        "title": title.strip() if isinstance(title, str) else str(title),
        "category_code": doc.get("category_code", ""),
        "category_name": doc.get("category_name", ""),
        "file_url": file_info.get("file_url"),
        "file_type": file_info.get("file_type"),
        "local_path": None,
    }


def run_crawl():
    """Full API crawl → manifest JSON."""
    categories = crawl_categories()

    all_docs = []
    for cat in categories:
        docs = crawl_documents_for_category(cat)
        all_docs.extend(docs)
        time.sleep(0.3)

    log.info("Total documents across all categories: %d", len(all_docs))

    # For each document, fetch file URLs from its Domino page
    manifest = []
    seen_urls = set()
    total = len(all_docs)
    for i, doc in enumerate(all_docs):
        if (i + 1) % 50 == 0 or i == 0:
            pct = (i + 1) / total * 100
            log.info("  Crawling file URLs: %d/%d (%.0f%%)", i + 1, total, pct)
        files = crawl_document_files(doc)
        if files:
            for f in files:
                url = f["file_url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    manifest.append(build_manifest_entry(doc, f))
        else:
            # Keep entry without file_url (document exists but no downloadable files)
            manifest.append(build_manifest_entry(doc, {}))
        time.sleep(0.3)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    downloadable = sum(1 for e in manifest if e.get("file_url"))
    log.info("Manifest saved: %d entries (%d with file URLs)", len(manifest), downloadable)
    return manifest


# ── Step 2: Download ─────────────────────────────────────────────────────────

def download_files(manifest: list) -> list:
    """Download all files from manifest, skipping already-downloaded ones."""
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    error_log = open(ERROR_LOG, "a", encoding="utf-8")

    downloaded = 0
    skipped = 0
    errors = 0

    downloadable = [e for e in manifest if e.get("file_url")]
    total = len(downloadable)
    log.info("Files to process: %d", total)

    for i, entry in enumerate(downloadable):
        url = entry["file_url"]
        ext = entry.get("file_type", "pdf") or "pdf"
        unid = entry.get("unid", "unknown")
        local_path = FILES_DIR / f"{unid}.{ext}"

        if local_path.exists() and local_path.stat().st_size > 0:
            entry["local_path"] = str(local_path)
            skipped += 1
        else:
            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=60, stream=True)
                    resp.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    entry["local_path"] = str(local_path)
                    downloaded += 1
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        log.error("Failed to download %s: %s", url, e)
                        error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} DOWNLOAD_ERROR {url}: {e}\n")
                        errors += 1
            time.sleep(0.5)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            log.info("  Download progress: %d/%d (%.0f%%) — %d new, %d skipped, %d errors",
                     i + 1, total, pct, downloaded, skipped, errors)

    error_log.close()

    # Save updated manifest with local paths
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("Download complete: %d downloaded, %d skipped, %d errors", downloaded, skipped, errors)
    return manifest


# ── Step 3: Text Extraction ──────────────────────────────────────────────────

def extract_pdf_text(path: str) -> list:
    """Extract text from PDF, returning list of (page_number, text) tuples."""
    import fitz
    pages = []
    try:
        doc = fitz.open(path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            # Strip common headers/footers
            lines = text.split("\n")
            cleaned = []
            for line in lines:
                stripped = line.strip()
                if re.match(r"^\d+$", stripped):
                    continue
                if re.match(r"^Stran \d+ od \d+", stripped):
                    continue
                cleaned.append(line)
            page_text = "\n".join(cleaned).strip()
            if len(page_text) >= 50:  # Skip near-empty pages (possibly scanned)
                pages.append((page_num + 1, page_text))
            else:
                if page_text:
                    log.debug("Skipping page %d of %s: possibly scanned (%d chars)", page_num + 1, path, len(page_text))
        doc.close()
    except Exception as e:
        log.error("Failed to extract PDF %s: %s", path, e)
    return pages


def extract_docx_text(path: str) -> list:
    """Extract text from DOCX, returning list of (page_number, text) tuples."""
    from docx import Document
    try:
        doc = Document(path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if len(full_text) >= 50:
            return [(1, full_text)]  # DOCX doesn't have reliable page numbers
    except Exception as e:
        log.error("Failed to extract DOCX %s: %s", path, e)
    return []


def extract_text_from_file(path: str, file_type: str) -> list:
    """Extract text from a file. Returns list of (page_number, text)."""
    if file_type == "pdf":
        return extract_pdf_text(path)
    elif file_type == "docx":
        return extract_docx_text(path)
    return []


# ── Step 4: Chunking ─────────────────────────────────────────────────────────

def chunk_by_articles(text: str, doc_title: str, category: str, doc_id: str,
                      file_url: str, page_number: int) -> list:
    """Split text by article boundaries (for legal documents)."""
    matches = list(ARTICLE_PATTERN.finditer(text))
    if len(matches) < 10:
        return []  # Not a legal-structured document

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()
        article_title = m.group(0).strip()

        if len(article_text) <= ARTICLE_MAX_CHARS:
            chunks.append({
                "text": f"[{doc_title} | {category}]\n{article_text}",
                "metadata": {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "category": category,
                    "file_url": file_url,
                    "page_number": page_number,
                    "chunk_index": len(chunks),
                    "article_title": article_title,
                    "source": "ZZZS e-gradiva",
                },
            })
        else:
            # Sub-split large articles
            sub_chunks = sliding_window_chunk(
                article_text, doc_title, category, doc_id, file_url, page_number,
                prefix=f"{article_title}: ",
            )
            for sc in sub_chunks:
                sc["metadata"]["article_title"] = article_title
            chunks.extend(sub_chunks)

    return chunks


def sliding_window_chunk(text: str, doc_title: str, category: str, doc_id: str,
                         file_url: str, page_number: int, prefix: str = "") -> list:
    """Sliding window chunking with smart boundary detection."""
    chunks = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        end = min(pos + CHUNK_SIZE, text_len)

        if end < text_len:
            # Try to break at paragraph boundary
            para_break = text.rfind("\n\n", pos + CHUNK_SIZE // 2, end)
            if para_break > pos:
                end = para_break
            else:
                # Try sentence boundary
                sent_break = text.rfind(". ", pos + CHUNK_SIZE // 2, end)
                if sent_break > pos:
                    end = sent_break + 1

        chunk_text = text[pos:end].strip()
        if chunk_text:
            header = f"[{doc_title} | {category} | str. {page_number}]"
            chunks.append({
                "text": f"{header}\n{prefix}{chunk_text}",
                "metadata": {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "category": category,
                    "file_url": file_url,
                    "page_number": page_number,
                    "chunk_index": len(chunks),
                    "article_title": "",
                    "source": "ZZZS e-gradiva",
                },
            })

        pos = end - CHUNK_OVERLAP if end < text_len else text_len

    return chunks


def chunk_document(entry: dict) -> list:
    """Extract text and chunk a single document."""
    path = entry.get("local_path")
    file_type = entry.get("file_type", "pdf")
    if not path or not Path(path).exists():
        return []

    pages = extract_text_from_file(path, file_type)
    if not pages:
        return []

    doc_id = entry.get("unid", "unknown")
    doc_title = entry.get("title", "Unknown document")
    category = entry.get("category_name", "")
    file_url = entry.get("file_url", "")

    # Combine all text to check for article structure
    full_text = "\n\n".join(text for _, text in pages)

    # Try article-level chunking first
    article_chunks = chunk_by_articles(
        full_text, doc_title, category, doc_id, file_url,
        page_number=pages[0][0] if pages else 1,
    )
    if article_chunks:
        return article_chunks

    # Fall back to sliding window per page
    all_chunks = []
    for page_num, page_text in pages:
        page_chunks = sliding_window_chunk(
            page_text, doc_title, category, doc_id, file_url, page_num,
        )
        all_chunks.extend(page_chunks)

    return all_chunks


# ── Step 5: Migrate Pravila OZZ ──────────────────────────────────────────────

def migrate_pravila_ozz() -> list:
    """Convert existing zzzs_rules.json into chunks for ChromaDB."""
    if not RULES_FILE.exists():
        log.warning("No zzzs_rules.json found, skipping Pravila OZZ migration")
        return []

    with open(RULES_FILE, "r", encoding="utf-8") as f:
        rules = json.load(f)

    chunks = []
    for i, rule in enumerate(rules):
        art_num = rule.get("article_number", 0)
        header = f"[Pravila OZZ NPB42 | {rule.get('category', '')}]"
        chunks.append({
            "text": f"{header}\n{rule.get('text', '')}",
            "id": f"pravila_ozz_npb42_chunk_{i}",
            "metadata": {
                "doc_id": "pravila_ozz_npb42",
                "doc_title": "Pravila OZZ NPB42",
                "category": rule.get("category", ""),
                "file_url": rule.get("source_url", ""),
                "page_number": 0,
                "chunk_index": art_num,
                "article_title": rule.get("title", ""),
                "section": rule.get("section", ""),
                "source": "Pravila OZZ NPB42",
            },
        })

    log.info("Migrated %d Pravila OZZ articles", len(chunks))
    return chunks


# ── Step 6: Build ChromaDB Index ─────────────────────────────────────────────

def build_index(manifest: list):
    """Chunk all documents and upsert into ChromaDB."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    log.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection to rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Chunk all downloaded documents
    all_chunks = []
    processable = [e for e in manifest if e.get("local_path")]
    total_docs = len(processable)
    log.info("Chunking %d documents...", total_docs)
    for i, entry in enumerate(processable):
        doc_chunks = chunk_document(entry)
        all_chunks.extend(doc_chunks)
        if (i + 1) % 50 == 0 or (i + 1) == total_docs:
            pct = (i + 1) / total_docs * 100
            log.info("  Chunking progress: %d/%d (%.0f%%) — %d chunks so far",
                     i + 1, total_docs, pct, len(all_chunks))

    log.info("Generated %d chunks from %d documents", len(all_chunks), total_docs)

    # Migrate Pravila OZZ rules
    pravila_chunks = migrate_pravila_ozz()
    all_chunks.extend(pravila_chunks)

    log.info("Total chunks (including Pravila OZZ): %d", len(all_chunks))

    if not all_chunks:
        log.warning("No chunks to index!")
        return

    # Batch upsert
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]

        ids = []
        documents = []
        metadatas = []

        for j, chunk in enumerate(batch):
            chunk_id = chunk.get("id") or f"{chunk['metadata']['doc_id']}_chunk_{i + j}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            # Ensure all metadata values are strings or numbers (ChromaDB requirement)
            meta = {}
            for k, v in chunk["metadata"].items():
                if v is None:
                    meta[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        done = min(i + batch_size, len(all_chunks))
        pct = done / len(all_chunks) * 100
        if (i // batch_size) % 10 == 0 or done == len(all_chunks):
            log.info("  Embedding + indexing: %d/%d (%.0f%%)", done, len(all_chunks), pct)

    log.info("ChromaDB index built: %d chunks in collection '%s'", collection.count(), COLLECTION_NAME)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build ZZZS e-gradiva search index")
    parser.add_argument("--crawl", action="store_true", help="API crawl only")
    parser.add_argument("--download", action="store_true", help="Download files only")
    parser.add_argument("--index", action="store_true", help="Chunk + embed + store only")
    args = parser.parse_args()

    # If no flags, run full pipeline
    run_all = not (args.crawl or args.download or args.index)

    manifest = None

    if args.crawl or run_all:
        log.info("=== Step 1: API Crawl ===")
        manifest = run_crawl()

    if args.download or run_all:
        log.info("=== Step 2: Download Files ===")
        if manifest is None:
            if MANIFEST_FILE.exists():
                with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            else:
                log.error("No manifest found. Run --crawl first.")
                sys.exit(1)
        manifest = download_files(manifest)

    if args.index or run_all:
        log.info("=== Step 3: Build Index ===")
        if manifest is None:
            if MANIFEST_FILE.exists():
                with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            else:
                log.error("No manifest found. Run --crawl first.")
                sys.exit(1)
        build_index(manifest)

    log.info("Done!")


if __name__ == "__main__":
    main()
