#!/usr/bin/env python3
"""Build the standalone `pravila_ozz` ChromaDB collection from data/zzzs_rules.json.

This is the authoritative Pravila OZZ (NPB42) corpus — 307 atomic articles, isolated
from the broad e-gradiva index so the dedicated `get_pravila_ozz` tool searches only
authoritative rules. Idempotent: deletes and rebuilds the collection in place.

Note: `pravila.initialize_pravila()` self-builds this collection at startup if it's
missing, so running this script is optional (useful for prebuilding into the image
seed or rebuilding after zzzs_rules.json changes).

Usage:  python scripts/build_pravila_index.py
"""

import sys
from pathlib import Path

# Allow importing the project module when run from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import pravila  # noqa: E402


def main() -> None:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    # Load articles + TF-IDF (also sets pravila.RULES used by _build_collection)
    pravila.initialize_pravila()

    if not pravila.RULES:
        print("ERROR: no Pravila OZZ articles loaded; check data/zzzs_rules.json")
        sys.exit(1)

    # Force a clean rebuild of the dense collection
    pravila.EMBED_FN = SentenceTransformerEmbeddingFunction(model_name=pravila.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(pravila.CHROMA_DIR))
    coll = pravila._build_collection(client)

    print(f"\nDone: collection '{pravila.COLLECTION_NAME}' has {coll.count()} articles "
          f"at {pravila.CHROMA_DIR}")


if __name__ == "__main__":
    main()
