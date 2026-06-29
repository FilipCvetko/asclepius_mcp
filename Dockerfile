FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first (skips ~1.3GB of CUDA), then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# --- Runtime stage (drops build-essential) ---
FROM python:3.13-slim

WORKDIR /app

# OCR (scanned PDFs/.tif, Slovenian) + .doc extraction for the in-process periodic refresh.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-slv wv && rm -rf /var/lib/apt/lists/*

# Copy installed packages and model cache from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /root/.cache

# Copy application code (incl. the refresh orchestrator + path helper)
COPY main.py cache.py contacts.py drugs.py egradiva.py pravila.py obracun.py sifranti.py mtp.py icd10.py spa.py zzzs.py templates.py refresh.py paths.py ./

# Build/fetch scripts the refresh imports (crawl/extract/chunk + catalog builders)
COPY scripts/build_egradiva_index.py scripts/fetch_sifranti.py scripts/build_sifrant_catalog.py \
     scripts/fetch_mtp.py scripts/build_mtp_catalog.py ./scripts/

# Copy small data files needed at runtime (+ MTP exclusion/timska companions for refresh rebuilds)
COPY data/icd10_codes.json data/zzzs_rules.json data/sifrant_catalog.json data/mtp_catalog.json data/
COPY data/mtp_companions/ data/mtp_companions/

# Copy ChromaDB as seed for initial volume population
COPY data/chromadb/ /seed/chromadb/

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
