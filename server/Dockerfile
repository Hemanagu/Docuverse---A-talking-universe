# ─────────────────────────────────────────────────────────────────────────────
# DocuVerse — Python application image
# Same image is used for both the API server and the arq worker.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System build deps (Pillow, pdfplumber C extensions, faiss-cpu, ffmpeg for pydub)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git curl libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. Python dependencies ────────────────────────────────────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── 2. spaCy model — en_core_web_sm (~12 MB, Pattern 5) ─────────────────
RUN python -m spacy download en_core_web_sm

# ── 3. Pre-cache primary embedding model (Pattern 12 & 14) ─────────────
#    all-MiniLM-L6-v2 is downloaded at build time so the first request is instant.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('✓ Primary embedding model cached (dim=%d)' % m.encode(['test']).shape[1])"

# ── 4. Warm up scikit-learn TF-IDF imports (Pattern 14 fallback) ─────────
RUN python -c "from sklearn.feature_extraction.text import TfidfVectorizer; \
               from sklearn.decomposition import TruncatedSVD; \
               print('✓ TF-IDF fallback ready')"

# ── 5. Verify Numpy (Pattern 14 vector-DB fallback) ─────────────────────
RUN python -c "import numpy; print('✓ Numpy version:', numpy.__version__)"

# ── 6. Copy application source ────────────────────────────────────────────
COPY . .

# ── 7. Runtime directories ────────────────────────────────────────────────
RUN mkdir -p uploads static

# Create empty collections registry if missing
RUN [ -f collections.json ] || echo "[]" > collections.json

EXPOSE 8000

# Default: API server
# Override CMD in docker-compose to run the arq worker
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
