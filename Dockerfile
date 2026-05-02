# ── Base: slim saves ~200 MB vs full python:3.10
FROM python:3.10-slim-bookworm

# ── libglib2.0-0: required by Pillow image codecs at runtime
# ── Clean apt cache in the same layer to keep the layer small
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ────────────────────────────────────────────────────────────────────────────
# Layer A — typing-extensions from PyPI  (tiny, ~1 MB)
# ────────────────────────────────────────────────────────────────────────────
# WHY A SEPARATE LAYER:
#   The pytorch CPU wheel index hosts this package as "typing_extensions"
#   (underscore), but pip requires it as "typing-extensions" (hyphen).
#   pip sees the name mismatch, discards every candidate, and fails.
#   Pre-installing from PyPI first means pip sees "Requirement already
#   satisfied" during the torch install and never touches the pytorch index
#   for this package.
RUN pip install --no-cache-dir "typing-extensions>=4.10.0"

# ────────────────────────────────────────────────────────────────────────────
# Layer B — PyTorch CPU wheels  (~700 MB, cached unless version changes)
# ────────────────────────────────────────────────────────────────────────────
# --index-url replaces PyPI entirely so pip picks the +cpu builds, not the
# 2 GB CUDA builds. typing-extensions is already installed (Layer A) so
# pip's "Requirement already satisfied" check fires and skips the index.
RUN pip install --no-cache-dir \
        torch==2.11.0 \
        torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ────────────────────────────────────────────────────────────────────────────
# Layer C — Lightweight API dependencies  (~20 MB)
# ────────────────────────────────────────────────────────────────────────────
# Copied before application code so this layer is a cache-hit when only
# api/app.py changes (the most common edit during development).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# ────────────────────────────────────────────────────────────────────────────
# Layer D — Application code + baked model  (changes most often → last)
# ────────────────────────────────────────────────────────────────────────────
# .dockerignore excludes data/, src/, notebooks/, tests/ so only api/ is
# sent to the build context, keeping the context upload fast.
COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]