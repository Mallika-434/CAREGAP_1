# ── CareGap — Hugging Face Spaces Dockerfile ──────────────────────
# Port 7860 is required by HF Spaces.
# Non-root user (uid=1000) is required by HF Spaces Docker SDK.
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user expected by HF Spaces
RUN useradd -m -u 1000 user
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────
COPY requirements.txt .

# Install CPU-only PyTorch first (~200 MB instead of ~2.5 GB CUDA build).
RUN pip install --no-cache-dir \
        torch==2.6.0 \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ── Application files ─────────────────────────────────────────────
COPY --chown=user:user . .

# ── Build-time database setup ─────────────────────────────────────
# 1. Create tables (produces db_demo.sqlite3)
RUN python manage.py migrate --no-input

# 2. Seed 8 demo patients so train_models has data at first startup
RUN python manage.py seed_demo_data

# 3. Build FAISS knowledge index (reads static knowledge files, not DB)
RUN python manage.py build_rag_index || true

# 4. Collect static assets for WhiteNoise
RUN python manage.py collectstatic --no-input

# Fix ownership: build-time RUN commands run as root, so db and
# generated files must be re-chowned before switching to non-root user
RUN chown -R user:user /app

# Switch to non-root user
USER user

EXPOSE 7860

# At container startup:
#   migrate  — no-op on normal restarts; picks up any new migrations on redeploy
#   setup_demo — trains ML models on seeded data + warms dashboard cache
#   gunicorn — serves the app
CMD python manage.py migrate --no-input && \
    python manage.py setup_demo && \
    python manage.py precompute_forecast && \
    exec gunicorn caregap.wsgi:application \
        --bind 0.0.0.0:7860 \
        --workers 1 \
        --timeout 300 \
        --access-logfile -
