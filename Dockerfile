# ── CareGap — Hugging Face Spaces Dockerfile ──────────────────────
# Port 7860 is required by HF Spaces.
# Non-root user (uid=1000) is required by HF Spaces Docker SDK.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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

# requirements.txt has --extra-index-url pointing to the torch CPU wheel
# index, so pip automatically fetches torch==2.1.0+cpu (~200 MB, no CUDA).
RUN pip install --no-cache-dir -r requirements.txt

# ── Application files ─────────────────────────────────────────────
# db.sqlite3 is intentionally copied so the demo works out of the box.
# (It is excluded from the GitHub repo via .gitignore but included here.)
COPY --chown=user:user . .

# Collect static assets for WhiteNoise to serve
RUN python manage.py collectstatic --no-input

# Copy SQLite DB to /tmp so the non-root user can write to it at runtime
RUN cp /app/db.sqlite3 /tmp/db.sqlite3 && chmod 666 /tmp/db.sqlite3

# Create writable cache dir before switching to non-root user
RUN mkdir -p /tmp/caregap_cache && chmod 777 /tmp/caregap_cache

# Environment variables for HF Spaces runtime
ENV DB_PATH=/tmp/db.sqlite3
ENV ENVIRONMENT=hf_spaces
ENV DEPLOYMENT_MODE=demo
ENV DEBUG=False
ENV ALLOWED_HOSTS=.hf.space,localhost,127.0.0.1
ENV CSRF_TRUSTED_ORIGINS=https://*.hf.space
ENV USE_X_FORWARDED_HOST=True
ENV OLLAMA_MODEL=phi3:latest
ENV CARE_GAP_CACHE_DIR=/tmp/caregap_cache
ENV WARM_CACHE_ON_START=false

# Switch to non-root user
USER user

EXPOSE 7860

# Run migrations → one-time ML setup → gunicorn
CMD python manage.py migrate --no-input && \
    if [ "$WARM_CACHE_ON_START" = "true" ]; then python manage.py warm_cache; fi && \
    exec gunicorn caregap.wsgi:application \
        --bind 0.0.0.0:7860 \
        --workers 1 \
        --timeout 180 \
        --access-logfile -
