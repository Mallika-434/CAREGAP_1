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

# requirements.txt has --extra-index-url pointing to the torch CPU wheel
# index, so pip automatically fetches torch==2.1.0+cpu (~200 MB, no CUDA).
RUN pip install --no-cache-dir -r requirements.txt

# ── Application files ─────────────────────────────────────────────
# db.sqlite3 is intentionally copied so the demo works out of the box.
# (It is excluded from the GitHub repo via .gitignore but included here.)
COPY --chown=user:user . .

# Collect static assets for WhiteNoise to serve
RUN python manage.py collectstatic --no-input

# Switch to non-root user
USER user

EXPOSE 7860

# Run migrations → one-time ML setup → gunicorn
CMD python manage.py migrate --no-input && \
    python manage.py setup_demo && \
    exec gunicorn caregap.wsgi:application \
        --bind 0.0.0.0:7860 \
        --workers 1 \
        --timeout 180 \
        --access-logfile -
