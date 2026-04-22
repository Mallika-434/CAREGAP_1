---
title: CareGap
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# CareGap - Healthcare Analytics Platform
Patient Risk Management System built with Django.
33,990 Synthea patients | Saint Louis University | Group 5

## Local setup

1. Create a virtual environment and install the packages from `requirements.txt`.
2. Copy `.env.example` to `.env` and fill in any local secrets you need.
3. Run `python manage.py migrate`.
4. Start the app with `python manage.py runserver`.

## Deployment modes

- `DEPLOYMENT_MODE=internal`: for staff or private/LAN use. Local LLM providers are enabled in this mode, so MedGemma and Ollama can be used when configured.
- `DEPLOYMENT_MODE=demo`: for the free/public demo path. Local LLM providers are skipped, so the app relies on Gemini if configured, otherwise rule-based fallbacks.
- The default in `.env.example` is `internal`. The Docker Space defaults to `demo`.

## Production settings

- Set `ENVIRONMENT=production` and provide `SECRET_KEY` in the deployment environment. The app now refuses to start in production with the default development key.
- Set `ALLOWED_HOSTS` to your deployed hostname list, separated by commas.
- Keep `CORS_ALLOW_ALL_ORIGINS=False` in production and set `CORS_ALLOWED_ORIGINS` and `CSRF_TRUSTED_ORIGINS` explicitly.
- If your platform terminates HTTPS before the app, keep `USE_X_FORWARDED_HOST=True` and configure secure cookies and SSL redirect as needed for that platform.

## Quality and safety notes

- Keep real API keys out of source control and out of shared screenshots or archives.
- If your environment cannot write to the default cache path, set `CARE_GAP_CACHE_DIR` in `.env`.
- The bundled SQLite databases and model files are useful demo assets, but they are not a production storage strategy.
- Core patient risk logic and urgent-care matching now have automated tests under `patients/tests/`.

## RAG deployment modes

- Preferred healthcare-aware local setup: expose a MedGemma-compatible endpoint with `MEDGEMMA_URL` and set `MEDGEMMA_MODEL`.
- Secondary local fallback: run Ollama and set `OLLAMA_MODEL=phi3:latest` (or your chosen local model).
- Optional cloud fallback: set `GEMINI_API_KEY` as a deployment secret if you want Gemini available when local models are down or unreachable.
- Safe default behavior: if neither MedGemma, Ollama, nor Gemini is available, the app still returns deterministic rule-based suggestions and explanations.
- Do not commit real values in `.env`; keep real deployment secrets in your hosting platform config.

## Hugging Face Spaces

- This repo is configured for a Docker Space on port `7860`.
- The Space defaults to `DEPLOYMENT_MODE=demo`, so it does not require Ollama or MedGemma to be available.
- Add a Space secret for `SECRET_KEY` before deployment.
- Optional secret: `GEMINI_API_KEY` if you want Gemini fallback in the RAG layer.
- Startup on Spaces now runs migrations and starts Gunicorn without retraining models on every restart.
- If you want the analytics cache prewarmed on boot, set `WARM_CACHE_ON_START=true` in the Space variables.
