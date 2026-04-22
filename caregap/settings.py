"""
CareGap Analytics — Django Settings
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from project root (if present).
# Variables already set in the environment take precedence.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development').strip().lower()
DEPLOYMENT_MODE = os.environ.get('DEPLOYMENT_MODE', 'internal').strip().lower()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return list(default or [])
    return [item.strip() for item in value.split(',') if item.strip()]

SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    'caregap-dev-secret-key-change-in-production',
)
DEBUG = _env_bool('DEBUG', default=False)
ALLOWED_HOSTS = _env_list(
    'ALLOWED_HOSTS',
    default=['localhost', '127.0.0.1', 'testserver', '.ngrok-free.app', '.ngrok.io', '.ngrok-free.dev'],
)

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'patients',
    'rag',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'caregap.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'caregap.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.environ.get('DB_PATH', str(BASE_DIR / 'db.sqlite3')),
    }
}

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
_static_dir = BASE_DIR / 'static'
STATICFILES_DIRS = [_static_dir] if _static_dir.exists() else []

# Django 4.2+ STORAGES dict replaces the old STATICFILES_STORAGE setting.
# WhiteNoise compresses + fingerprints static files for efficient serving.
STORAGES = {
    'default': {
        'BACKEND': 'django.core.files.storage.FileSystemStorage',
    },
    'staticfiles': {
        'BACKEND': 'whitenoise.storage.CompressedManifestStaticFilesStorage',
    },
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

CORS_ALLOW_ALL_ORIGINS = _env_bool('CORS_ALLOW_ALL_ORIGINS', default=DEBUG)
CORS_ALLOWED_ORIGINS = _env_list('CORS_ALLOWED_ORIGINS')
CSRF_TRUSTED_ORIGINS = _env_list('CSRF_TRUSTED_ORIGINS')

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
USE_X_FORWARDED_HOST = _env_bool('USE_X_FORWARDED_HOST', default=not DEBUG)

SESSION_COOKIE_SECURE = _env_bool('SESSION_COOKIE_SECURE', default=not DEBUG)
CSRF_COOKIE_SECURE = _env_bool('CSRF_COOKIE_SECURE', default=not DEBUG)
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'
SECURE_SSL_REDIRECT = _env_bool('SECURE_SSL_REDIRECT', default=False)
SECURE_HSTS_SECONDS = int(os.environ.get('SECURE_HSTS_SECONDS', '0'))
SECURE_HSTS_INCLUDE_SUBDOMAINS = _env_bool('SECURE_HSTS_INCLUDE_SUBDOMAINS', default=False)
SECURE_HSTS_PRELOAD = _env_bool('SECURE_HSTS_PRELOAD', default=False)

# ── File-based cache ───────────────────────────────────────────────
# /tmp/caregap_cache is world-writable and works for both local dev
# and HF Spaces (non-root uid=1000 cannot write to /app/cache).
_default_cache_dir = Path(os.environ.get('CARE_GAP_CACHE_DIR', '/tmp/caregap_cache'))
if _default_cache_dir.exists() and not _default_cache_dir.is_dir():
    _default_cache_dir = BASE_DIR / 'cache'
try:
    _default_cache_dir.mkdir(parents=True, exist_ok=True)
except FileExistsError:
    _default_cache_dir = BASE_DIR / 'cache'
    _default_cache_dir.mkdir(parents=True, exist_ok=True)

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': str(_default_cache_dir),
        'TIMEOUT': 300,
        'OPTIONS': {'MAX_ENTRIES': 200},
    }
}

# ── Synthea CSV data directory ─────────────────────────────────────
# Override by setting SYNTHEA_DATA_DIR env var, otherwise defaults
# to the relative path data/synthea_ca_seed43438_p30000/
SYNTHEA_DATA_DIR = os.environ.get(
    'SYNTHEA_DATA_DIR',
    str(BASE_DIR / 'data' / 'synthea_ca_seed43438_p30000'),
)

# ── HuggingFace Inference API ──────────────────────────────────────
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', '')

# ── Ollama / LLaMA settings (legacy — superseded by HF API) ───────
OLLAMA_BASE_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'phi3:latest')

# ── FAISS vector index path ────────────────────────────────────────
MEDGEMMA_URL = os.environ.get('MEDGEMMA_URL', '')
MEDGEMMA_MODEL = os.environ.get('MEDGEMMA_MODEL', 'google/medgemma-1.5-4b-it')
FAISS_INDEX_PATH = BASE_DIR / 'rag' / 'faiss_index'

if DEPLOYMENT_MODE not in {'internal', 'demo'}:
    raise RuntimeError("DEPLOYMENT_MODE must be 'internal' or 'demo'.")

if ENVIRONMENT == 'production' and SECRET_KEY == 'caregap-dev-secret-key-change-in-production':
    raise RuntimeError('Set SECRET_KEY in the environment for production deployments.')
