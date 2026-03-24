"""
CareGap Analytics — Django Settings
"""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    'SECRET_KEY',
    'caregap-dev-secret-key-change-in-production',
)
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
ALLOWED_HOSTS = ['*']

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
        'NAME': BASE_DIR / 'db.sqlite3',
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

CORS_ALLOW_ALL_ORIGINS = True

# ── File-based cache (makes triage 5-min cache actually work) ──────
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': str(BASE_DIR / 'cache'),
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
OLLAMA_BASE_URL = 'http://localhost:11434'
OLLAMA_MODEL    = 'llama3'

# ── FAISS vector index path ────────────────────────────────────────
FAISS_INDEX_PATH = BASE_DIR / 'rag' / 'faiss_index'
