# CareGap Analytics — Patient Risk Management System
### AA-5960-02 Masters Research Project · Group 5 · Saint Louis University

---

## Architecture

```
caregap/
├── caregap/               # Django project config
│   ├── settings.py        # ← Set SYNTHEA_DATA_DIR and OLLAMA_MODEL here
│   ├── urls.py
│   └── wsgi.py
├── patients/              # Core patient app
│   ├── models.py          # Patient, Observation, Encounter, Condition, UrgentCare
│   ├── risk_engine.py     # ← Risk scoring: EMERGENCY / HIGH / MODERATE / PREVENTIVE / NORMAL
│   ├── urgent_care_matcher.py  # ← Geometric distance + Insurance matching
│   ├── views.py           # REST API endpoints (Stats, Search, Profile, Triage)
│   ├── serializers.py
│   ├── urls.py
│   └── management/commands/
│       ├── import_synthea.py   # ← CSV → SQLite importer
│       ├── mark_deceased.py    # ← Flags patients with >5yr inactivity
│       └── build_rag_index.py
├── rag/
│   └── pipeline.py        # ← RAG logic (Currently disabled per requirements)
├── templates/
│   └── dashboard.html     # ← Full SPA frontend (Charts, Action Required, Trends)
├── requirements.txt
└── manage.py
```

---

## Setup (Step-by-Step)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Point to your Synthea CSV files
Edit `caregap/settings.py`:
```python
SYNTHEA_DATA_DIR = '/path/to/your/synthea/output/csv'
# Files needed: patients.csv, observations.csv, encounters.csv,
#               conditions.csv, payers.csv, payer_transitions.csv
```

### 3. Set up the database
```bash
python manage.py migrate
```

### 4. Import Synthea data
```bash
python manage.py import_synthea
# Optional: --data-dir /custom/path   --clear (to wipe and re-import)
```

### 5. Build the RAG FAISS index
```bash
python manage.py build_rag_index
# Downloads 'all-MiniLM-L6-v2' model on first run (~80 MB)
```

### 6. Set up Ollama with LLaMA
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3          # or: llama3.2, mistral, phi3
ollama serve                # Keep this running in a separate terminal
```

Change model in `settings.py` if using a different one:
```python
OLLAMA_MODEL = 'llama3'     # match your pulled model name exactly
```

### 7. Run the server
```bash
python manage.py runserver
```

Open: **http://localhost:8000**

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/patients/search/?q=<name>` | Fuzzy patient search |
| GET | `/api/patients/stats/` | Demographic and Population Risk analytics |
| GET | `/api/patients/triage/` | Generates EMERGENCY and HIGH risk triage lists |
| GET | `/api/patients/<id>/` | Full longitudinal patient profile and history |
| GET | `/api/patients/<id>/risk/` | Real-time risk tier computation + reasons |
| GET | `/api/patients/<id>/urgent-care/` | Nearby urgent cares geometrically matched |

---

## Risk Tier Logic (`risk_engine.py`)

| Tier | Score | Trigger Conditions | Action |
|------|-------|--------------------|--------|
| **EMERGENCY**| ≥ 80 | SBP ≥ 160 mmHg, OR HbA1c ≥ 9.0%, OR high composite score | Dispatch to ER automatically immediately |
| **HIGH** | 60–79 | Elevated risk score | Match to Urgent Care (within 24-48 hours) |
| **MODERATE** | 30–59 | HbA1c approaching overdue, OR SBP ≥ 140, OR borderline HbA1c | Schedule Follow-up (within 30 days) |
| **PREVENTIVE** | 10–29 | At-risk demographics, mild vitals concern | Preventive guidance (within 90 days) |
| **NORMAL** | < 10 | No current care gaps detected | Routine Monitoring |

---

## Workflow Per Patient

```
Search patient by name
        ↓
Risk Assessment (automatic)
        ↓
    ┌───┴────────────────────────────┐
   HIGH                         MODERATE
    ↓                               ↓
Find urgent cares              Schedule follow-up
matched by insurance           visit recommendation
+ city proximity
        ↓ (any tier)
RAG Habit Suggestions
via LLaMA (Ollama)
FAISS retrieves relevant
clinical guidelines → 
personalized recommendations
```

---

## Extending the Knowledge Base

Add to `rag/pipeline.py` → `KNOWLEDGE_BASE` list:
```python
{
    "id": "dm_new_guideline",
    "condition": "diabetes",   # or "hypertension" or "preventive"
    "text": "Your clinical guideline text here..."
},
```
Then rebuild: `python manage.py build_rag_index`
