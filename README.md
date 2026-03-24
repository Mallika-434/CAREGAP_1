# CareGap Analytics — Patient Risk Management System
### AA-5960-02 Masters Research Project · Group 5 · Saint Louis University
### Asmi Basnet | Mallika Chand | Mandapalli Jagadeeshwari
### Instructor: Dr. Srikanth Mudigonda

---

## Project Overview

CareGap is a full-stack healthcare analytics dashboard built
for nurse case managers managing patients with hypertension
and Type 2 diabetes. It uses synthetic EHR data from Synthea
(California, 30,000 patients) to identify care gaps, calculate
risk scores, and surface actionable clinical insights.

---

## Architecture

```
caregap/
├── caregap/               # Django project config
│   ├── settings.py        # Set SYNTHEA_DATA_DIR here
│   ├── urls.py
│   └── wsgi.py
├── patients/              # Core patient app
│   ├── models.py          # Patient, Observation, Encounter,
│   │                      # Condition, Medication, Organization
│   ├── risk_engine.py     # Risk scoring: CRITICAL/WARNING/STABLE
│   ├── urgent_care_matcher.py  # Clinic matching logic
│   ├── views.py           # REST API endpoints
│   ├── serializers.py
│   ├── urls.py
│   └── management/commands/
│       ├── import_synthea.py   # CSV → SQLite importer
│       ├── mark_deceased.py    # Flags inactive patients
│       └── build_rag_index.py  # FAISS index builder
├── rag/
│   └── pipeline.py        # RAG logic (FAISS + HF Inference API)
├── templates/
│   └── dashboard.html     # Full SPA frontend
├── requirements.txt
└── manage.py
```

---

## Patient Cohorts

The system imports all 33,990 Synthea patients split into
4 cohorts:

| Cohort | Description | Count |
|--------|-------------|-------|
| chronic | Adults 18-110 with HTN or T2D | ~6,267 |
| at_risk | Adults 18-110 without chronic disease | ~16,776 |
| pediatric | Patients under 18 | ~6,957 |
| deceased | Patients with death date | ~3,990 |

---

## Risk Tier Logic

| Tier | Score | Action |
|------|-------|--------|
| CRITICAL | ≥ 60 | Immediate outreach required |
| WARNING | 30-59 | Schedule follow-up within 2 weeks |
| STABLE | < 30 | Routine monitoring |

---

## Care Gap Rules

| Care Gap | Rule |
|----------|------|
| HbA1c Overdue | No HbA1c test in > 365 days |
| BP Follow-up Missing | SBP ≥ 160 with no encounter in 30 days |
| Missing Medication | No active medication on record |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Mallika-434/CAREGAP_1.git
cd CAREGAP_1
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Synthea CSV files
Download the Synthea California dataset and place in:
```
data/synthea_ca_seed43438_p30000/
```

Files needed:
- patients.csv
- conditions.csv
- observations.csv
- encounters.csv
- medications.csv
- organizations.csv
- payers.csv
- payer_transitions.csv

### 5. Update settings.py
Edit `caregap/settings.py`:
```python
SYNTHEA_DATA_DIR = 'data/synthea_ca_seed43438_p30000'
```

### 6. Run migrations
```bash
python manage.py migrate
```

### 7. Import all 33,990 patients
```bash
python manage.py import_synthea --clear
```

> This will take 45–90 minutes for the full dataset.

### 8. Run the server
```bash
python manage.py runserver
```

Open: http://localhost:8000

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | /api/patients/search/ | Patient search with cohort filter |
| GET | /api/patients/stats/ | Population analytics |
| GET | /api/patients/triage/ | High risk triage list |
| GET | /api/patients/\<id\>/ | Full patient profile |
| GET | /api/patients/\<id\>/risk/ | Risk assessment |
| GET | /api/patients/\<id\>/urgent-care/ | Clinic recommendations |

---

## Dashboard Pages

1. **Analytics Explorer** — Population filtering & predictive insights
2. **Population Dashboard** — Population-level charts and metrics
3. **Patient Search** — Searchable directory of all 30,000 patients
4. **Action Required** — Emergency and urgent care triage queue

---

## Dataset

Synthea California synthetic dataset:
- Seed: 43438
- Total patients: 33,990
- Source: https://github.com/synthetichealth/synthea
- Note: Dataset not included in repo due to size (~5 GB compressed).
  Contact the team for access or generate using Synthea.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.14 + Django 6.0 + Django REST Framework |
| Frontend | Vanilla JS SPA + Chart.js |
| Database | SQLite (development) |
| Caching | Django file-based cache |
| RAG | FAISS + HF Inference API (Mistral-7B) |
| Predictions | scikit-learn + pandas |
| Deployment | Hugging Face Spaces + Docker |
