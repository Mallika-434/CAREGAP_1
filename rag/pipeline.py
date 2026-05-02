"""
RAG Pipeline — Habit & Lifestyle Suggestions
─────────────────────────────────────────────
Architecture:
  1. A curated knowledge base of clinical lifestyle guidelines
     (diabetes, hypertension, diet, exercise) is embedded into
     a FAISS vector store using sentence-transformers.
  2. At query time, the patient's risk profile is used to form
     a retrieval query.
  3. Relevant chunks are retrieved and injected as context into
     a prompt sent to Ollama (LLaMA3 or any local model).
  4. Ollama returns personalized habit suggestions.

Setup:
  python manage.py build_rag_index   ← run once to build FAISS index
"""

import os
import json
import requests
import logging
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Guardrails and prompt constants ───────────────────────────────────────────
GUARDRAIL_PROMPT = (
    "You are a clinical assistant for CareGap Analytics. "
    "Provide accurate, evidence-based clinical information only. "
    "Never provide diagnoses or replace professional medical judgment. "
    "Be concise and focused on care coordination."
)

COORDINATOR_PROMPT = (
    "You are assisting a care coordinator reviewing patient data. "
    "Be concise, actionable, and focused on care coordination tasks. "
    "Reference the patient data provided when answering."
)

OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions related to clinical care coordination, "
    "patient risk assessment, and population health management. "
    "Please ask a question related to your patients or clinical guidelines."
)

# ── Try to import vector store deps (optional at import time) ──────
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.debug(
        "faiss-cpu or sentence-transformers not installed — RAG index disabled. "
        "Install with: pip install faiss-cpu sentence-transformers"
    )


# ── Knowledge Base ─────────────────────────────────────────────────
# These chunks are embedded into the FAISS index.
# Expand with real clinical guidelines (ADA, ACC/AHA, etc.)
KNOWLEDGE_BASE = [
    # Diabetes management
    {
        "id": "dm_diet_1",
        "condition": "diabetes",
        "text": (
            "For patients with Type 2 diabetes, a low-glycemic diet is strongly recommended. "
            "Focus on non-starchy vegetables, whole grains, lean proteins, and healthy fats. "
            "Limit refined carbohydrates, sugary beverages, and processed foods. "
            "Aim for consistent meal timing to stabilize blood glucose levels."
        )
    },
    {
        "id": "dm_exercise_1",
        "condition": "diabetes",
        "text": (
            "Regular physical activity is a cornerstone of diabetes management. "
            "The ADA recommends at least 150 minutes per week of moderate-intensity aerobic exercise, "
            "such as brisk walking, swimming, or cycling. Resistance training 2-3 times per week "
            "improves insulin sensitivity. Avoid prolonged sitting — break up sedentary time every 30 minutes."
        )
    },
    {
        "id": "dm_monitoring_1",
        "condition": "diabetes",
        "text": (
            "Self-monitoring of blood glucose (SMBG) helps patients understand how food, activity, "
            "and stress affect their glucose levels. HbA1c should be checked at least twice a year "
            "for stable patients, and quarterly for those with poor control or recent therapy changes. "
            "Target HbA1c is below 7% for most non-pregnant adults with diabetes."
        )
    },
    {
        "id": "dm_sleep_1",
        "condition": "diabetes",
        "text": (
            "Poor sleep quality and sleep deprivation are linked to insulin resistance and higher HbA1c levels. "
            "Diabetic patients should aim for 7-9 hours of quality sleep per night. "
            "Screen for sleep apnea, which is more common in people with obesity and Type 2 diabetes."
        )
    },
    {
        "id": "dm_stress_1",
        "condition": "diabetes",
        "text": (
            "Chronic stress elevates cortisol, which raises blood glucose levels. "
            "Stress management techniques such as mindfulness, deep breathing, yoga, and cognitive "
            "behavioral therapy can help improve glycemic control. Social support from family and "
            "peer groups also positively impacts diabetes outcomes."
        )
    },
    # Hypertension management
    {
        "id": "htn_diet_1",
        "condition": "hypertension",
        "text": (
            "The DASH (Dietary Approaches to Stop Hypertension) diet is the gold standard for "
            "blood pressure management. It emphasizes fruits, vegetables, whole grains, low-fat dairy, "
            "and limits sodium to under 1,500–2,300 mg/day. Reducing processed food intake and "
            "reading nutrition labels for sodium content are practical first steps."
        )
    },
    {
        "id": "htn_exercise_1",
        "condition": "hypertension",
        "text": (
            "Aerobic exercise reduces systolic blood pressure by an average of 5–8 mmHg. "
            "Patients with hypertension should aim for 30 minutes of moderate aerobic activity "
            "most days of the week. Isometric resistance exercises (e.g., wall sits, handgrip) "
            "have also shown significant BP-lowering effects in recent studies."
        )
    },
    {
        "id": "htn_alcohol_1",
        "condition": "hypertension",
        "text": (
            "Alcohol consumption raises blood pressure and can interfere with antihypertensive medications. "
            "Patients should limit alcohol to no more than 1 drink per day for women and 2 per day for men. "
            "Eliminating alcohol entirely often produces measurable reductions in blood pressure within weeks."
        )
    },
    {
        "id": "htn_smoking_1",
        "condition": "hypertension",
        "text": (
            "Smoking causes acute spikes in blood pressure and damages arterial walls, compounding "
            "cardiovascular risk. Smoking cessation is one of the most impactful lifestyle changes "
            "for hypertensive patients. Nicotine replacement therapy, varenicline, and behavioral "
            "counseling are all evidence-based cessation strategies."
        )
    },
    {
        "id": "htn_weight_1",
        "condition": "hypertension",
        "text": (
            "Losing even 5–10% of body weight can significantly reduce blood pressure in overweight patients. "
            "A structured weight loss program combining dietary changes and increased physical activity "
            "is recommended. Even modest weight loss of 5 kg has been shown to lower systolic BP by 4–5 mmHg."
        )
    },
    # General preventive
    {
        "id": "prev_hydration_1",
        "condition": "preventive",
        "text": (
            "Adequate hydration supports cardiovascular function and helps the kidneys regulate blood pressure. "
            "Patients should aim for 6-8 glasses of water per day, adjusted for activity level and climate. "
            "Sugary drinks and excessive caffeine should be minimized."
        )
    },
    {
        "id": "prev_mental_1",
        "condition": "preventive",
        "text": (
            "Mental health and physical health are deeply intertwined. Anxiety and depression are "
            "associated with worse outcomes in both diabetes and hypertension. Encourage patients "
            "to seek mental health support, maintain social connections, and engage in activities "
            "that bring meaning and reduce stress."
        )
    },
    {
        "id": "prev_checkup_1",
        "condition": "preventive",
        "text": (
            "Preventive care visits allow early detection of risk factor progression. "
            "Annual wellness exams, lipid panels, kidney function tests (eGFR, urine albumin), "
            "and eye exams are important for patients at risk of metabolic disease. "
            "Staying current with vaccinations (flu, pneumococcal) also reduces hospitalization risk."
        )
    },
]


class RAGPipeline:
    """
    Manages embedding, retrieval, and LLM generation for
    personalized habit suggestions.
    """

    def __init__(self):
        self.index       = None
        self.chunks      = []
        self.model       = None
        self.index_path  = Path(settings.FAISS_INDEX_PATH)
        self._loaded     = False

    def _load_model(self):
        if self.model is None:
            if not FAISS_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers and faiss-cpu are required. "
                    "Install: pip install sentence-transformers faiss-cpu"
                )
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _ollama_url(self) -> str:
        return os.environ.get('OLLAMA_URL', 'http://localhost:11434')

    def _ollama_model(self) -> str:
        return os.environ.get('OLLAMA_MODEL', 'phi3:latest')

    def _deployment_mode(self) -> str:
        return getattr(settings, 'DEPLOYMENT_MODE', 'internal')

    def _local_llm_enabled(self) -> bool:
        return self._deployment_mode() == 'internal'

    def _medgemma_url(self) -> str:
        if not self._local_llm_enabled():
            return ''
        return os.environ.get('MEDGEMMA_URL') or getattr(settings, 'MEDGEMMA_URL', '')

    def _medgemma_model(self) -> str:
        return os.environ.get('MEDGEMMA_MODEL') or getattr(
            settings,
            'MEDGEMMA_MODEL',
            'google/medgemma-1.5-4b-it',
        )

    def _gemini_key(self) -> str | None:
        key = os.environ.get('GEMINI_API_KEY') or getattr(settings, 'GEMINI_API_KEY', None)
        return key or None

    def _call_medgemma(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 400,
        temperature: float = 0.2,
        timeout: int = 180,
    ) -> str | None:
        medgemma_url = self._medgemma_url()
        if not medgemma_url:
            return None

        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})

        payload = {
            'model': self._medgemma_model(),
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }

        response = requests.post(medgemma_url, json=payload, timeout=timeout)
        if response.status_code != 200:
            logger.warning("MedGemma returned status %s", response.status_code)
            return None

        data = response.json()
        try:
            choice = data['choices'][0]['message']['content']
            if isinstance(choice, list):
                parts = [part.get('text', '') for part in choice if isinstance(part, dict)]
                return ''.join(parts).strip() or None
            return str(choice).strip() or None
        except (KeyError, IndexError, TypeError):
            text = data.get('generated_text') or data.get('response')
            if text:
                return str(text).strip() or None
            logger.warning("MedGemma response did not contain generated text")
            return None

    def _call_ollama(self, prompt: str, *, system: str | None = None, timeout: int = 180) -> str | None:
        if not self._local_llm_enabled():
            return None

        payload = {
            'model': self._ollama_model(),
            'prompt': prompt,
            'stream': False,
        }
        payload['options'] = {'temperature': 0, 'top_p': 0.9}
        if system:
            payload['system'] = system

        response = requests.post(
            f'{self._ollama_url()}/api/generate',
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 200:
            logger.warning("Ollama returned status %s", response.status_code)
            return None
        return response.json().get('response', '').strip() or None

    def _call_gemini(self, prompt: str, *, max_output_tokens: int, temperature: float) -> str | None:
        gemini_key = self._gemini_key()
        if not gemini_key:
            return None

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash-lite:generateContent?key={gemini_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
            },
        }
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code != 200:
            logger.warning("Gemini returned status %s", response.status_code)
            return None

        data = response.json()
        try:
            return data['candidates'][0]['content']['parts'][0]['text'].strip() or None
        except (KeyError, IndexError, TypeError):
            logger.warning("Gemini response did not contain generated text")
            return None

    def build_index(self):
        """Build FAISS index from KNOWLEDGE_BASE. Run once via management command."""
        self._load_model()
        texts = [chunk['text'] for chunk in KNOWLEDGE_BASE]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)

        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs
        index.add(embeddings)

        # Save
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path / 'knowledge.index'))
        with open(self.index_path / 'chunks.json', 'w') as f:
            json.dump(KNOWLEDGE_BASE, f, indent=2)

        self.index  = index
        self.chunks = KNOWLEDGE_BASE
        self._loaded = True
        logger.info(f"FAISS index built with {len(KNOWLEDGE_BASE)} chunks.")

    def _ensure_loaded(self):
        if self._loaded:
            return
        index_file = self.index_path / 'knowledge.index'
        chunks_file = self.index_path / 'chunks.json'

        if index_file.exists() and chunks_file.exists():
            self._load_model()
            self.index  = faiss.read_index(str(index_file))
            with open(chunks_file) as f:
                self.chunks = json.load(f)
            self._loaded = True
        else:
            raise RuntimeError(
                "RAG index not built. Run: python manage.py build_rag_index"
            )

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Embed query and retrieve top-k most relevant chunks."""
        self._ensure_loaded()
        q_embed = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(q_embed)
        scores, indices = self.index.search(q_embed, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = dict(self.chunks[idx])
                chunk['score'] = float(score)
                results.append(chunk)
        return results

    def generate_suggestions(self, patient_profile: dict) -> dict:
        """
        Full RAG pipeline:
          1. Build retrieval query from patient profile
          2. Retrieve relevant knowledge chunks
          3. Construct prompt with context
          4. Call Ollama → get personalized suggestions
        """
        # Step 1: Build retrieval query
        conditions = []
        if patient_profile.get('has_diabetes'):
            conditions.append('diabetes')
        if patient_profile.get('has_hypertension'):
            conditions.append('hypertension')
        if not conditions:
            conditions.append('preventive')

        query = (
            f"Lifestyle and habit recommendations for a patient with "
            f"{', '.join(conditions)}. "
            f"Age: {patient_profile.get('age', 'unknown')}. "
            f"HbA1c: {patient_profile.get('hba1c_value') or 'N/A'}%. "
            f"Systolic BP: {patient_profile.get('latest_sbp') or 'N/A'} mmHg."
        )

        # Step 2: Retrieve context
        try:
            chunks = self.retrieve(query, top_k=4)
            context = "\n\n".join([c['text'] for c in chunks])
        except RuntimeError as e:
            # Index not built yet — use raw knowledge base as fallback
            context = "\n\n".join([c['text'] for c in KNOWLEDGE_BASE[:4]])
            chunks  = []

        # Step 3: Build prompt
        prompt = _build_prompt(patient_profile, context)

        # Step 4: Try MedGemma first for healthcare-aware local generation
        try:
            text = self._call_medgemma(prompt, max_tokens=400, temperature=0.2)
            if text:
                return {
                    'query': query,
                    'context_used': [c.get('id') for c in chunks],
                    'suggestions': text,
                    'model': f'medgemma-{self._medgemma_model()}',
                }
        except Exception as e:
            logger.warning("MedGemma RAG call failed: %s", e)

        # Secondary local fallback: Ollama
        try:
            text = self._call_ollama(prompt)
            if text:
                return {
                    'query': query,
                    'context_used': [c.get('id') for c in chunks],
                    'suggestions': text,
                    'model': f'ollama-{self._ollama_model()}',
                }
        except Exception as e:
            logger.warning("Ollama RAG call failed: %s", e)

        # Cloud fallback: Gemini only when explicitly configured
        try:
            text = self._call_gemini(prompt, max_output_tokens=400, temperature=0.3)
            if text:
                return {
                    'query': query,
                    'context_used': [c.get('id') for c in chunks],
                    'suggestions': text,
                    'model': 'gemini-fallback',
                }
        except Exception as e:
            logger.warning("Gemini RAG call failed: %s", e)

        return {
            'query':        query,
            'context_used': [c.get('id') for c in chunks],
            'suggestions':  _rule_based_suggestions(patient_profile),
            'model':        'rule-based-fallback',
        }

    def explain_patient_result(self, explanation_type: str, patient_data: dict) -> dict:
        """Generate a plain English explanation of a patient result using Gemini.
        explanation_type: 'chronic_prediction', 'onset_risk', 'bmi_assessment'
        """
        if explanation_type == 'chronic_prediction':
            risk_level = 'HIGH' if (patient_data.get('ensemble_pct') or 0) >= 60 else 'MODERATE' if (patient_data.get('ensemble_pct') or 0) >= 35 else 'LOW'

            prompt = f"""You are a clinical assistant. Write exactly 3 bullet points explaining this patient's results to a care coordinator. Each bullet is one sentence. Start each bullet with a dash. Do not add any other text. Do not interpret or change the risk level — use exactly what is given.

Facts:
- Patient: {patient_data.get('name')}, {patient_data.get('age')} years old
- Conditions: {patient_data.get('conditions', 'None')}
- HbA1c: {patient_data.get('hba1c') or 'Not tested'}
- Systolic BP: {patient_data.get('sbp') or 'Not recorded'}
- Risk level: {risk_level} ({patient_data.get('ensemble_pct')}% chance of deterioration in 6 months)
- Recommendation: {patient_data.get('recommendation')}

Write 3 bullets:
- Bullet 1: What the HbA1c and BP numbers mean for this patient
- Bullet 2: What the {risk_level} risk level means (use the exact percentage {patient_data.get('ensemble_pct')}%)
- Bullet 3: What the care coordinator should do next based on the recommendation"""

        elif explanation_type == 'chat_prediction':
            risk_level = 'HIGH' if (patient_data.get('ensemble_pct') or 0) >= 60 else 'MODERATE' if (patient_data.get('ensemble_pct') or 0) >= 35 else 'LOW'
            prompt = f"""RESPOND WITH EXACTLY 3 SHORT SENTENCES. No more. Each sentence under 15 words.

Patient {patient_data.get('name')}: HbA1c={patient_data.get('hba1c') or 'not tested'}, BP={patient_data.get('sbp') or 'not recorded'}, Risk={risk_level} {patient_data.get('ensemble_pct')}%, Action={patient_data.get('recommendation')}

Sentence 1: HbA1c and BP status.
Sentence 2: What {patient_data.get('ensemble_pct')}% {risk_level} risk means.
Sentence 3: What to do next.

OUTPUT FORMAT: Three sentences only. Nothing else."""

        elif explanation_type == 'onset_risk':
            prompt = f"""You are a clinical assistant. Write exactly 3 bullet points. Each bullet starts with a dash on a new line. One sentence per bullet. No extra text before or after.

Facts:
- Patient: {patient_data.get('name')}, {patient_data.get('age')} years old
- HTN risk: {patient_data.get('htn_ensemble')}% (Lasso: {patient_data.get('htn_lasso')}%, RF: {patient_data.get('htn_rf')}%, GB: {patient_data.get('htn_gb')}%)
- T2D risk: {patient_data.get('t2d_ensemble')}% (Lasso: {patient_data.get('t2d_lasso')}%, RF: {patient_data.get('t2d_rf')}%, GB: {patient_data.get('t2d_gb')}%)
- Current SBP: {patient_data.get('sbp')} mmHg, BMI: {patient_data.get('bmi')}
- Days since last visit: {patient_data.get('days_since_encounter')}

Write 3 bullets:
- Bullet 1: What the HTN risk score means using the exact percentage
- Bullet 2: What the T2D risk score means using the exact percentage
- Bullet 3: What the care coordinator should do next"""

        elif explanation_type == 'bmi_assessment':
            prompt = f"""You are a clinical assistant. Write exactly 3 bullet points. Each bullet starts with a dash on a new line. One sentence per bullet. No extra text before or after.

Facts:
- Patient: {patient_data.get('name')}, {patient_data.get('age')} years old, {patient_data.get('gender')}
- BMI: {patient_data.get('bmi')}
- Category: {patient_data.get('category')}
- Recommendation: {patient_data.get('recommendation')}

Write 3 bullets:
- Bullet 1: What the BMI value means for a child this age
- Bullet 2: What the {patient_data.get('category')} category means for their health
- Bullet 3: What the care coordinator should do next"""

        else:
            return {"explanation": "No explanation available.", "source": "none"}

        # Try MedGemma first
        system_prompt = (
            'You are a clinical assistant. Never change or reinterpret numerical '
            'values or risk levels given to you. Use only the facts provided. '
            'Be concise.'
        )
        try:
            text = self._call_medgemma(
                prompt,
                system=system_prompt,
                max_tokens=300,
                temperature=0.1,
            )
            if text:
                return {
                    "explanation": text,
                    "source": f"medgemma-{self._medgemma_model()}",
                }
        except Exception as e:
            logger.warning("MedGemma explain call failed: %s", e)

        # Secondary local fallback: Ollama
        try:
            text = self._call_ollama(
                prompt,
                system=system_prompt,
            )
            if text:
                logger.info("Ollama explain response: %s", text[:200])
                ensemble_pct = patient_data.get('ensemble_pct', '')
                if text and ensemble_pct and explanation_type == 'chronic_prediction':
                    pct_int = str(int(float(ensemble_pct))) if ensemble_pct else ''
                    pct_float = str(float(ensemble_pct)) if ensemble_pct else ''
                    if pct_int not in text and pct_float not in text:
                        logger.warning("Hallucination detected: expected %s%% in explanation", pct_int)
                        return {
                            "explanation": self._rule_based_explanation(explanation_type, patient_data),
                            "source": "rule_based_validated"
                        }
                if text and explanation_type == 'chat_prediction':
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
                    sentences = [s.strip() for s in sentences if s.strip()][:3]
                    text = ' '.join(sentences)
                return {
                    "explanation": text,
                    "source": f"ollama-{self._ollama_model()}",
                }
        except Exception as e:
            logger.warning("Ollama explain call failed: %s", e)

        # Cloud fallback: Gemini only when configured
        try:
            text = self._call_gemini(prompt, max_output_tokens=300, temperature=0.4)
            if text:
                return {"explanation": text, "source": "gemini"}
        except Exception as e:
            logger.warning("Gemini explain call failed: %s", e)

        return {"explanation": self._rule_based_explanation(explanation_type, patient_data), "source": "rule_based"}

    def _rule_based_explanation(self, explanation_type: str, patient_data: dict) -> str:
        """Fallback explanation when Gemini is unavailable."""
        if explanation_type == 'chronic_prediction':
            pct = patient_data.get('ensemble_pct', 0)
            if pct >= 60:
                return "This patient has a high chance of getting worse in the next 6 months. Immediate follow-up is recommended."
            elif pct >= 35:
                return "This patient has a moderate chance of deterioration. Schedule a follow-up visit soon."
            else:
                return "This patient appears stable. Continue the current care plan and monitor regularly."
        elif explanation_type == 'onset_risk':
            htn = patient_data.get('htn_ensemble', 0)
            t2d = patient_data.get('t2d_ensemble', 0)
            return f"This patient has a {htn}% risk profile for hypertension and {t2d}% for diabetes based on their current health indicators."
        elif explanation_type == 'bmi_assessment':
            return f"This child's BMI of {patient_data.get('bmi')} is classified as {patient_data.get('category')}. {patient_data.get('recommendation')}"
        return "No explanation available."

    # ── Unified LLM caller ────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Unified caller: MedGemma → Ollama → Gemini → raises RuntimeError."""
        text = self._call_medgemma(prompt, max_tokens=400, temperature=0.2)
        if text:
            return text
        text = self._call_ollama(prompt)
        if text:
            return text
        text = self._call_gemini(prompt, max_output_tokens=400, temperature=0.3)
        if text:
            return text
        raise RuntimeError("No LLM backend available")

    def is_out_of_scope(self, question: str) -> bool:
        """Return True if the question is clearly non-clinical."""
        _oob = {'recipe', 'weather', 'sports', 'movie', 'music', 'politics',
                'stock', 'crypto', 'bitcoin', 'game', 'joke', 'poem', 'song'}
        return any(kw in question.lower() for kw in _oob)

    def _mock_llm_response(self, question: str) -> str:
        """Rule-based fallback when no LLM backend is reachable."""
        q = question.lower()
        if 'hba1c' in q:
            return ("HbA1c measures average blood glucose over 2-3 months. "
                    "Normal < 5.7%; prediabetes 5.7-6.4%; diabetes >= 6.5%.")
        if any(k in q for k in ('blood pressure', 'sbp', 'hypertension')):
            return ("SBP above 130 mmHg is elevated. Stage 1 HTN: 130-139 mmHg. "
                    "Stage 2 HTN: 140+ mmHg. Hypertensive crisis: 180+ mmHg.")
        if any(k in q for k in ('emergency', 'threshold', 'tier')):
            return ("Emergency: HbA1c >= 9% OR SBP >= 160 mmHg — immediate outreach. "
                    "High: HbA1c 8-8.9% OR SBP 140-159 — urgent care within 24-48 h. "
                    "Moderate: active chronic dx, controlled labs — follow-up within 30 days.")
        if any(k in q for k in ('risk', 'predict', 'score', 'ensemble')):
            return ("Ensemble risk is a composite score from Lasso, Random Forest, and "
                    "GradientBoosting models. Scores >= 60% indicate high risk of "
                    "clinical deterioration in the next 6 months.")
        return ("I can help with questions about patient risk scores, HbA1c, blood pressure, "
                "care gaps, and population health metrics. Please try rephrasing your question.")

    # ── Three methods required by rag/views.py ───────────────────────────────

    def explain_prediction(self, patient_profile: dict, prediction_result: dict) -> str:
        """Plain English explanation of any prediction result. Routes by cohort."""
        is_pediatric = patient_profile.get('age', 99) < 18
        cohort = prediction_result.get('cohort', '')

        if is_pediatric or cohort == 'pediatric':
            prompt = f"""{GUARDRAIL_PROMPT}
Explain this pediatric BMI assessment to a non-technical health coordinator in plain English.

PATIENT: {patient_profile.get('name')} ({patient_profile.get('age')}y {patient_profile.get('gender')})
BMI: {prediction_result.get('bmi', 'N/A')}
CDC PERCENTILE: {prediction_result.get('percentile', 'N/A')}th
CATEGORY: {prediction_result.get('risk_tier') or prediction_result.get('category', 'N/A')}
CARE GAPS: {', '.join(prediction_result.get('care_gaps', [])) or 'None detected'}
RECOMMENDATION: {prediction_result.get('recommendation')}

Keep it to 2-3 sentences. Focus on what the percentile means for this child and what the coordinator should do next."""
        else:
            prob = prediction_result.get('progression_probability') or prediction_result.get('onset_probability')
            prob_str = f"{round(prob * 100)}%" if prob is not None else 'N/A'
            prompt = f"""{GUARDRAIL_PROMPT}
Explain this clinical prediction to a non-technical health coordinator in plain English.

PATIENT: {patient_profile.get('name')} ({patient_profile.get('age')}y {patient_profile.get('gender')})
ENSEMBLE RISK: {prob_str}
MODEL SCORES: {prediction_result.get('model_scores', {})}
RECOMMENDATION: {prediction_result.get('recommendation', 'N/A')}

Keep it to 2-3 sentences. Focus on what the risk score means and what action the coordinator should take."""
        try:
            return self._call_llm(prompt)
        except Exception:
            prob = prediction_result.get('progression_probability') or prediction_result.get('onset_probability')
            prob_str = f"{round(prob * 100)}%" if prob is not None else 'N/A'
            return (
                f"{patient_profile.get('name')} has an ensemble risk score of {prob_str}. "
                f"Recommendation: {prediction_result.get('recommendation', 'Continue monitoring.')} "
                f"Please review the patient's latest labs and follow up accordingly."
            )

    def generate_coordinator_answer(self, patient_profile: dict, question: str, history: list = None) -> str:
        """Open-ended Q&A for care coordinators asking about a specific patient."""
        if self.is_out_of_scope(question):
            return OUT_OF_SCOPE_MESSAGE

        history_text = ""
        if history:
            history_text = "\nPREVIOUS CONVERSATION:\n"
            for msg in history[-6:]:
                role = "User" if msg.get("isUser") else "Assistant"
                history_text += f"{role}: {msg.get('text', '')}\n"

        prompt = (
            f"{GUARDRAIL_PROMPT}\n{COORDINATOR_PROMPT}\n"
            f"PATIENT DATA: {patient_profile}\n"
            f"{history_text}\n"
            f"COORDINATOR QUESTION: {question}\n\n"
            f"Provide a specific, clinical answer based on the patient context above."
        )
        try:
            return self._call_llm(prompt)
        except Exception:
            return self._mock_llm_response(question)

    def generate_analytics_answer(self, question: str, history: list = None) -> str:
        """Answers questions about population analytics, tiers, and metrics."""
        if self.is_out_of_scope(question):
            return OUT_OF_SCOPE_MESSAGE

        system_context = """
CAREGAP ANALYTICS CONTEXT:
- ENSEMBLE RISK: Composite prediction (0-100%) from Lasso, Random Forest, and XGBoost models.
- EMERGENCY: HbA1c >= 9% or SBP >= 160 mmHg. Immediate outreach required.
- HIGH: HbA1c 8-8.9% or SBP 140-159 mmHg. Urgent care routing within 24-48 hours.
- MODERATE: Active Diabetes/HTN with controlled labs. Schedule follow-up within 30 days.
- PREVENTIVE: Stable patients at future risk. Lifestyle reinforcement.
- NORMAL: No high-risk labs or conditions. Routine monitoring.
- THRESHOLDS: Diabetes (HbA1c >= 6.5%), Stage 2 HTN (SBP >= 140 mmHg).
"""
        history_text = ""
        if history:
            history_text = "\nPREVIOUS CONVERSATION:\n"
            for msg in history[-6:]:
                role = "User" if msg.get("isUser") else "Assistant"
                history_text += f"{role}: {msg.get('text', '')}\n"

        prompt = (
            f"{GUARDRAIL_PROMPT}\n{system_context}\n"
            f"{history_text}\n"
            f"Analytics question: {question}\n\n"
            f"Answer as a Clinical Data Analyst. Be specific about thresholds and tiers."
        )
        try:
            return self._call_llm(prompt)
        except Exception:
            return self._mock_llm_response(question)


def _build_prompt(profile: dict, context: str) -> str:
    """Construct the RAG prompt injected into Ollama."""
    name       = profile.get('name', 'the patient')
    age        = profile.get('age', 'unknown')
    gender     = profile.get('gender', 'unknown')
    conditions = []
    if profile.get('has_diabetes'):
        conditions.append('Type 2 Diabetes')
    if profile.get('has_hypertension'):
        conditions.append('Hypertension')
    hba1c = profile.get('hba1c_value')
    sbp   = profile.get('latest_sbp')

    condition_str = ', '.join(conditions) if conditions else 'No confirmed chronic conditions'

    return f"""You are a nurse case management assistant providing personalized, evidence-based \
lifestyle and habit recommendations to support patient health.

PATIENT PROFILE
───────────────
Name:       {name}
Age:        {age}
Gender:     {gender}
Conditions: {condition_str}
HbA1c:      {f"{hba1c}%" if hba1c else "Not recently tested"}
Systolic BP:{f"{sbp} mmHg" if sbp else "Not recently recorded"}

CLINICAL GUIDELINES CONTEXT (retrieved)
────────────────────────────────────────
{context}

TASK
────
Give exactly 4 lifestyle recommendations for this patient.
Format: one line per recommendation, starting with a bold title like **Title:** followed by one sentence only.
No paragraphs. No extra explanation. No follow-up questions. No repetition.
Example format:
**Exercise:** Walk 30 minutes daily to lower blood pressure.
**Diet:** Follow DASH diet focusing on fruits, vegetables and low sodium foods.
**Sleep:** Aim for 7-9 hours nightly to support cardiovascular health.
**Stress:** Practice deep breathing for 10 minutes daily to reduce cortisol levels.
"""


def _call_huggingface(prompt: str, token: str) -> str:
    """Send prompt to HuggingFace Inference API (Mistral-7B) and return generated text."""
    url     = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "temperature":    0.7,
            "top_p":          0.9,
            "return_full_text": False,
        },
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and data:
        return data[0].get('generated_text', '').strip()
    return str(data).strip()


def _rule_based_suggestions(profile: dict) -> str:
    """
    Deterministic, evidence-based suggestions when no LLM is available.
    Returns a formatted string of 4–5 bullet recommendations.
    """
    has_dm  = profile.get('has_diabetes',     False)
    has_htn = profile.get('has_hypertension', False)
    hba1c   = profile.get('hba1c_value')
    sbp     = profile.get('latest_sbp')
    age     = profile.get('age', 0) or 0

    bullets: list[str] = []

    # ── Glycemic control ──────────────────────────────────────────
    if has_dm:
        if hba1c and hba1c >= 9.0:
            bullets.append(
                "**Urgent Glycemic Review** — Your HbA1c is critically elevated "
                f"({hba1c}%). Contact your care team this week to review medications "
                "and consider a diabetes care specialist referral."
            )
        elif hba1c and hba1c >= 6.5:
            bullets.append(
                "**Low-Glycemic Diet** — Prioritize non-starchy vegetables, whole grains, "
                "and lean proteins. Limit refined carbohydrates, sugary beverages, and "
                "processed foods. Consistent meal timing helps stabilize blood glucose."
            )
        bullets.append(
            "**Regular Physical Activity** — Aim for at least 150 minutes per week of "
            "moderate aerobic exercise (brisk walking, cycling, swimming). "
            "Break up sitting time every 30 minutes to improve insulin sensitivity."
        )

    # ── Blood pressure control ────────────────────────────────────
    if has_htn:
        if sbp and sbp >= 160:
            bullets.append(
                "**Immediate BP Follow-Up Needed** — Your systolic blood pressure is "
                f"{sbp} mmHg, which is Stage 2 hypertension. Please schedule a clinic "
                "visit within the next week and reduce sodium intake immediately."
            )
        bullets.append(
            "**DASH Diet** — Follow the Dietary Approaches to Stop Hypertension diet: "
            "emphasize fruits, vegetables, whole grains, and low-fat dairy. "
            "Limit sodium to under 1,500–2,300 mg/day by reducing processed foods."
        )

    # ── Sleep ─────────────────────────────────────────────────────
    bullets.append(
        "**Quality Sleep** — Aim for 7–9 hours of restful sleep each night. "
        "Poor sleep raises cortisol and blood glucose. "
        + ("Screen for sleep apnea, which is more common with diabetes and obesity." if has_dm
           else "Consistent sleep schedules support cardiovascular health.")
    )

    # ── Stress management ─────────────────────────────────────────
    bullets.append(
        "**Stress Management** — Chronic stress raises both blood glucose and blood pressure. "
        "Practice mindfulness, deep breathing, or gentle yoga for 10–15 minutes daily. "
        "Social support from family and peer groups also improves chronic disease outcomes."
    )

    # ── Age-specific monitoring ───────────────────────────────────
    if age >= 65:
        bullets.append(
            "**Fall & Balance Safety** — At your age, maintaining muscle strength through "
            "light resistance training (chair squats, resistance bands) reduces fall risk. "
            "Review all medications with your provider for interactions that affect balance."
        )

    return "\n\n".join(f"• {b}" for b in bullets)


# Singleton instance
rag_pipeline = RAGPipeline()
