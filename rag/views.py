"""
RAG API Views
─────────────
POST /api/rag/suggest/   → generate habit suggestions for a patient
GET  /api/rag/status/    → check if Ollama is reachable and index is built
"""

import requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from patients.models import Patient, Observation, Condition
from patients.risk_engine import assess_risk
from .pipeline import rag_pipeline


@api_view(['POST'])
def generate_suggestions(request):
    """
    Generate personalized lifestyle habit suggestions for a patient.

    POST body:
    {
        "patient_id": "uuid-string"
    }
    """
    patient_id = request.data.get('patient_id', '').strip()
    if not patient_id:
        return Response({'error': 'patient_id is required.'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        patient = Patient.objects.get(patient_id=patient_id)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found.'}, status=status.HTTP_404_NOT_FOUND)

    # Run risk engine to populate profile
    observations = Observation.objects.filter(patient=patient)
    conditions   = Condition.objects.filter(patient=patient)
    risk_result  = assess_risk(patient, observations, conditions)

    patient_profile = {
        'name':            patient.full_name(),
        'age':             patient.age,
        'gender':          patient.gender,
        'city':            patient.city,
        'has_diabetes':    risk_result.has_diabetes,
        'has_hypertension': risk_result.has_hypertension,
        'hba1c_value':     risk_result.hba1c_value,
        'latest_sbp':      risk_result.latest_sbp,
        'risk_tier':       risk_result.tier,
        'risk_score':      risk_result.score,
    }

    result = rag_pipeline.generate_suggestions(patient_profile)

    return Response({
        'patient_id':   patient_id,
        'patient_name': patient.full_name(),
        'risk_tier':    risk_result.tier,
        'risk_score':   risk_result.score,
        **result,
    })


@api_view(['GET'])
def rag_status(request):
    """
    Health check:
    - Is Ollama reachable?
    - Is the FAISS index built?
    """
    # Check Ollama
    ollama_ok = False
    ollama_models = []
    try:
        r = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            ollama_ok = True
            ollama_models = [m['name'] for m in r.json().get('models', [])]
    except Exception:
        pass

    # Check FAISS index
    index_path = settings.FAISS_INDEX_PATH
    index_built = (
        (index_path / 'knowledge.index').exists() and
        (index_path / 'chunks.json').exists()
    )

    return Response({
        'ollama_reachable': ollama_ok,
        'ollama_url':       settings.OLLAMA_BASE_URL,
        'configured_model': settings.OLLAMA_MODEL,
        'available_models': ollama_models,
        'faiss_index_built': index_built,
        'index_path':       str(index_path),
        'status': 'ready' if (ollama_ok and index_built) else 'not_ready',
        'instructions': (
            None if (ollama_ok and index_built) else
            "Run: python manage.py build_rag_index — then ensure Ollama is running with your model."
        )
    })
