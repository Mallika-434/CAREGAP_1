"""
Patient API Views
─────────────────
GET  /api/patients/search/?q=<name>          → fuzzy patient search
GET  /api/patients/<patient_id>/             → full patient profile
GET  /api/patients/<patient_id>/risk/        → risk assessment result
GET  /api/patients/<patient_id>/urgent-care/ → nearby urgent cares (HIGH risk)
"""

import logging
from datetime import timedelta

from django.db.models import Count, Exists, FloatField, Max, OuterRef, Sum, IntegerField
from django.db.models.functions import Cast
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

logger = logging.getLogger(__name__)

from .analytics_services import get_analytics_payload
from .models import Patient, Observation, Encounter, Condition, Medication
from .patient_services import (
    get_patient_detail_payload,
    get_patient_prediction_payload,
    get_patient_risk_payload,
    get_patient_urgent_care_payload,
    search_patients,
)
from .stats_services import get_dashboard_stats_basic_payload, get_dashboard_stats_payload
from .triage_services import get_resource_forecast_payload, get_triage_payload


def _to_json_safe(obj):
    """Recursively convert numpy/pandas/datetime objects to JSON-safe Python types."""
    import numpy as np
    from datetime import date, datetime
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


# ── 1. Patient Search ─────────────────────────────────────────────
@api_view(['GET'])
def patient_search(request):
    """
    Search patients by name or city with pagination.
    ?q=<term>       filter by first/last/city (blank = all)
    ?cohort=<val>   filter by cohort (chronic|at_risk|pediatric|deceased)
                    default: excludes deceased (shows all alive cohorts)
    ?limit=<n>      page size, default 50, max 200
    ?offset=<n>     skip first N rows, default 0
    Returns: { total, count, offset, results }
    """
    query = request.GET.get('q', '').strip()
    cohort = request.GET.get('cohort', '').strip()
    try:
        limit = min(int(request.GET.get('limit', 50)), 200)
    except (ValueError, TypeError):
        limit = 50
    try:
        offset = max(int(request.GET.get('offset', 0)), 0)
    except (ValueError, TypeError):
        offset = 0

    return Response(
        search_patients(query=query, cohort=cohort, limit=limit, offset=offset)
    )


# ── 2. Patient Profile ────────────────────────────────────────────
@api_view(['GET'])
def patient_detail(request, patient_id):
    """Full patient profile with all related data."""
    try:
        return Response(get_patient_detail_payload(patient_id))
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found.'}, status=status.HTTP_404_NOT_FOUND)


# ── 3. Risk Assessment ────────────────────────────────────────────
@api_view(['GET'])
def patient_risk(request, patient_id):
    """
    Run the risk engine for a patient and return structured result.
    Used to drive the dashboard risk card.
    Cached 10 min per patient; prefetches observations+conditions in
    one shot to avoid separate lazy queries inside assess_risk.
    """
    try:
        return Response(get_patient_risk_payload(patient_id))
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found.'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
def patient_urgent_cares(request, patient_id):
    """
    Returns nearby urgent care facilities matched to patient's
    city and insurance type. Intended for HIGH-risk patients only
    but can be called for any patient.
    """
    try:
        return Response(get_patient_urgent_care_payload(patient_id))
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found.'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
def dashboard_stats_basic(request):
    """
    Returns only cohort counts + condition rates.
    All indexed COUNT queries - responds quickly.
    Called first by the dashboard to populate the stat cards
    while the full /stats/ endpoint computes in the background.
    """
    return Response(get_dashboard_stats_basic_payload())


@api_view(['GET'])
def dashboard_stats(request):
    """
    Full population analytics used by all dashboard charts.
    Cached 10 minutes after first compute (~5???15 s on SQLite cold).

    HbA1c and BP distributions use a raw GROUP BY + MAX(date) query
    so SQLite resolves the latest value per patient in a single pass ???
    no correlated subquery, no Python-side blocking.
    """
    return Response(get_dashboard_stats_payload())


@api_view(['GET'])
def analytics(request):
    """
    Flexible population analytics with user-defined filters.
    Params: cohort, gender, age_min, age_max, condition
    Returns: count, hba1c_dist, bp_dist, age_dist, top_conditions
    Cached 10 minutes per unique filter combination.
    All raw SQL uses patient_id (UUID string), not id (integer),
    because all FK relations use to_field='patient_id'.
    """
    return Response(
        get_analytics_payload(
            cohort=request.GET.get('cohort', ''),
            gender=request.GET.get('gender', ''),
            age_min=request.GET.get('age_min', ''),
            age_max=request.GET.get('age_max', ''),
            condition=request.GET.get('condition', ''),
        )
    )


@api_view(['GET'])
def patient_predict(request, patient_id):
    """
    Ensemble ML 6-month risk forecast for a single patient.

    Runs 3 models (Lasso LR, Random Forest, GradientBoosting) and returns:
      - ensemble probability (average of all 3)
      - individual model scores
      - sugar vs BP risk decomposition
      - multi-model HbA1c and SBP trajectory projections

    Falls back to rule-based score when no trained models exist.
    """
    try:
        payload = get_patient_prediction_payload(patient_id)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found'}, status=404)

    return Response(payload)

@api_view(['GET'])
def patient_onset_risk(request, patient_id):
    """GET /api/patients/<id>/onset-risk/
    Returns HTN and T2D onset risk scores for at-risk patients.
    """
    from .ml_models import predict_onset_risk

    try:
        patient = Patient.objects.get(patient_id=patient_id)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found'}, status=404)

    if patient.cohort != 'at_risk':
        return Response({
            'error': 'not_applicable',
            'message': 'Onset risk models are only available for at-risk patients.',
            'cohort': patient.cohort,
        })

    observations = list(patient.observations.all())
    encounters   = list(patient.encounters.all())
    medications  = list(patient.medications.all())
    conditions   = list(patient.conditions.all())

    result = predict_onset_risk(observations, encounters, medications, conditions, patient)

    return Response({
        'patient_id': str(patient.patient_id),
        'name': f"{patient.first} {patient.last}",
        'cohort': patient.cohort,
        'onset_risk': result,
    })


@api_view(['GET'])
def patient_bmi_assessment(request, patient_id):
    """GET /api/patients/<id>/bmi-assessment/
    Returns age-appropriate BMI assessment for pediatric patients.
    """
    from datetime import date

    try:
        patient = Patient.objects.get(patient_id=patient_id)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found'}, status=404)

    if patient.cohort != 'pediatric':
        return Response({
            'error': 'not_applicable',
            'message': 'BMI assessment is only available for pediatric patients.',
            'cohort': patient.cohort,
        })

    # Get latest BMI observation
    bmi_obs = patient.observations.filter(
        code='39156-5'
    ).order_by('-date').first()

    if not bmi_obs:
        return Response({
            'available': False,
            'message': 'No BMI measurement on record for this patient.',
        })

    try:
        bmi = float(bmi_obs.value)
    except (TypeError, ValueError):
        return Response({
            'available': False,
            'message': 'BMI value could not be read.',
        })

    age = patient.age or 0
    gender = patient.gender or 'F'

    # Simplified CDC-based age/gender cutoffs
    # Boys thresholds are 0.5 higher than girls
    gender_adj = 0.5 if gender == 'M' else 0.0

    if age <= 5:
        overweight  = 18.0 + gender_adj
        obese       = 18.5 + gender_adj
        underweight = 14.0
    elif age <= 11:
        overweight  = 19.0 + gender_adj
        obese       = 21.0 + gender_adj
        underweight = 14.5
    else:
        overweight  = 22.0 + gender_adj
        obese       = 25.0 + gender_adj
        underweight = 16.0

    if bmi < underweight:
        category  = 'Underweight'
        color     = 'blue'
        recommend = 'Refer to pediatric nutritionist. Monitor weight gain.'
    elif bmi < overweight:
        category  = 'Healthy Weight'
        color     = 'green'
        recommend = 'Continue routine monitoring. Encourage physical activity.'
    elif bmi < obese:
        category  = 'Overweight'
        color     = 'amber'
        recommend = 'Recommend dietary review and increased physical activity.'
    else:
        category  = 'Obese'
        color     = 'red'
        recommend = 'Refer to pediatric specialist. Discuss lifestyle intervention.'

    return Response({
        'available':  True,
        'patient_id': str(patient.patient_id),
        'name':       f"{patient.first} {patient.last}",
        'age':        age,
        'gender':     gender,
        'bmi':        round(bmi, 1),
        'category':   category,
        'color':      color,
        'recommend':  recommend,
        'thresholds': {
            'underweight': underweight,
            'overweight':  overweight,
            'obese':       obese,
        }
    })


@api_view(['POST'])
@csrf_exempt
def explain_result(request):
    """POST /api/rag/explain/
    Generates a plain English explanation of a patient result using Gemini.
    Body: { explanation_type, patient_data }
    """
    from rag.pipeline import rag_pipeline

    explanation_type = request.data.get('explanation_type')
    patient_data     = request.data.get('patient_data', {})

    if not explanation_type:
        return Response({'error': 'explanation_type required'}, status=400)

    result = rag_pipeline.explain_patient_result(explanation_type, patient_data)
    return Response(result)


# ── 8. Triage Dashboard (Emergency / Urgent Care) ─────────────────
# Source: ACC/AHA 2023 Hypertension Guidelines
# Source: ADA Standards of Medical Care 2024
# Source: JNC 8 Guidelines
@api_view(['GET'])
def triage_list(request):
    """
    Fast triage using pure DB queries ??? no assess_risk() loops.

    EMERGENCY (immediate outreach):
      - SBP >= 160 mmHg          [ACC/AHA 2023: Stage 2 Hypertension]
      - OR HbA1c >= 10%          [ADA 2024: very poorly controlled diabetes]
      - OR HbA1c >= 9% with no encounter in last 30 days
                                 [ADA 2024: poorly controlled, no recent follow-up]

    URGENT (same-day or next-day visit):
      - SBP 140???159 mmHg         [JNC 8: Stage 1 requiring medication]
      - OR HbA1c 7.0???9.9%        [ADA 2024: above goal for most adults]
      - OR HbA1c overdue (no test in 365 days) with active HTN/diabetes dx
                                 [ADA 2024: annual HbA1c for stable patients]
      - OR SBP >= 130 with no encounter in last 90 days
                                 [ACC/AHA 2023: elevated BP requiring follow-up]

    Result cached 5 minutes.
    """
    return Response(get_triage_payload())


@api_view(['GET'])
def resource_forecast(request):
    import os
    import pickle
    from .forecaster import forecast_resources

    cache_path = os.path.join('patients', 'data', 'forecast_cache.pkl')
    triage_path = os.path.join('patients', 'data', 'triage_cache.pkl')

    # Try pickle cache first (fastest)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            if 'high_risk_volume' not in data and 'risk_breakdown' in data:
                rb = data['risk_breakdown']
                data['high_risk_volume'] = (
                    rb.get('emergency', 0) + rb.get('high', 0) +
                    rb.get('moderate', 0) + rb.get('elevated', 0)
                )
            return Response(data)
        except Exception as e:
            logger.error("Failed to load forecast cache: %s", e)

    # Fallback: compute from triage pickle if available
    if os.path.exists(triage_path):
        try:
            with open(triage_path, 'rb') as f:
                triage_data = pickle.load(f)
            risk_breakdown = triage_data.get('risk_breakdown', {})
            result = forecast_resources(risk_breakdown)
            result['generated_at'] = timezone.now().isoformat()
            return Response(result)
        except Exception as e:
            logger.error("Failed to compute forecast from triage cache: %s", e)

    # Final fallback: compute on demand from Django ORM
    try:
        payload = get_resource_forecast_payload()
        return Response(payload)
    except Exception as e:
        logger.error("Failed to compute forecast on demand: %s", e)
        return Response({
            'error': 'Forecast unavailable.',
            'generated_at': timezone.now().isoformat(),
            'resources': {}
        }, status=503)
