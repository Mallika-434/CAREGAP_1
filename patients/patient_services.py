import time

from django.core.cache import cache
from django.db.models import Q

from .models import Patient
from .risk_engine import assess_risk
from .serializers import PatientDetailSerializer, PatientListSerializer
from .urgent_care_matcher import find_urgent_cares


def search_patients(query="", cohort="", limit=50, offset=0):
    query = (query or "").strip()
    cohort = (cohort or "").strip()
    limit = min(limit, 200)
    offset = max(offset, 0)

    if cohort in ("chronic", "at_risk", "pediatric", "deceased"):
        qs = Patient.objects.filter(cohort=cohort)
    else:
        qs = Patient.objects.filter(is_deceased=False)

    if query:
        for term in query.split():
            qs = qs.filter(
                Q(first__icontains=term)
                | Q(last__icontains=term)
                | Q(city__icontains=term)
                | Q(patient_id__icontains=term)
            )

    total = qs.count()
    serializer = PatientListSerializer(qs[offset : offset + limit], many=True)
    return {
        "query": query,
        "total": total,
        "offset": offset,
        "count": len(serializer.data),
        "results": serializer.data,
    }


def get_patient_detail_payload(patient_id):
    cache_key = f"patient_{patient_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    t0 = time.time()
    patient = Patient.objects.prefetch_related(
        "observations", "encounters", "conditions", "medications"
    ).get(patient_id=patient_id)
    print(f"[profile] patient+prefetch fetch: {time.time()-t0:.3f}s")

    t1 = time.time()
    data = PatientDetailSerializer(patient).data
    print(f"[profile] serialization: {time.time()-t1:.3f}s")
    print(f"[profile] total (uncached): {time.time()-t0:.3f}s")

    cache.set(cache_key, data, 300)
    return data


def get_patient_risk_payload(patient_id):
    cache_key = f"patient_risk_{patient_id}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    t0 = time.time()
    patient = Patient.objects.prefetch_related("observations", "conditions").get(
        patient_id=patient_id
    )
    print(f"[risk] patient+prefetch fetch: {time.time()-t0:.3f}s")

    t1 = time.time()
    result = assess_risk(patient, patient.observations.all(), patient.conditions.all())
    print(f"[risk] assess_risk (in-memory): {time.time()-t1:.3f}s")
    print(f"[risk] total (uncached): {time.time()-t0:.3f}s")

    payload = {
        "patient_id": patient.patient_id,
        "patient_name": patient.full_name(),
        "tier": result.tier,
        "score": result.score,
        "reasons": result.reasons,
        "hba1c_days_gap": result.hba1c_days_gap,
        "hba1c_value": result.hba1c_value,
        "latest_sbp": result.latest_sbp,
        "has_diabetes": result.has_diabetes,
        "has_hypertension": result.has_hypertension,
        "recommended_action": result.recommended_action,
        "followup_urgency_days": result.followup_urgency_days,
    }
    cache.set(cache_key, payload, 600)
    return payload


def get_patient_urgent_care_payload(patient_id):
    patient = Patient.objects.get(patient_id=patient_id)
    return {
        "patient_id": patient.patient_id,
        "patient_name": patient.full_name(),
        "patient_city": patient.city,
        "patient_insurance": patient.insurance,
        "facilities": find_urgent_cares(patient, max_results=5),
    }


def get_patient_prediction_payload(patient_id):
    from patients.ml_models import (
        decompose_risk,
        extract_features,
        predict_ensemble_score,
        predict_multi_hba1c_trajectory,
        predict_multi_sbp_trajectory,
    )

    patient = Patient.objects.get(patient_id=patient_id)

    if patient.cohort == "pediatric":
        return {
            "error": "not_applicable",
            "message": "Prediction models are not applicable for pediatric patients.",
            "cohort": "pediatric",
        }

    if patient.cohort == "at_risk":
        return {
            "error": "not_available",
            "message": "Disease onset prediction for at-risk patients is coming soon.",
            "cohort": "at_risk",
        }

    observations = list(patient.observations.all())
    conditions = list(patient.conditions.all())
    medications = list(patient.medications.all())
    encounters = list(patient.encounters.all())

    features_dict, features_arr = extract_features(
        patient, observations, conditions, medications, encounters
    )

    ensemble = predict_ensemble_score(features_arr, features_dict)
    model_available = ensemble["model_available"]
    prob = ensemble["probability"]

    if not model_available:
        result = assess_risk(patient, observations, conditions)
        prob = min(result.score / 100.0, 0.99)
        ensemble["probability"] = round(prob, 3)

    decomposed = decompose_risk(features_dict)
    hba1c_proj = predict_multi_hba1c_trajectory(observations)
    sbp_proj = predict_multi_sbp_trajectory(observations)

    hba1c_trend = hba1c_proj["trend"]
    sbp_trend = sbp_proj["trend"]

    trends = {hba1c_trend, sbp_trend}
    if "worsening" in trends:
        risk_trajectory = "worsening"
    elif "improving" in trends and "stable" not in trends:
        risk_trajectory = "improving"
    else:
        risk_trajectory = "stable"

    if not model_available:
        confidence = "low"
    elif prob >= 0.75 or prob <= 0.25:
        confidence = "high"
    elif prob >= 0.60 or prob <= 0.40:
        confidence = "medium"
    else:
        confidence = "low"

    if prob >= 0.70 or risk_trajectory == "worsening":
        recommendation = "Immediate intervention recommended"
    elif prob >= 0.40:
        recommendation = "Schedule follow-up within 30 days"
    else:
        recommendation = "Continue current care plan"

    return {
        "patient_id": patient_id,
        "patient_name": patient.full_name(),
        "cohort": patient.cohort,
        "age": patient.age,
        "gender": patient.gender,
        "progression_probability": prob,
        "model_scores": ensemble["model_scores"],
        "range_min": ensemble["range_min"],
        "range_max": ensemble["range_max"],
        "model_available": model_available,
        "sugar_risk": round(decomposed["sugar"], 3),
        "bp_risk": round(decomposed["bp"], 3),
        "sugar_forecast": {
            "lasso": hba1c_proj["lasso"],
            "rf": hba1c_proj["rf"],
            "xgb": hba1c_proj["xgb"],
        },
        "bp_forecast": {
            "lasso": sbp_proj["lasso"],
            "rf": sbp_proj["rf"],
            "xgb": sbp_proj["xgb"],
        },
        "predicted_hba1c_6mo": hba1c_proj["lasso"],
        "predicted_sbp_6mo": sbp_proj["lasso"],
        "hba1c_trend": hba1c_trend,
        "sbp_trend": sbp_trend,
        "risk_trajectory": risk_trajectory,
        "trend_direction": risk_trajectory,
        "confidence": confidence,
        "recommendation": recommendation,
        "features": features_dict,
    }
