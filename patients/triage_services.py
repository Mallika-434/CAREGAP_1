from datetime import date as current_date
from datetime import timedelta

from django.core.cache import cache
from django.utils import timezone

from .models import Condition, Encounter, Observation, Patient


def _age(birthdate):
    if not birthdate:
        return None
    today = current_date.today()
    bd = birthdate if hasattr(birthdate, "year") else birthdate
    return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))


def get_triage_payload():
    cached = cache.get("triage_list")
    if cached is not None:
        return cached

    now = timezone.now()
    one_year_ago = now - timedelta(days=365)
    ninety_ago = now - timedelta(days=90)
    thirty_ago = now - timedelta(days=30)

    sbp_map = {}
    hba1c_map = {}

    for pid, val in (
        Observation.objects.filter(
            patient__is_deceased=False, code=Observation.LOINC_SBP
        )
        .order_by("-date")
        .values_list("patient__patient_id", "value")
        .iterator(chunk_size=5000)
    ):
        if pid not in sbp_map:
            try:
                sbp_map[pid] = float(val)
            except (ValueError, TypeError):
                pass

    for pid, val in (
        Observation.objects.filter(
            patient__is_deceased=False, code=Observation.LOINC_HBA1C
        )
        .order_by("-date")
        .values_list("patient__patient_id", "value")
        .iterator(chunk_size=5000)
    ):
        if pid not in hba1c_map:
            try:
                hba1c_map[pid] = float(val)
            except (ValueError, TypeError):
                pass

    encounter_map = {}
    for pid, start in (
        Encounter.objects.filter(patient__is_deceased=False)
        .order_by("-start")
        .values_list("patient__patient_id", "start")
        .iterator(chunk_size=5000)
    ):
        if pid not in encounter_map:
            encounter_map[pid] = start

    critical_pids = set()
    for pid, sbp in sbp_map.items():
        if sbp >= 160:
            critical_pids.add(pid)
    for pid, hba1c in hba1c_map.items():
        if hba1c >= 10.0:
            critical_pids.add(pid)
    for pid, hba1c in hba1c_map.items():
        if hba1c >= 9.0:
            last_enc = encounter_map.get(pid)
            if last_enc is None or (
                hasattr(last_enc, "date") and last_enc < thirty_ago.date()
            ) or (
                hasattr(last_enc, "year")
                and not hasattr(last_enc, "date")
                and last_enc < thirty_ago
            ):
                critical_pids.add(pid)

    chronic_pids = set(
        Patient.objects.filter(cohort="chronic", is_deceased=False).values_list(
            "patient_id", flat=True
        )
    )

    urgent_pids = set()
    for pid, sbp in sbp_map.items():
        if 140 <= sbp <= 159 and pid in chronic_pids:
            urgent_pids.add(pid)
    for pid, hba1c in hba1c_map.items():
        if 7.0 <= hba1c <= 9.9 and pid in chronic_pids:
            urgent_pids.add(pid)

    active_dx_pids = set(
        Condition.objects.filter(
            patient__is_deceased=False,
            patient__cohort="chronic",
            stop__isnull=True,
            code__in=Condition.HYPERTENSION_CODES + Condition.DIABETES_CODES,
        )
        .values_list("patient__patient_id", flat=True)
        .distinct()
    )
    recent_hba1c_pids = set(
        Observation.objects.filter(
            patient__patient_id__in=active_dx_pids,
            code=Observation.LOINC_HBA1C,
            date__gte=one_year_ago,
        ).values_list("patient__patient_id", flat=True)
    )
    urgent_pids |= active_dx_pids - recent_hba1c_pids

    recently_seen_pids = set(
        Encounter.objects.filter(
            patient__cohort="chronic",
            patient__is_deceased=False,
            start__gte=ninety_ago,
        ).values_list("patient__patient_id", flat=True)
    )
    for pid, sbp in sbp_map.items():
        if sbp >= 130 and pid in chronic_pids and pid not in recently_seen_pids:
            urgent_pids.add(pid)

    urgent_pids -= critical_pids

    def _fetch_patients(pids, tier, limit=50):
        rows = (
            Patient.objects.filter(patient_id__in=list(pids)[:limit])
            .values("patient_id", "first", "last", "birthdate", "city")
        )
        result = []
        for patient_row in rows:
            pid = patient_row["patient_id"]
            result.append(
                {
                    "patient_id": pid,
                    "name": f"{patient_row['first']} {patient_row['last']}",
                    "age": _age(patient_row["birthdate"]),
                    "city": patient_row["city"],
                    "tier": tier,
                    "hba1c": hba1c_map.get(pid),
                    "sbp": sbp_map.get(pid),
                }
            )
        return result

    def _fetch_urgent(pids, limit=50):
        rows = (
            Patient.objects.filter(patient_id__in=list(pids))
            .values("patient_id", "first", "last", "birthdate", "city")
        )
        result = []
        for patient_row in rows:
            pid = patient_row["patient_id"]
            result.append(
                {
                    "patient_id": pid,
                    "name": f"{patient_row['first']} {patient_row['last']}",
                    "age": _age(patient_row["birthdate"]),
                    "city": patient_row["city"],
                    "tier": "WARNING",
                    "hba1c": hba1c_map.get(pid),
                    "sbp": sbp_map.get(pid),
                    "_last_enc": encounter_map.get(pid),
                }
            )
        result.sort(
            key=lambda item: (
                -(item["sbp"] or 0),
                -(item["hba1c"] or 0),
                (item["_last_enc"] or "1900-01-01"),
            )
        )
        for row in result:
            row.pop("_last_enc", None)
        return result[:limit]

    payload = {
        "emergency_patients": _fetch_patients(critical_pids, "CRITICAL", limit=10),
        "urgent_patients": _fetch_urgent(urgent_pids, limit=50),
    }
    cache.set("triage_list", payload, 300)
    return payload


def get_resource_forecast_payload():
    from datetime import datetime

    from .forecaster import forecast_resources

    triage = cache.get("triage_list")
    if triage:
        risk_breakdown = {
            "emergency": len(triage.get("emergency_patients", [])),
            "high": len(triage.get("urgent_patients", [])),
            "moderate": len(triage.get("warning_patients", [])),
            "elevated": len(triage.get("stable_patients", [])),
        }
    else:
        chronic_count = int(Patient.objects.filter(cohort="chronic").count())
        risk_breakdown = {
            "emergency": int(chronic_count * 0.05),
            "high": int(chronic_count * 0.15),
            "moderate": int(chronic_count * 0.30),
            "elevated": int(chronic_count * 0.50),
        }

    forecast = forecast_resources(risk_breakdown)
    forecast["generated_at"] = datetime.now().isoformat()
    return forecast
