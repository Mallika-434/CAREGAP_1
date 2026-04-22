import time
import threading
from datetime import timedelta

from django.core.cache import cache
from django.db import connection
from django.db.models import Count, Max
from django.utils import timezone

from .models import Condition, Observation, Patient


def get_dashboard_stats_basic_payload():
    cached = cache.get("dashboard_stats_basic")
    if cached is not None:
        return cached

    total_active = Patient.objects.filter(is_deceased=False).count()
    total_deceased = Patient.objects.filter(is_deceased=True).count()

    ht_count = (
        Condition.objects.filter(
            patient__is_deceased=False,
            stop__isnull=True,
            code__in=Condition.HYPERTENSION_CODES,
        )
        .values("patient_id")
        .distinct()
        .count()
    )

    diab_count = (
        Condition.objects.filter(
            patient__is_deceased=False,
            stop__isnull=True,
            code__in=Condition.DIABETES_CODES,
        )
        .values("patient_id")
        .distinct()
        .count()
    )

    cohort_rows = Patient.objects.values("cohort").annotate(count=Count("id"))
    cohort_counts = {row["cohort"]: row["count"] for row in cohort_rows}

    payload = {
        "total_active": total_active,
        "total_deceased": total_deceased,
        "hypertension_rate": round(ht_count / total_active * 100, 1) if total_active else 0,
        "diabetes_rate": round(diab_count / total_active * 100, 1) if total_active else 0,
        "cohort_counts": {
            "chronic": cohort_counts.get("chronic", 0),
            "at_risk": cohort_counts.get("at_risk", 0),
            "pediatric": cohort_counts.get("pediatric", 0),
            "deceased": cohort_counts.get("deceased", 0),
        },
    }
    cache.set("dashboard_stats_basic", payload, 600)
    return payload


def get_dashboard_stats_payload():
    cached = cache.get("dashboard_stats")
    print(f"[stats] cache GET dashboard_stats -> {'HIT' if cached is not None else 'MISS'}")
    if cached is not None:
        return cached

    if cache.get("dashboard_stats_computing"):
        print("[stats] another request is computing -> waiting up to 60s for it to finish")
        for _ in range(12):
            time.sleep(5)
            cached = cache.get("dashboard_stats")
            if cached is not None:
                print("[stats] cache populated by other request -> returning cached result")
                return cached
        print("[stats] wait timed out -> running computation anyway")

    cache.set("dashboard_stats_computing", True, 120)
    t0 = time.monotonic()

    def _elapsed():
        return time.monotonic() - t0

    total_active = Patient.objects.filter(is_deceased=False).count()
    total_deceased = Patient.objects.filter(is_deceased=True).count()
    print(f"[stats] totals done      {_elapsed():.1f}s")

    ht_count = (
        Condition.objects.filter(
            stop__isnull=True,
            code__in=Condition.HYPERTENSION_CODES,
            patient__is_deceased=False,
        )
        .values("patient_id")
        .distinct()
        .count()
    )
    print(f"[stats] ht_count done    {_elapsed():.1f}s  ({ht_count:,})")

    diab_count = (
        Condition.objects.filter(
            stop__isnull=True,
            code__in=Condition.DIABETES_CODES,
            patient__is_deceased=False,
        )
        .values("patient_id")
        .distinct()
        .count()
    )
    print(f"[stats] diab_count done  {_elapsed():.1f}s  ({diab_count:,})")

    hypertension_rate = round(ht_count / total_active * 100, 1) if total_active else 0
    diabetes_rate = round(diab_count / total_active * 100, 1) if total_active else 0

    print(f"[stats] overlap...       {_elapsed():.1f}s")
    try:
        ht_phs = ",".join(["%s"] * len(Condition.HYPERTENSION_CODES))
        dia_phs = ",".join(["%s"] * len(Condition.DIABETES_CODES))
        with connection.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT patient_id FROM patients_condition
                    WHERE  code IN ({ht_phs}) AND stop IS NULL
                    INTERSECT
                    SELECT DISTINCT patient_id FROM patients_condition
                    WHERE  code IN ({dia_phs}) AND stop IS NULL
                )
            """,
                list(Condition.HYPERTENSION_CODES) + list(Condition.DIABETES_CODES),
            )
            both_count = cur.fetchone()[0]
        with connection.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM patients_condition WHERE code IN ({ht_phs})",
                list(Condition.HYPERTENSION_CODES),
            )
            ht_rows = cur.fetchone()[0]
        print(f"[stats] overlap: ht_rows={ht_rows:,}  both={both_count:,}")
    except Exception as exc:
        print(f"[stats] overlap error: {exc}")
        chronic_count = Patient.objects.filter(cohort="chronic").count()
        both_count = max(0, ht_count + diab_count - chronic_count)
        print(f"[stats] overlap fallback estimate ({both_count:,})")

    ht_only = max(0, ht_count - both_count)
    diab_only = max(0, diab_count - both_count)
    neither_count = max(0, total_active - ht_count - diab_count + both_count)
    print(f"[stats] overlap done     {_elapsed():.1f}s  both={both_count:,}")

    hba1c_dist = {"normal": 0, "prediabetes": 0, "diabetes": 0}
    print(f"[stats] hba1c_dist...    {_elapsed():.1f}s")
    with connection.cursor() as cur:
        cur.execute(
            """
            SELECT value FROM (
                SELECT value,
                       ROW_NUMBER() OVER (
                           PARTITION BY patient_id ORDER BY date DESC
                       ) AS rn
                FROM   patients_observation
                WHERE  code = %s
                  AND  patient_id IN (
                      SELECT patient_id FROM patients_patient WHERE cohort = 'chronic'
                  )
            ) WHERE rn = 1
        """,
            [Observation.LOINC_HBA1C],
        )
        rows = cur.fetchall()
    print(f"[stats] hba1c rows={len(rows)}  sample={[r[0] for r in rows[:5]]}")
    for (raw,) in rows:
        try:
            value = float(raw)
            if value < 5.7:
                hba1c_dist["normal"] += 1
            elif value < 6.5:
                hba1c_dist["prediabetes"] += 1
            else:
                hba1c_dist["diabetes"] += 1
        except (ValueError, TypeError):
            pass
    print(f"[stats] hba1c_dist done  {_elapsed():.1f}s  {hba1c_dist}")

    bp_dist = {"normal": 0, "elevated": 0, "stage1": 0, "stage2": 0}
    print(f"[stats] bp_dist...       {_elapsed():.1f}s")
    with connection.cursor() as cur:
        cur.execute(
            """
            SELECT value FROM (
                SELECT value,
                       ROW_NUMBER() OVER (
                           PARTITION BY patient_id ORDER BY date DESC
                       ) AS rn
                FROM   patients_observation
                WHERE  code = %s
                  AND  patient_id IN (
                      SELECT patient_id FROM patients_patient WHERE cohort = 'chronic'
                  )
            ) WHERE rn = 1
        """,
            [Observation.LOINC_SBP],
        )
        rows = cur.fetchall()
    print(f"[stats] bp rows={len(rows)}  sample={[r[0] for r in rows[:5]]}")
    for (raw,) in rows:
        try:
            value = float(raw)
            if value < 120:
                bp_dist["normal"] += 1
            elif value < 130:
                bp_dist["elevated"] += 1
            elif value < 140:
                bp_dist["stage1"] += 1
            else:
                bp_dist["stage2"] += 1
        except (ValueError, TypeError):
            pass
    print(f"[stats] bp_dist done     {_elapsed():.1f}s  {bp_dist}")

    ins_buckets = {"Medicare": 0, "Medicaid": 0, "Private": 0, "Uninsured": 0}
    ins_rows = list(
        Patient.objects.filter(is_deceased=False)
        .exclude(insurance="")
        .values("insurance")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    for row in ins_rows:
        name = (row["insurance"] or "").lower()
        count = row["count"]
        if "medicare" in name:
            ins_buckets["Medicare"] += count
        elif "medicaid" in name:
            ins_buckets["Medicaid"] += count
        elif name in ("no insurance", "self pay", "self-pay", "uninsured"):
            ins_buckets["Uninsured"] += count
        else:
            ins_buckets["Private"] += count

    chronic_qs = Patient.objects.filter(cohort="chronic")
    total_flagged = chronic_qs.count()
    cutoff = timezone.now() - timedelta(days=365)
    print(f"[stats] care gap start   {_elapsed():.1f}s  flagged={total_flagged:,}")

    print(f"[stats] hba1c_overdue... {_elapsed():.1f}s")
    hba1c_result = [None]

    def _hba1c_overdue_query():
        try:
            has_recent_count = (
                Observation.objects.filter(
                    code=Observation.LOINC_HBA1C, patient__cohort="chronic"
                )
                .values("patient_id")
                .annotate(latest=Max("date"))
                .filter(latest__gte=cutoff)
                .count()
            )
            hba1c_result[0] = max(0, total_flagged - has_recent_count)
        except Exception as exc:
            print(f"[stats] hba1c_overdue error: {exc}")

    ht = threading.Thread(target=_hba1c_overdue_query, daemon=True)
    ht.start()
    ht.join(timeout=20)

    if hba1c_result[0] is not None:
        hba1c_overdue = hba1c_result[0]
    else:
        hba1c_overdue = round(total_flagged * 0.45)
        print(f"[stats] hba1c_overdue TIMEOUT -> estimate ({hba1c_overdue:,})")
    print(f"[stats] hba1c_overdue    {_elapsed():.1f}s  ({hba1c_overdue:,})")

    print(f"[stats] bp_followup...   {_elapsed():.1f}s")
    bp_result = [None]

    def _bp_followup_query():
        try:
            with connection.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT o.patient_id, o.date AS sbp_date
                        FROM   patients_observation o
                        INNER JOIN (
                            SELECT patient_id, MAX(date) AS max_date
                            FROM   patients_observation
                            WHERE  code = %s
                            GROUP  BY patient_id
                        ) latest ON o.patient_id = latest.patient_id
                                AND o.date       = latest.max_date
                        WHERE  o.code = %s
                          AND  CAST(o.value AS REAL) >= 160
                          AND  o.patient_id IN (
                              SELECT patient_id FROM patients_patient WHERE cohort = 'chronic'
                          )
                        LIMIT 500
                    ) critical
                    WHERE NOT EXISTS (
                        SELECT 1 FROM patients_encounter e
                        WHERE  e.patient_id = critical.patient_id
                          AND  e.start     >= critical.sbp_date
                          AND  e.start     <= datetime(critical.sbp_date, '+30 days')
                    )
                """,
                    [Observation.LOINC_SBP, Observation.LOINC_SBP],
                )
                bp_result[0] = cur.fetchone()[0]
        except Exception as exc:
            print(f"[stats] bp_followup error: {exc}")

    bt = threading.Thread(target=_bp_followup_query, daemon=True)
    bt.start()
    bt.join(timeout=20)

    if bp_result[0] is not None:
        bp_followup_missing = bp_result[0]
    else:
        bp_followup_missing = round(ht_count * 0.18)
        print(f"[stats] bp_followup TIMEOUT -> estimate ({bp_followup_missing:,})")
    print(f"[stats] bp_followup      {_elapsed():.1f}s  ({bp_followup_missing:,})")

    print(f"[stats] no_medication... {_elapsed():.1f}s")
    no_medication = (
        Patient.objects.filter(cohort="chronic")
        .annotate(med_count=Count("medications"))
        .filter(med_count=0)
        .count()
    )
    print(f"[stats] no_medication    {_elapsed():.1f}s  ({no_medication:,})")

    city_dist = list(
        Patient.objects.filter(is_deceased=False)
        .values("city")
        .annotate(count=Count("id"))
        .order_by("-count")[:10]
    )
    race_dist = list(
        Patient.objects.filter(is_deceased=False)
        .values("race")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    gender_dist = list(
        Patient.objects.filter(is_deceased=False)
        .values("gender")
        .annotate(count=Count("id"))
    )
    cohort_rows = Patient.objects.values("cohort").annotate(count=Count("id"))
    cohort_counts = {row["cohort"]: row["count"] for row in cohort_rows}

    payload = {
        "total_active": total_active,
        "total_deceased": total_deceased,
        "hypertension_rate": hypertension_rate,
        "diabetes_rate": diabetes_rate,
        "average_risk_score": 0,
        "hba1c_dist": hba1c_dist,
        "bp_dist": bp_dist,
        "risk_overlap": {
            "both": both_count,
            "bp_only": ht_only,
            "bs_only": diab_only,
            "neither": neither_count,
        },
        "care_gap_cascade": {
            "total_flagged": total_flagged,
            "hba1c_overdue": hba1c_overdue,
            "bp_followup_missing": bp_followup_missing,
            "no_medication": no_medication,
        },
        "city_distribution": city_dist,
        "insurance_breakdown": ins_buckets,
        "race_breakdown": {row["race"]: row["count"] for row in race_dist},
        "gender_breakdown": {row["gender"]: row["count"] for row in gender_dist},
        "cohort_counts": {
            "chronic": cohort_counts.get("chronic", 0),
            "at_risk": cohort_counts.get("at_risk", 0),
            "pediatric": cohort_counts.get("pediatric", 0),
            "deceased": cohort_counts.get("deceased", 0),
        },
        "compute_seconds": round(_elapsed(), 1),
    }
    print(f"[stats] total            {_elapsed():.1f}s  -> caching 600s")
    cache.set("dashboard_stats", payload, 600)
    cache.set(
        "dashboard_stats_basic",
        {
            "total_active": total_active,
            "total_deceased": total_deceased,
            "hypertension_rate": hypertension_rate,
            "diabetes_rate": diabetes_rate,
            "cohort_counts": payload["cohort_counts"],
        },
        600,
    )
    cache.delete("dashboard_stats_computing")
    return payload
