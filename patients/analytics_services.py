import hashlib
import json
from datetime import date as current_date

from django.core.cache import cache
from django.db import connection

from .models import Condition, Observation


def get_analytics_payload(cohort="", gender="", age_min="", age_max="", condition=""):
    cohort = (cohort or "").strip()
    gender = (gender or "").strip()
    age_min = (age_min or "").strip()
    age_max = (age_max or "").strip()
    condition = (condition or "").strip()

    cache_key = "analytics_" + hashlib.md5(
        json.dumps(
            sorted(
                {
                    "c": cohort,
                    "g": gender,
                    "ai": age_min,
                    "ax": age_max,
                    "cd": condition,
                }.items()
            )
        ).encode()
    ).hexdigest()

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    clauses = []
    params = []

    if cohort == "deceased":
        clauses.append("p.is_deceased = 1")
    else:
        clauses.append("p.is_deceased = 0")
        if cohort in ("chronic", "at_risk", "pediatric"):
            clauses.append("p.cohort = %s")
            params.append(cohort)

    if gender in ("M", "F"):
        clauses.append("p.gender = %s")
        params.append(gender)

    today = current_date.today()
    if age_min.isdigit():
        n = int(age_min)
        try:
            bdate_max = today.replace(year=today.year - n)
        except ValueError:
            bdate_max = today.replace(year=today.year - n, month=2, day=28)
        clauses.append("p.birthdate <= %s")
        params.append(bdate_max.isoformat())

    if age_max.isdigit():
        n = int(age_max)
        try:
            bdate_min = today.replace(year=today.year - n - 1)
        except ValueError:
            bdate_min = today.replace(year=today.year - n - 1, month=2, day=28)
        clauses.append("p.birthdate >= %s")
        params.append(bdate_min.isoformat())

    if condition in ("hypertension", "diabetes"):
        codes = (
            Condition.HYPERTENSION_CODES
            if condition == "hypertension"
            else Condition.DIABETES_CODES
        )
        phs = ",".join(["%s"] * len(codes))
        clauses.append(
            f"""p.patient_id IN (
            SELECT DISTINCT patient_id FROM patients_condition
            WHERE  code IN ({phs}) AND stop IS NULL
        )"""
        )
        params.extend(codes)

    where = " AND ".join(clauses) if clauses else "1=1"

    with connection.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM patients_patient p WHERE {where}", params)
        count = cur.fetchone()[0]

    hba1c_dist = {"normal": 0, "prediabetes": 0, "diabetes": 0}
    with connection.cursor() as cur:
        cur.execute(
            f"""
            SELECT value FROM (
                SELECT o.value,
                       ROW_NUMBER() OVER (
                           PARTITION BY o.patient_id ORDER BY o.date DESC
                       ) AS rn
                FROM   patients_observation o
                WHERE  o.code = %s
                  AND  o.patient_id IN (
                      SELECT p.patient_id FROM patients_patient p WHERE {where}
                  )
            ) WHERE rn = 1
        """,
            [Observation.LOINC_HBA1C] + params,
        )
        for (raw,) in cur.fetchall():
            try:
                v = float(raw)
                if v < 5.7:
                    hba1c_dist["normal"] += 1
                elif v < 6.5:
                    hba1c_dist["prediabetes"] += 1
                else:
                    hba1c_dist["diabetes"] += 1
            except (ValueError, TypeError):
                pass

    bp_dist = {"normal": 0, "elevated": 0, "stage1": 0, "stage2": 0}
    with connection.cursor() as cur:
        cur.execute(
            f"""
            SELECT value FROM (
                SELECT o.value,
                       ROW_NUMBER() OVER (
                           PARTITION BY o.patient_id ORDER BY o.date DESC
                       ) AS rn
                FROM   patients_observation o
                WHERE  o.code = %s
                  AND  o.patient_id IN (
                      SELECT p.patient_id FROM patients_patient p WHERE {where}
                  )
            ) WHERE rn = 1
        """,
            [Observation.LOINC_SBP] + params,
        )
        for (raw,) in cur.fetchall():
            try:
                v = float(raw)
                if v < 120:
                    bp_dist["normal"] += 1
                elif v < 130:
                    bp_dist["elevated"] += 1
                elif v < 140:
                    bp_dist["stage1"] += 1
                else:
                    bp_dist["stage2"] += 1
            except (ValueError, TypeError):
                pass

    age_dist = {"0-18": 0, "19-35": 0, "36-50": 0, "51-65": 0, "65+": 0}
    current_year = current_date.today().year
    with connection.cursor() as cur:
        cur.execute(
            f"""
            SELECT {current_year}
                   - CAST(strftime('%%Y', p.birthdate) AS INT) AS approx_age
            FROM   patients_patient p
            WHERE  {where} AND p.birthdate IS NOT NULL
        """,
            params,
        )
        for (age,) in cur.fetchall():
            if age is None:
                continue
            if age <= 18:
                age_dist["0-18"] += 1
            elif age <= 35:
                age_dist["19-35"] += 1
            elif age <= 50:
                age_dist["36-50"] += 1
            elif age <= 65:
                age_dist["51-65"] += 1
            else:
                age_dist["65+"] += 1

    with connection.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.description, COUNT(DISTINCT c.patient_id) AS cnt
            FROM   patients_condition c
            WHERE  c.stop IS NULL
              AND  c.patient_id IN (
                  SELECT p.patient_id FROM patients_patient p WHERE {where}
              )
            GROUP  BY c.description
            ORDER  BY cnt DESC
            LIMIT  5
        """,
            params,
        )
        top_conditions = [{"name": row[0], "count": row[1]} for row in cur.fetchall()]

    payload = {
        "count": count,
        "hba1c_dist": hba1c_dist,
        "bp_dist": bp_dist,
        "age_dist": age_dist,
        "top_conditions": top_conditions,
        "filters": {
            "cohort": cohort,
            "gender": gender,
            "age_min": age_min,
            "age_max": age_max,
            "condition": condition,
        },
    }
    cache.set(cache_key, payload, 600)
    return payload
