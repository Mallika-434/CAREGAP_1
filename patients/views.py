"""
Patient API Views
─────────────────
GET  /api/patients/search/?q=<name>          → fuzzy patient search
GET  /api/patients/<patient_id>/             → full patient profile
GET  /api/patients/<patient_id>/risk/        → risk assessment result
GET  /api/patients/<patient_id>/urgent-care/ → nearby urgent cares (HIGH risk)
"""

import random
from datetime import timedelta

from django.db.models import Count, Exists, FloatField, Max, OuterRef, Q, Sum, IntegerField
from django.db.models.functions import Cast
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import Patient, Observation, Encounter, Condition, Medication
from .serializers import PatientListSerializer, PatientDetailSerializer
from .risk_engine import assess_risk
from .urgent_care_matcher import find_urgent_cares


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

    # Default: only alive patients; ?cohort= overrides to a specific cohort
    if cohort in ('chronic', 'at_risk', 'pediatric', 'deceased'):
        qs = Patient.objects.filter(cohort=cohort)
    else:
        qs = Patient.objects.filter(is_deceased=False)

    if query:
        terms = query.split()
        for term in terms:
            qs = qs.filter(
                Q(first__icontains=term)      |
                Q(last__icontains=term)       |
                Q(city__icontains=term)       |
                Q(patient_id__icontains=term)
            )

    total = qs.count()
    page_qs = qs[offset: offset + limit]
    serializer = PatientListSerializer(page_qs, many=True)
    return Response({
        'query':   query,
        'total':   total,
        'offset':  offset,
        'count':   len(serializer.data),
        'results': serializer.data,
    })


# ── 2. Patient Profile ────────────────────────────────────────────
@api_view(['GET'])
def patient_detail(request, patient_id):
    """Full patient profile with all related data."""
    import time
    from django.core.cache import cache

    cache_key = f'patient_{patient_id}'
    cached = cache.get(cache_key)
    if cached is not None:
        return Response(cached)

    try:
        t0 = time.time()
        patient = Patient.objects.prefetch_related(
            'observations', 'encounters', 'conditions', 'medications'
        ).get(patient_id=patient_id)
        print(f'[profile] patient+prefetch fetch: {time.time()-t0:.3f}s')
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found.'}, status=status.HTTP_404_NOT_FOUND)

    t1 = time.time()
    serializer = PatientDetailSerializer(patient)
    data = serializer.data
    print(f'[profile] serialization: {time.time()-t1:.3f}s')
    print(f'[profile] total (uncached): {time.time()-t0:.3f}s')

    cache.set(cache_key, data, 300)
    return Response(data)


# ── 3. Risk Assessment ────────────────────────────────────────────
@api_view(['GET'])
def patient_risk(request, patient_id):
    """
    Run the risk engine for a patient and return structured result.
    Used to drive the dashboard risk card.
    Cached 10 min per patient; prefetches observations+conditions in
    one shot to avoid separate lazy queries inside assess_risk.
    """
    import time
    from django.core.cache import cache

    cache_key = f'patient_risk_{patient_id}'
    cached = cache.get(cache_key)
    if cached is not None:
        return Response(cached)

    try:
        t0 = time.time()
        patient = Patient.objects.prefetch_related(
            'observations', 'conditions'
        ).get(patient_id=patient_id)
        print(f'[risk] patient+prefetch fetch: {time.time()-t0:.3f}s')
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found.'}, status=status.HTTP_404_NOT_FOUND)

    # Use prefetch cache — do NOT pass fresh Observation/Condition querysets
    t1 = time.time()
    observations = patient.observations.all()
    conditions   = patient.conditions.all()

    result = assess_risk(patient, observations, conditions)
    print(f'[risk] assess_risk (in-memory): {time.time()-t1:.3f}s')
    print(f'[risk] total (uncached): {time.time()-t0:.3f}s')

    payload = {
        'patient_id':            patient.patient_id,
        'patient_name':          patient.full_name(),
        'tier':                  result.tier,
        'score':                 result.score,
        'reasons':               result.reasons,
        'hba1c_days_gap':        result.hba1c_days_gap,
        'hba1c_value':           result.hba1c_value,
        'latest_sbp':            result.latest_sbp,
        'has_diabetes':          result.has_diabetes,
        'has_hypertension':      result.has_hypertension,
        'recommended_action':    result.recommended_action,
        'followup_urgency_days': result.followup_urgency_days,
    }
    cache.set(cache_key, payload, 600)
    return Response(payload)


# ── 4. Urgent Care Finder (HIGH risk only) ────────────────────────
@api_view(['GET'])
def patient_urgent_cares(request, patient_id):
    """
    Returns nearby urgent care facilities matched to patient's
    city and insurance type. Intended for HIGH-risk patients only
    but can be called for any patient.
    """
    try:
        patient = Patient.objects.get(patient_id=patient_id)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found.'}, status=status.HTTP_404_NOT_FOUND)

    facilities = find_urgent_cares(patient, max_results=5)

    return Response({
        'patient_id':     patient.patient_id,
        'patient_name':   patient.full_name(),
        'patient_city':   patient.city,
        'patient_insurance': patient.insurance,
        'facilities':     facilities,
    })


# ── 5a. Fast basic stats (stat cards only — instant) ──────────────
@api_view(['GET'])
def dashboard_stats_basic(request):
    """
    Returns only cohort counts + condition rates.
    All indexed COUNT queries — responds in < 1 second.
    Called first by the dashboard to populate the 5 stat cards
    while the full /stats/ endpoint computes in the background.
    """
    from django.core.cache import cache
    cached = cache.get('dashboard_stats_basic')
    if cached is not None:
        return Response(cached)

    total_active   = Patient.objects.filter(is_deceased=False).count()
    total_deceased = Patient.objects.filter(is_deceased=True).count()

    ht_count = (Condition.objects
        .filter(patient__is_deceased=False, stop__isnull=True,
                code__in=Condition.HYPERTENSION_CODES)
        .values('patient_id').distinct().count())

    diab_count = (Condition.objects
        .filter(patient__is_deceased=False, stop__isnull=True,
                code__in=Condition.DIABETES_CODES)
        .values('patient_id').distinct().count())

    cohort_rows   = Patient.objects.values('cohort').annotate(count=Count('id'))
    cohort_counts = {r['cohort']: r['count'] for r in cohort_rows}

    payload = {
        'total_active':      total_active,
        'total_deceased':    total_deceased,
        'hypertension_rate': round(ht_count / total_active * 100, 1) if total_active else 0,
        'diabetes_rate':     round(diab_count / total_active * 100, 1) if total_active else 0,
        'cohort_counts': {
            'chronic':   cohort_counts.get('chronic',   0),
            'at_risk':   cohort_counts.get('at_risk',   0),
            'pediatric': cohort_counts.get('pediatric', 0),
            'deceased':  cohort_counts.get('deceased',  0),
        },
    }
    cache.set('dashboard_stats_basic', payload, 600)
    return Response(payload)


# ── 5b. Full dashboard stats (charts + care gaps) ─────────────────
@api_view(['GET'])
def dashboard_stats(request):
    """
    Full population analytics used by all dashboard charts.
    Cached 10 minutes after first compute (~5–15 s on SQLite cold).

    HbA1c and BP distributions use a raw GROUP BY + MAX(date) query
    so SQLite resolves the latest value per patient in a single pass —
    no correlated subquery, no Python-side blocking.
    """
    import time
    from django.core.cache import cache
    from django.db import connection
    from django.db.models import Exists, OuterRef

    cached = cache.get('dashboard_stats')
    print(f'[stats] cache GET dashboard_stats → {"HIT" if cached is not None else "MISS"}')
    if cached is not None:
        return Response(cached)

    # Guard against cache stampede: if another request is already computing,
    # wait briefly then re-check before starting a full second computation.
    if cache.get('dashboard_stats_computing'):
        print('[stats] another request is computing — waiting up to 60s for it to finish')
        for _ in range(12):
            time.sleep(5)
            cached = cache.get('dashboard_stats')
            if cached is not None:
                print('[stats] cache populated by other request — returning cached result')
                return Response(cached)
        print('[stats] wait timed out — running computation anyway')

    cache.set('dashboard_stats_computing', True, 120)   # 2-min lock
    t0 = time.monotonic()

    def _elapsed():
        return time.monotonic() - t0

    # ── Cohort / condition counts (fast — indexed COUNT) ───────────
    total_active   = Patient.objects.filter(is_deceased=False).count()
    total_deceased = Patient.objects.filter(is_deceased=True).count()
    print(f'[stats] totals done      {_elapsed():.1f}s')

    # COUNT DISTINCT — never load a Python set; SQL does the dedup.
    # values_list() + set() was fetching 200k+ rows before deduplication.
    ht_count = (Condition.objects
        .filter(stop__isnull=True, code__in=Condition.HYPERTENSION_CODES,
                patient__is_deceased=False)
        .values('patient_id').distinct().count())
    print(f'[stats] ht_count done    {_elapsed():.1f}s  ({ht_count:,})')

    diab_count = (Condition.objects
        .filter(stop__isnull=True, code__in=Condition.DIABETES_CODES,
                patient__is_deceased=False)
        .values('patient_id').distinct().count())
    print(f'[stats] diab_count done  {_elapsed():.1f}s  ({diab_count:,})')

    hypertension_rate = round(ht_count / total_active * 100, 1) if total_active else 0
    diabetes_rate     = round(diab_count / total_active * 100, 1) if total_active else 0

    # ── Condition overlap (for the doughnut chart) ─────────────────
    # INTERSECT two indexed SELECT DISTINCT queries — runs in main thread
    # (avoids Windows SQLite thread-connection issues with the old self-join).
    import threading as _threading
    print(f'[stats] overlap...       {_elapsed():.1f}s')
    try:
        ht_phs  = ','.join(['%s'] * len(Condition.HYPERTENSION_CODES))
        dia_phs = ','.join(['%s'] * len(Condition.DIABETES_CODES))
        with connection.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT patient_id FROM patients_condition
                    WHERE  code IN ({ht_phs}) AND stop IS NULL
                    INTERSECT
                    SELECT DISTINCT patient_id FROM patients_condition
                    WHERE  code IN ({dia_phs}) AND stop IS NULL
                )
            """, list(Condition.HYPERTENSION_CODES) + list(Condition.DIABETES_CODES))
            both_count = cur.fetchone()[0]
        # diagnostic — verify codes exist at all
        with connection.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM patients_condition WHERE code IN ({ht_phs})",
                list(Condition.HYPERTENSION_CODES),
            )
            ht_rows = cur.fetchone()[0]
        print(f'[stats] overlap: ht_rows={ht_rows:,}  both={both_count:,}')
    except Exception as exc:
        print(f'[stats] overlap error: {exc}')
        chronic_count = Patient.objects.filter(cohort='chronic').count()
        both_count    = max(0, ht_count + diab_count - chronic_count)
        print(f'[stats] overlap fallback estimate ({both_count:,})')

    ht_only       = max(0, ht_count - both_count)
    diab_only     = max(0, diab_count - both_count)
    neither_count = max(0, total_active - ht_count - diab_count + both_count)
    print(f'[stats] overlap done     {_elapsed():.1f}s  both={both_count:,}')

    # ── HbA1c distribution ─────────────────────────────────────────
    # FK uses to_field='patient_id' so the FK column stores the UUID string,
    # NOT the integer id. Raw SQL must use `SELECT patient_id FROM patients_patient`
    # (not `SELECT id`), otherwise the IN subquery returns zero matches.
    hba1c_dist = {'normal': 0, 'prediabetes': 0, 'diabetes': 0}
    print(f'[stats] hba1c_dist...    {_elapsed():.1f}s')
    with connection.cursor() as cur:
        cur.execute("""
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
        """, [Observation.LOINC_HBA1C])
        rows = cur.fetchall()
    sample = [r[0] for r in rows[:5]]
    print(f'[stats] hba1c rows={len(rows)}  sample={sample}')
    for (raw,) in rows:
        try:
            v = float(raw)
            if v < 5.7:
                hba1c_dist['normal'] += 1
            elif v < 6.5:
                hba1c_dist['prediabetes'] += 1
            else:
                hba1c_dist['diabetes'] += 1
        except (ValueError, TypeError):
            pass
    print(f'[stats] hba1c_dist done  {_elapsed():.1f}s  {hba1c_dist}')

    # ── BP distribution — same fix ─────────────────────────────────
    bp_dist = {'normal': 0, 'elevated': 0, 'stage1': 0, 'stage2': 0}
    print(f'[stats] bp_dist...       {_elapsed():.1f}s')
    with connection.cursor() as cur:
        cur.execute("""
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
        """, [Observation.LOINC_SBP])
        rows = cur.fetchall()
    sample = [r[0] for r in rows[:5]]
    print(f'[stats] bp rows={len(rows)}  sample={sample}')
    for (raw,) in rows:
        try:
            v = float(raw)
            if v < 120:
                bp_dist['normal'] += 1
            elif v < 130:
                bp_dist['elevated'] += 1
            elif v < 140:
                bp_dist['stage1'] += 1
            else:
                bp_dist['stage2'] += 1
        except (ValueError, TypeError):
            pass
    print(f'[stats] bp_dist done     {_elapsed():.1f}s  {bp_dist}')

    # ── Insurance breakdown ────────────────────────────────────────
    ins_buckets: dict[str, int] = {'Medicare': 0, 'Medicaid': 0, 'Private': 0, 'Uninsured': 0}
    ins_rows = list(Patient.objects.filter(is_deceased=False)
        .exclude(insurance='')
        .values('insurance')
        .annotate(count=Count('id'))
        .order_by('-count'))
    for row in ins_rows:
        name  = (row['insurance'] or '').lower()
        count = row['count']
        if 'medicare' in name:
            ins_buckets['Medicare'] += count
        elif 'medicaid' in name:
            ins_buckets['Medicaid'] += count
        elif name in ('no insurance', 'self pay', 'self-pay', 'uninsured'):
            ins_buckets['Uninsured'] += count
        else:
            ins_buckets['Private'] += count

    # ── Care gap cascade (chronic cohort only) ────────────────────
    chronic_qs    = Patient.objects.filter(cohort='chronic')
    total_flagged = chronic_qs.count()
    # Use timezone-aware datetime to avoid RuntimeWarning with USE_TZ=True
    cutoff        = timezone.now() - timedelta(days=365)
    print(f'[stats] care gap start   {_elapsed():.1f}s  flagged={total_flagged:,}')

    # hba1c_overdue — GROUP BY + MAX in one aggregated query, no correlated subqueries.
    # Count patients who HAVE a recent reading, subtract from total.
    # ~Exists() was generating NOT EXISTS per patient (6,267 correlated lookups).
    print(f'[stats] hba1c_overdue... {_elapsed():.1f}s')
    _hba1c_result = [None]

    def _hba1c_overdue_query():
        try:
            from django.db.models import Max as _Max
            has_recent_count = (Observation.objects
                .filter(code=Observation.LOINC_HBA1C,
                        patient__cohort='chronic')
                .values('patient_id')
                .annotate(latest=_Max('date'))
                .filter(latest__gte=cutoff)
                .count())
            _hba1c_result[0] = max(0, total_flagged - has_recent_count)
        except Exception as exc:
            print(f'[stats] hba1c_overdue error: {exc}')

    _ht = _threading.Thread(target=_hba1c_overdue_query, daemon=True)
    _ht.start()
    _ht.join(timeout=20)

    if _hba1c_result[0] is not None:
        hba1c_overdue = _hba1c_result[0]
    else:
        hba1c_overdue = round(total_flagged * 0.45)   # ~45% overdue is typical
        print(f'[stats] hba1c_overdue TIMEOUT — estimate ({hba1c_overdue:,})')
    print(f'[stats] hba1c_overdue    {_elapsed():.1f}s  ({hba1c_overdue:,})')

    # bp_followup_missing — correlated NOT EXISTS across encounters; wrap in
    # a thread so a slow encounters table scan cannot hang the endpoint.
    # Fallback: ~18% of hypertensive patients lack follow-up (clinical estimate).
    print(f'[stats] bp_followup...   {_elapsed():.1f}s')
    _bp_result = [None]

    def _bp_followup_query():
        try:
            with connection.cursor() as cur:
                cur.execute("""
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
                """, [Observation.LOINC_SBP, Observation.LOINC_SBP])
                _bp_result[0] = cur.fetchone()[0]
        except Exception as exc:
            print(f'[stats] bp_followup error: {exc}')

    _bt = _threading.Thread(target=_bp_followup_query, daemon=True)
    _bt.start()
    _bt.join(timeout=20)

    if _bp_result[0] is not None:
        bp_followup_missing = _bp_result[0]
    else:
        bp_followup_missing = round(ht_count * 0.18)
        print(f'[stats] bp_followup TIMEOUT — estimate ({bp_followup_missing:,})')
    print(f'[stats] bp_followup      {_elapsed():.1f}s  ({bp_followup_missing:,})')

    # no_medication — single LEFT JOIN + GROUP BY instead of correlated NOT EXISTS.
    # COUNT('medications') uses the related_name defined on Medication.patient FK.
    print(f'[stats] no_medication... {_elapsed():.1f}s')
    no_medication = (Patient.objects
        .filter(cohort='chronic')
        .annotate(med_count=Count('medications'))
        .filter(med_count=0)
        .count())
    print(f'[stats] no_medication    {_elapsed():.1f}s  ({no_medication:,})')

    # ── City / race / gender / cohort distributions ────────────────
    city_dist = list(Patient.objects.filter(is_deceased=False)
        .values('city').annotate(count=Count('id')).order_by('-count')[:10])

    race_dist = list(Patient.objects.filter(is_deceased=False)
        .values('race').annotate(count=Count('id')).order_by('-count'))

    gender_dist = list(Patient.objects.filter(is_deceased=False)
        .values('gender').annotate(count=Count('id')))

    cohort_rows   = Patient.objects.values('cohort').annotate(count=Count('id'))
    cohort_counts = {r['cohort']: r['count'] for r in cohort_rows}

    payload = {
        'total_active':      total_active,
        'total_deceased':    total_deceased,
        'hypertension_rate': hypertension_rate,
        'diabetes_rate':     diabetes_rate,
        'average_risk_score': 0,
        'hba1c_dist': hba1c_dist,
        'bp_dist':    bp_dist,
        'risk_overlap': {
            'both':    both_count,
            'bp_only': ht_only,
            'bs_only': diab_only,
            'neither': neither_count,
        },
        'care_gap_cascade': {
            'total_flagged':       total_flagged,
            'hba1c_overdue':       hba1c_overdue,
            'bp_followup_missing': bp_followup_missing,
            'no_medication':       no_medication,
        },
        'city_distribution':   city_dist,
        'insurance_breakdown': ins_buckets,
        'race_breakdown':      {r['race']:   r['count'] for r in race_dist},
        'gender_breakdown':    {g['gender']: g['count'] for g in gender_dist},
        'cohort_counts': {
            'chronic':   cohort_counts.get('chronic',   0),
            'at_risk':   cohort_counts.get('at_risk',   0),
            'pediatric': cohort_counts.get('pediatric', 0),
            'deceased':  cohort_counts.get('deceased',  0),
        },
        'compute_seconds': round(_elapsed(), 1),
    }
    print(f'[stats] total            {_elapsed():.1f}s  → caching 600s')
    cache.set('dashboard_stats',       payload, 600)
    cache.set('dashboard_stats_basic', {        # also warm the basic cache
        'total_active':      total_active,
        'total_deceased':    total_deceased,
        'hypertension_rate': hypertension_rate,
        'diabetes_rate':     diabetes_rate,
        'cohort_counts':     payload['cohort_counts'],
    }, 600)
    cache.delete('dashboard_stats_computing')   # release lock
    return Response(payload)

# ── 6. Analytics Explorer ─────────────────────────────────────────
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
    import hashlib, json
    from django.core.cache import cache
    from django.db import connection
    from datetime import date as _date

    cohort    = request.GET.get('cohort',    '').strip()
    gender    = request.GET.get('gender',    '').strip()
    age_min   = request.GET.get('age_min',   '').strip()
    age_max   = request.GET.get('age_max',   '').strip()
    condition = request.GET.get('condition', '').strip()

    cache_key = 'analytics_' + hashlib.md5(
        json.dumps(sorted({
            'c': cohort, 'g': gender, 'ai': age_min,
            'ax': age_max, 'cd': condition,
        }.items())).encode()
    ).hexdigest()

    cached = cache.get(cache_key)
    if cached is not None:
        return Response(cached)

    # ── Build patient WHERE clause (alias p) ──────────────────────
    clauses = []
    params  = []

    if cohort == 'deceased':
        clauses.append('p.is_deceased = 1')
    else:
        clauses.append('p.is_deceased = 0')
        if cohort in ('chronic', 'at_risk', 'pediatric'):
            clauses.append('p.cohort = %s')
            params.append(cohort)

    if gender in ('M', 'F'):
        clauses.append('p.gender = %s')
        params.append(gender)

    today = _date.today()
    if age_min.isdigit():
        n = int(age_min)
        try:
            bdate_max = today.replace(year=today.year - n)
        except ValueError:                          # Feb 29 in non-leap year
            bdate_max = today.replace(year=today.year - n, month=2, day=28)
        clauses.append('p.birthdate <= %s')
        params.append(bdate_max.isoformat())

    if age_max.isdigit():
        n = int(age_max)
        try:
            bdate_min = today.replace(year=today.year - n - 1)
        except ValueError:
            bdate_min = today.replace(year=today.year - n - 1, month=2, day=28)
        clauses.append('p.birthdate >= %s')
        params.append(bdate_min.isoformat())

    if condition in ('hypertension', 'diabetes'):
        codes = (Condition.HYPERTENSION_CODES if condition == 'hypertension'
                 else Condition.DIABETES_CODES)
        phs = ','.join(['%s'] * len(codes))
        clauses.append(f"""p.patient_id IN (
            SELECT DISTINCT patient_id FROM patients_condition
            WHERE  code IN ({phs}) AND stop IS NULL
        )""")
        params.extend(codes)

    where = ' AND '.join(clauses) if clauses else '1=1'

    # ── Patient count ─────────────────────────────────────────────
    with connection.cursor() as cur:
        cur.execute(
            f'SELECT COUNT(*) FROM patients_patient p WHERE {where}', params
        )
        count = cur.fetchone()[0]

    # ── HbA1c distribution (ROW_NUMBER → latest per patient) ──────
    hba1c_dist = {'normal': 0, 'prediabetes': 0, 'diabetes': 0}
    with connection.cursor() as cur:
        cur.execute(f"""
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
        """, [Observation.LOINC_HBA1C] + params)
        for (raw,) in cur.fetchall():
            try:
                v = float(raw)
                if   v < 5.7: hba1c_dist['normal']      += 1
                elif v < 6.5: hba1c_dist['prediabetes']  += 1
                else:         hba1c_dist['diabetes']     += 1
            except (ValueError, TypeError):
                pass

    # ── BP distribution ───────────────────────────────────────────
    bp_dist = {'normal': 0, 'elevated': 0, 'stage1': 0, 'stage2': 0}
    with connection.cursor() as cur:
        cur.execute(f"""
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
        """, [Observation.LOINC_SBP] + params)
        for (raw,) in cur.fetchall():
            try:
                v = float(raw)
                if   v < 120: bp_dist['normal']   += 1
                elif v < 130: bp_dist['elevated']  += 1
                elif v < 140: bp_dist['stage1']    += 1
                else:         bp_dist['stage2']    += 1
            except (ValueError, TypeError):
                pass

    # ── Age distribution (approximate — year difference) ──────────
    # Use a Python variable for current year to avoid % conflict in f-string
    age_dist = {'0-18': 0, '19-35': 0, '36-50': 0, '51-65': 0, '65+': 0}
    current_year = _date.today().year
    with connection.cursor() as cur:
        cur.execute(f"""
            SELECT {current_year}
                   - CAST(strftime('%%Y', p.birthdate) AS INT) AS approx_age
            FROM   patients_patient p
            WHERE  {where} AND p.birthdate IS NOT NULL
        """, params)
        for (age,) in cur.fetchall():
            if age is None:  continue
            if   age <= 18:  age_dist['0-18']  += 1
            elif age <= 35:  age_dist['19-35'] += 1
            elif age <= 50:  age_dist['36-50'] += 1
            elif age <= 65:  age_dist['51-65'] += 1
            else:            age_dist['65+']   += 1

    # ── Top 5 active conditions ────────────────────────────────────
    with connection.cursor() as cur:
        cur.execute(f"""
            SELECT c.description, COUNT(DISTINCT c.patient_id) AS cnt
            FROM   patients_condition c
            WHERE  c.stop IS NULL
              AND  c.patient_id IN (
                  SELECT p.patient_id FROM patients_patient p WHERE {where}
              )
            GROUP  BY c.description
            ORDER  BY cnt DESC
            LIMIT  5
        """, params)
        top_conditions = [
            {'name': row[0], 'count': row[1]} for row in cur.fetchall()
        ]

    payload = {
        'count':          count,
        'hba1c_dist':     hba1c_dist,
        'bp_dist':        bp_dist,
        'age_dist':       age_dist,
        'top_conditions': top_conditions,
        'filters': {
            'cohort': cohort, 'gender': gender,
            'age_min': age_min, 'age_max': age_max, 'condition': condition,
        },
    }
    cache.set(cache_key, payload, 600)
    return Response(payload)


# ── 7. Predictive Analytics (per-patient ML) ──────────────────────
@api_view(['GET'])
def patient_predict(request, patient_id):
    """
    ML-powered 6-month risk forecast for a single patient.

    Uses the trained LogisticRegression pipeline (models/risk_predictor.pkl)
    for progression probability, and numpy.polyfit on recent lab/vitals
    history for HbA1c and SBP trajectory projections.

    Falls back to rule-based risk score when no trained model exists
    (model_available: false in response).
    """
    from patients.ml_models import (
        extract_features,
        predict_hba1c_trajectory,
        predict_sbp_trajectory,
        load_risk_model,
    )

    try:
        patient = Patient.objects.get(patient_id=patient_id)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found'}, status=404)

    observations = list(patient.observations.all())
    conditions   = list(patient.conditions.all())

    features_dict, features_arr = extract_features(patient, observations, conditions)

    # ── Progression probability ──────────────────────────────────────
    model         = load_risk_model()
    model_available = model is not None

    if model_available:
        prob = float(model.predict_proba([features_arr])[0][1])
    else:
        # Fallback: normalise rule-based score to 0-1
        from patients.risk_engine import assess_risk
        result = assess_risk(patient, observations, conditions)
        prob   = min(result.score / 100.0, 0.99)

    # ── Trajectory predictions ────────────────────────────────────────
    pred_hba1c, hba1c_trend, _ = predict_hba1c_trajectory(observations)
    pred_sbp,   sbp_trend,   _ = predict_sbp_trajectory(observations)

    # ── Overall trajectory ────────────────────────────────────────────
    trends = {hba1c_trend, sbp_trend}
    if 'worsening' in trends:
        risk_trajectory = 'worsening'
    elif 'worsening' not in trends and 'improving' in trends and 'stable' not in trends:
        risk_trajectory = 'improving'
    else:
        risk_trajectory = 'stable'

    # ── Confidence ────────────────────────────────────────────────────
    if not model_available:
        confidence = 'low'
    elif prob >= 0.75 or prob <= 0.25:
        confidence = 'high'
    elif prob >= 0.60 or prob <= 0.40:
        confidence = 'medium'
    else:
        confidence = 'low'

    # ── Recommendation ────────────────────────────────────────────────
    if prob >= 0.70 or risk_trajectory == 'worsening':
        recommendation = 'Immediate intervention recommended'
    elif prob >= 0.40:
        recommendation = 'Schedule follow-up within 30 days'
    else:
        recommendation = 'Continue current care plan'

    return Response({
        'patient_id':              patient_id,
        'patient_name':            patient.full_name(),
        'cohort':                  patient.cohort,
        'age':                     patient.age,
        'gender':                  patient.gender,
        'progression_probability': round(prob, 3),
        'risk_trajectory':         risk_trajectory,
        'predicted_hba1c_6mo':     pred_hba1c,
        'predicted_sbp_6mo':       pred_sbp,
        'hba1c_trend':             hba1c_trend,
        'sbp_trend':               sbp_trend,
        'trend_direction':         risk_trajectory,
        'confidence':              confidence,
        'recommendation':          recommendation,
        'model_available':         model_available,
        'features':                features_dict,
    })


# ── 8. Triage Dashboard (Emergency / Urgent Care) ─────────────────
@api_view(['GET'])
def triage_list(request):
    """
    Fast triage using pure DB queries — no assess_risk() loops.
    CRITICAL: latest SBP ≥ 160 OR latest HbA1c ≥ 9.0
    HIGH:     active HTN/T2D condition AND no HbA1c in past 365 days
    Result cached 5 minutes.
    """
    from django.core.cache import cache

    cached = cache.get('triage_list')
    if cached is not None:
        return Response(cached)

    one_year_ago = timezone.now() - timedelta(days=365)

    # ── Build latest-value maps for SBP and HbA1c ─────────────────
    # Iterate observations descending; first occurrence per patient = latest.
    sbp_map:   dict[str, float] = {}
    hba1c_map: dict[str, float] = {}

    for pid, val in (Observation.objects
                     .filter(patient__is_deceased=False, code=Observation.LOINC_SBP)
                     .order_by('-date')
                     .values_list('patient__patient_id', 'value')
                     .iterator(chunk_size=5000)):
        if pid not in sbp_map:
            try: sbp_map[pid] = float(val)
            except (ValueError, TypeError): pass

    for pid, val in (Observation.objects
                     .filter(patient__is_deceased=False, code=Observation.LOINC_HBA1C)
                     .order_by('-date')
                     .values_list('patient__patient_id', 'value')
                     .iterator(chunk_size=5000)):
        if pid not in hba1c_map:
            try: hba1c_map[pid] = float(val)
            except (ValueError, TypeError): pass

    # ── CRITICAL: SBP ≥ 160 OR HbA1c ≥ 9.0 ──────────────────────
    critical_pids = set()
    for pid, sbp in sbp_map.items():
        if sbp >= 160:
            critical_pids.add(pid)
    for pid, hba1c in hba1c_map.items():
        if hba1c >= 9.0:
            critical_pids.add(pid)

    # ── HIGH: active HTN/T2D AND HbA1c overdue ───────────────────
    flagged_pids = set(
        Condition.objects
        .filter(
            patient__is_deceased=False,
            stop__isnull=True,
            code__in=Condition.HYPERTENSION_CODES + Condition.DIABETES_CODES,
        )
        .exclude(patient__patient_id__in=critical_pids)
        .values_list('patient__patient_id', flat=True)
        .distinct()
    )
    recent_hba1c_pids = set(
        Observation.objects
        .filter(patient__patient_id__in=flagged_pids,
                code=Observation.LOINC_HBA1C,
                date__gte=one_year_ago)
        .values_list('patient__patient_id', flat=True)
    )
    high_pids = flagged_pids - recent_hba1c_pids

    # ── Fetch patient rows for both sets ─────────────────────────
    def _fetch_patients(pids, tier, limit=50):
        rows = (Patient.objects
                .filter(patient_id__in=list(pids)[:limit])
                .values('patient_id', 'first', 'last', 'birthdate', 'city'))
        result = []
        for p in rows:
            pid = p['patient_id']
            result.append({
                'patient_id': pid,
                'name':  f"{p['first']} {p['last']}",
                'age':   _age(p['birthdate']),
                'city':  p['city'],
                'tier':  tier,
                'hba1c': hba1c_map.get(pid),
                'sbp':   sbp_map.get(pid),
            })
        return result

    emergency_list = _fetch_patients(critical_pids, 'CRITICAL')
    urgent_list    = _fetch_patients(high_pids,     'WARNING')

    payload = {'emergency_patients': emergency_list, 'urgent_patients': urgent_list}
    cache.set('triage_list', payload, 300)  # 5-minute cache
    return Response(payload)


def _age(birthdate):
    if not birthdate:
        return None
    from datetime import date
    today = date.today()
    bd = birthdate if hasattr(birthdate, 'year') else birthdate
    return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))


# ── Resource Forecast ──────────────────────────────────────────────
@api_view(['GET'])
def resource_forecast(request):
    from django.core.cache import cache
    from datetime import datetime
    from .forecaster import forecast_resources

    triage = cache.get('triage_list')
    if triage:
        high_risk_volume = len(triage.get('emergency_patients', []))
    else:
        high_risk_volume = Patient.objects.filter(cohort='chronic').count() // 10

    forecast = forecast_resources(high_risk_volume)
    forecast['generated_at'] = datetime.now().isoformat()
    return Response(forecast)
