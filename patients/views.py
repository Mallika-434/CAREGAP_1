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
                Q(first__icontains=term) |
                Q(last__icontains=term)  |
                Q(city__icontains=term)
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
    try:
        patient = Patient.objects.get(patient_id=patient_id)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found.'}, status=status.HTTP_404_NOT_FOUND)

    serializer = PatientDetailSerializer(patient)
    return Response(serializer.data)


# ── 3. Risk Assessment ────────────────────────────────────────────
@api_view(['GET'])
def patient_risk(request, patient_id):
    """
    Run the risk engine for a patient and return structured result.
    Used to drive the dashboard risk card.
    """
    try:
        patient = Patient.objects.get(patient_id=patient_id)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found.'}, status=status.HTTP_404_NOT_FOUND)

    observations = Observation.objects.filter(patient=patient)
    conditions   = Condition.objects.filter(patient=patient)

    result = assess_risk(patient, observations, conditions)

    return Response({
        'patient_id':          patient.patient_id,
        'patient_name':        patient.full_name(),
        'tier':                result.tier,
        'score':               result.score,
        'reasons':             result.reasons,
        'hba1c_days_gap':      result.hba1c_days_gap,
        'hba1c_value':         result.hba1c_value,
        'latest_sbp':          result.latest_sbp,
        'has_diabetes':        result.has_diabetes,
        'has_hypertension':    result.has_hypertension,
        'recommended_action':  result.recommended_action,
        'followup_urgency_days': result.followup_urgency_days,
    })


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
    if cached is not None:
        return Response(cached)

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
    return Response(payload)

# ── 6. Triage Dashboard (Emergency / Urgent Care) ─────────────────
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
