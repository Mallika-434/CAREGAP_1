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
    Cached 10 minutes after first compute.
    Uses correlated subqueries so only ONE row per patient is fetched
    for HbA1c and BP — no full-table scans.
    """
    import time
    from django.core.cache import cache
    from django.db.models import Subquery, OuterRef, Exists

    cached = cache.get('dashboard_stats')
    if cached is not None:
        return Response(cached)

    t0 = time.monotonic()

    def _elapsed():
        return time.monotonic() - t0

    # ── Cohort / basic counts (fast — indexed) ─────────────────────
    total_active   = Patient.objects.filter(is_deceased=False).count()
    total_deceased = Patient.objects.filter(is_deceased=True).count()

    ht_ids = set(Condition.objects.filter(
        patient__is_deceased=False, stop__isnull=True,
        code__in=Condition.HYPERTENSION_CODES,
    ).values_list('patient_id', flat=True))

    diab_ids = set(Condition.objects.filter(
        patient__is_deceased=False, stop__isnull=True,
        code__in=Condition.DIABETES_CODES,
    ).values_list('patient_id', flat=True))

    both_ids          = ht_ids & diab_ids
    hypertension_rate = round(len(ht_ids) / total_active * 100, 1) if total_active else 0
    diabetes_rate     = round(len(diab_ids) / total_active * 100, 1) if total_active else 0

    # ── HbA1c distribution — ONE subquery, one value per patient ───
    # SQL: for each chronic patient SELECT the latest HbA1c value
    # Uses (patient_id, code, date) index — does NOT scan all obs rows
    _TIMEOUT = 8  # seconds; if exceeded, return zeros for slow sections

    hba1c_dist = {'normal': 0, 'prediabetes': 0, 'diabetes': 0}
    bp_dist    = {'normal': 0, 'elevated': 0, 'stage1': 0, 'stage2': 0}

    if _elapsed() < _TIMEOUT:
        latest_hba1c_sq = (Observation.objects
            .filter(patient_id=OuterRef('patient_id'),
                    code=Observation.LOINC_HBA1C)
            .order_by('-date')
            .values('value')[:1])

        for raw in (Patient.objects
                    .filter(cohort='chronic')
                    .annotate(latest_hba1c=Subquery(latest_hba1c_sq))
                    .exclude(latest_hba1c=None)
                    .values_list('latest_hba1c', flat=True)
                    .iterator(chunk_size=2000)):
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

    # ── BP distribution — same correlated-subquery pattern ─────────
    if _elapsed() < _TIMEOUT:
        latest_sbp_sq = (Observation.objects
            .filter(patient_id=OuterRef('patient_id'),
                    code=Observation.LOINC_SBP)
            .order_by('-date')
            .values('value')[:1])

        for raw in (Patient.objects
                    .filter(cohort='chronic')
                    .annotate(latest_sbp=Subquery(latest_sbp_sq))
                    .exclude(latest_sbp=None)
                    .values_list('latest_sbp', flat=True)
                    .iterator(chunk_size=2000)):
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

    one_year_ago = timezone.now() - timedelta(days=365)

    # hba1c_overdue: NOT EXISTS a recent HbA1c — single correlated subquery
    has_recent_hba1c = Observation.objects.filter(
        patient_id=OuterRef('patient_id'),
        code=Observation.LOINC_HBA1C,
        date__gte=one_year_ago,
    )
    hba1c_overdue = chronic_qs.filter(~Exists(has_recent_hba1c)).count()

    # bp_followup_missing: latest SBP ≥ 160 AND no follow-up encounter
    # Use subquery to get latest SBP per patient, then filter ≥ 160,
    # then check encounters in a batched loop (capped at 300 for speed).
    bp_followup_missing = 0
    if _elapsed() < _TIMEOUT:
        latest_sbp_sq2 = (Observation.objects
            .filter(patient_id=OuterRef('patient_id'),
                    code=Observation.LOINC_SBP)
            .order_by('-date')
            .values('value')[:1])
        latest_sbp_date_sq = (Observation.objects
            .filter(patient_id=OuterRef('patient_id'),
                    code=Observation.LOINC_SBP)
            .order_by('-date')
            .values('date')[:1])

        critical_rows = list(Patient.objects
            .filter(cohort='chronic')
            .annotate(latest_sbp=Subquery(latest_sbp_sq2),
                      latest_sbp_date=Subquery(latest_sbp_date_sq))
            .exclude(latest_sbp=None)
            .filter(latest_sbp__gte='160')   # string compare works for numeric strings
            .values_list('patient_id', 'latest_sbp_date')[:300])

        for pid, reading_dt in critical_rows:
            if reading_dt is None:
                bp_followup_missing += 1
                continue
            has_followup = Encounter.objects.filter(
                patient__patient_id=pid,
                start__gte=reading_dt,
                start__lte=reading_dt + timedelta(days=30),
            ).exists()
            if not has_followup:
                bp_followup_missing += 1

    # no_medication: chronic patients with no Medication rows
    has_med       = Medication.objects.filter(patient=OuterRef('pk'))
    no_medication = chronic_qs.filter(~Exists(has_med)).count()

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
            'both':    len(both_ids),
            'bp_only': len(ht_ids - both_ids),
            'bs_only': len(diab_ids - both_ids),
            'neither': total_active - len(ht_ids | diab_ids),
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
    cache.set('dashboard_stats',       payload, 600)  # 10-minute cache
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
