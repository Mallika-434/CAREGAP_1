"""
patients/ml_models.py
─────────────────────
Feature extraction, trajectory prediction, and multi-model ensemble I/O
for the Predictive Modeling tab.  No Django ORM queries happen here —
callers pass pre-fetched lists of Patient / Observation / Condition
objects so this module stays pure-Python and fast.

Models directory: <project_root>/models/
  lasso_logistic_regression.pkl  — L1-regularised Logistic Regression
  random_forest.pkl              — Random Forest Classifier
  xgboost.pkl                    — GradientBoosting (sklearn GradientBoostingClassifier)
  risk_predictor.pkl             — legacy single model (backward compat)
"""

from pathlib import Path
import numpy as np

MODELS_DIR      = Path(__file__).resolve().parent.parent / 'models'
RISK_MODEL_PATH = MODELS_DIR / 'risk_predictor.pkl'

# ── Multi-model registry ──────────────────────────────────────────────────────
_MODEL_FILES = {
    'Lasso':         MODELS_DIR / 'lasso_logistic_regression.pkl',
    'Random Forest': MODELS_DIR / 'random_forest.pkl',
    'GradientBoosting': MODELS_DIR / 'xgboost.pkl',
}

_MODEL_CACHE: dict = {}   # in-memory cache so models are loaded only once

FEATURE_NAMES = [
    'latest_hba1c',
    'latest_sbp',
    'latest_dbp',
    'latest_bmi',
    'latest_cholesterol',
    'age',
    'gender_m',
    'age_group',
    'has_diabetes',
    'has_hypertension',
    'is_comorbid',
    'total_encounters',
    'encounters_last_year',
    'active_medications',
    'active_conditions',
    'low_engagement',
    'undertreated',
    'high_condition_burden',
    'missing_hba1c',
]

# LOINC codes for vitals not defined in Observation model constants
_LOINC_DBP   = '8462-4'    # Diastolic Blood Pressure
_LOINC_BMI   = '39156-5'   # Body Mass Index
_LOINC_CHOL  = '2093-3'    # Total Cholesterol


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_date(dt):
    """Normalise a date/datetime to a date object."""
    return dt.date() if hasattr(dt, 'date') else dt


def _poly_slope(obs_sorted, max_n=3):
    """
    Return polyfit slope (per observation step) from the last max_n readings.
    obs_sorted is already sorted newest-first.  Returns 0.0 on failure.
    """
    recent = obs_sorted[:max_n]
    if len(recent) < 2:
        return 0.0
    try:
        y = [float(o.value) for o in reversed(recent)]
        x = list(range(len(y)))
        return float(np.polyfit(x, y, 1)[0])
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return 0.0


# ── public API — feature extraction ──────────────────────────────────────────

def extract_features(patient, observations, conditions,
                     medications=None, encounters=None):
    """
    Build the 19-element feature vector for a patient.

    Parameters
    ----------
    patient      : patients.models.Patient instance
    observations : iterable of Observation (pre-fetched, any order)
    conditions   : iterable of Condition   (pre-fetched, any order)
    medications  : iterable of Medication  (pre-fetched, any order) or None
    encounters   : iterable of Encounter   (pre-fetched, any order) or None

    If medications or encounters are None they are fetched lazily via the
    patient's related managers (fine for single-patient predict; for bulk
    training always pass pre-fetched lists).

    Returns
    -------
    (feature_dict, numpy_array)  — both contain the same 19 values.
    """
    from patients.models import Observation as Obs, Condition as Cond
    from datetime import date as _date, timedelta

    today        = _date.today()
    one_year_ago = today - timedelta(days=365)

    obs_list  = list(observations)
    cond_list = list(conditions)

    # Lazy-load medications / encounters only if not pre-fetched
    if medications is None:
        med_list = list(patient.medications.all())
    else:
        med_list = list(medications)

    if encounters is None:
        enc_list = list(patient.encounters.all())
    else:
        enc_list = list(encounters)

    # ── HbA1c ────────────────────────────────────────────────────────
    hba1c_obs = sorted(
        [o for o in obs_list if o.code == Obs.LOINC_HBA1C and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    latest_hba1c = 0.0
    if hba1c_obs:
        try:
            latest_hba1c = float(hba1c_obs[0].value)
        except (ValueError, TypeError):
            pass

    # ── SBP ──────────────────────────────────────────────────────────
    sbp_obs = sorted(
        [o for o in obs_list if o.code == Obs.LOINC_SBP and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    latest_sbp = 0.0
    if sbp_obs:
        try:
            latest_sbp = float(sbp_obs[0].value)
        except (ValueError, TypeError):
            pass

    # ── DBP ──────────────────────────────────────────────────────────
    dbp_obs = sorted(
        [o for o in obs_list if o.code == _LOINC_DBP and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    latest_dbp = 0.0
    if dbp_obs:
        try:
            latest_dbp = float(dbp_obs[0].value)
        except (ValueError, TypeError):
            pass

    # ── BMI ──────────────────────────────────────────────────────────
    bmi_obs = sorted(
        [o for o in obs_list if o.code == _LOINC_BMI and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    latest_bmi = 0.0
    if bmi_obs:
        try:
            latest_bmi = float(bmi_obs[0].value)
        except (ValueError, TypeError):
            pass

    # ── Cholesterol ───────────────────────────────────────────────────
    chol_obs = sorted(
        [o for o in obs_list if o.code == _LOINC_CHOL and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    latest_cholesterol = 0.0
    if chol_obs:
        try:
            latest_cholesterol = float(chol_obs[0].value)
        except (ValueError, TypeError):
            pass

    # ── Age ───────────────────────────────────────────────────────────
    age = patient.age or 0

    # ── Gender ───────────────────────────────────────────────────────
    gender_m = 1 if (patient.gender or '').upper() == 'M' else 0

    # ── Age group (0=0-18, 1=19-35, 2=36-50, 3=51-65, 4=65+) ────────
    if   age <= 18: age_group = 0
    elif age <= 35: age_group = 1
    elif age <= 50: age_group = 2
    elif age <= 65: age_group = 3
    else:           age_group = 4

    # ── Conditions ───────────────────────────────────────────────────
    active_conds     = [c for c in cond_list if c.stop is None]
    active_codes     = {c.code for c in active_conds}
    active_conditions = len(active_conds)
    has_diabetes     = any(c in active_codes for c in Cond.DIABETES_CODES)
    has_hypertension = any(c in active_codes for c in Cond.HYPERTENSION_CODES)
    is_comorbid      = int(has_diabetes and has_hypertension)

    # ── Medications ───────────────────────────────────────────────────
    active_medications = sum(1 for m in med_list if m.stop is None)

    # ── Encounters ────────────────────────────────────────────────────
    total_encounters = len(enc_list)
    encounters_last_year = sum(
        1 for e in enc_list
        if e.start is not None and _to_date(e.start) >= one_year_ago
    )

    # ── Derived flags ─────────────────────────────────────────────────
    # low_engagement: no encounters in the past year
    low_engagement = int(encounters_last_year == 0)

    # undertreated: has chronic dx but zero active medications
    undertreated = int(
        active_medications == 0 and (has_diabetes or has_hypertension)
    )

    # high_condition_burden: 3 or more active conditions
    high_condition_burden = int(active_conditions >= 3)

    # missing_hba1c: no HbA1c reading in last 365 days
    recent_hba1c = any(
        o for o in hba1c_obs
        if o.date is not None and _to_date(o.date) >= one_year_ago
    )
    missing_hba1c = int(not recent_hba1c)

    feature_dict = {
        'latest_hba1c':        latest_hba1c,
        'latest_sbp':          latest_sbp,
        'latest_dbp':          latest_dbp,
        'latest_bmi':          latest_bmi,
        'latest_cholesterol':  latest_cholesterol,
        'age':                 age,
        'gender_m':            gender_m,
        'age_group':           age_group,
        'has_diabetes':        int(has_diabetes),
        'has_hypertension':    int(has_hypertension),
        'is_comorbid':         is_comorbid,
        'total_encounters':    total_encounters,
        'encounters_last_year': encounters_last_year,
        'active_medications':  active_medications,
        'active_conditions':   active_conditions,
        'low_engagement':      low_engagement,
        'undertreated':        undertreated,
        'high_condition_burden': high_condition_burden,
        'missing_hba1c':       missing_hba1c,
    }
    feature_arr = np.array([feature_dict[k] for k in FEATURE_NAMES], dtype=float)
    return feature_dict, feature_arr


# ── public API — model loading ────────────────────────────────────────────────

def load_risk_models():
    """
    Load all 3 models from disk into the in-memory cache.
    Falls back to the legacy risk_predictor.pkl if individual files are missing.
    Returns dict: {'Lasso': model, 'Random Forest': model, 'GradientBoosting': model}
    Any missing model is absent from the dict.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE:
        return _MODEL_CACHE

    try:
        import joblib
    except ImportError:
        return {}

    loaded = {}
    for name, path in _MODEL_FILES.items():
        if path.exists():
            try:
                loaded[name] = joblib.load(path)
            except Exception as exc:
                print(f'[ml] could not load {name}: {exc}')

    # Backward-compat: if none of the 3 loaded, try legacy single model
    if not loaded and RISK_MODEL_PATH.exists():
        try:
            import joblib
            m = joblib.load(RISK_MODEL_PATH)
            loaded['Lasso'] = m   # treat legacy model as Lasso slot
        except Exception as exc:
            print(f'[ml] could not load risk_predictor: {exc}')

    _MODEL_CACHE = loaded
    return _MODEL_CACHE


def load_risk_model():
    """Legacy single-model loader — returns first available model or None."""
    models = load_risk_models()
    if not models:
        return None
    return next(iter(models.values()))


# ── public API — ensemble prediction ─────────────────────────────────────────

def predict_ensemble_score(features_arr, feature_dict=None):
    """
    Run all 3 models and return an ensemble result.

    Returns
    -------
    {
        'probability':     float,           # average of all available models
        'model_scores':    {'Lasso': float, 'Random Forest': float, 'GradientBoosting': float},
        'range_min':       float,
        'range_max':       float,
        'model_available': bool,
    }
    """
    models = load_risk_models()
    if not models:
        # Full fallback — return neutral values
        return {
            'probability':     0.5,
            'model_scores':    {},
            'range_min':       0.5,
            'range_max':       0.5,
            'model_available': False,
        }

    scores = {}
    for name, model in models.items():
        try:
            prob = float(model.predict_proba([features_arr])[0][1])
            scores[name] = round(prob, 3)
        except Exception as exc:
            print(f'[ml] predict_proba failed for {name}: {exc}')

    if not scores:
        return {
            'probability':     0.5,
            'model_scores':    {},
            'range_min':       0.5,
            'range_max':       0.5,
            'model_available': False,
        }

    vals = list(scores.values())
    return {
        'probability':     round(float(np.mean(vals)), 3),
        'model_scores':    scores,
        'range_min':       round(float(min(vals)), 3),
        'range_max':       round(float(max(vals)), 3),
        'model_available': True,
    }


# ── public API — risk decomposition ──────────────────────────────────────────

def decompose_risk(feature_dict):
    """
    Run the ensemble on 3 feature variants to decompose overall risk into
    sugar-driven vs BP-driven components.

    Returns
    -------
    {
        'overall':  float,   # ensemble probability on real features
        'sugar':    float,   # BP features zeroed out
        'bp':       float,   # sugar features zeroed out
    }
    """
    # Neutral reference values
    NORMAL_HBA1C = 5.4
    NORMAL_SBP   = 118.0

    def _arr(d):
        return np.array([d[k] for k in FEATURE_NAMES], dtype=float)

    overall_arr = _arr(feature_dict)

    sugar_only = dict(feature_dict)
    sugar_only['latest_sbp']    = NORMAL_SBP
    sugar_only['bp_trend']      = 0.0
    sugar_only['has_hypertension'] = 0

    bp_only = dict(feature_dict)
    bp_only['latest_hba1c']  = NORMAL_HBA1C
    bp_only['hba1c_trend']   = 0.0
    bp_only['has_diabetes']  = 0

    overall_score = predict_ensemble_score(overall_arr)['probability']
    sugar_score   = predict_ensemble_score(_arr(sugar_only))['probability']
    bp_score      = predict_ensemble_score(_arr(bp_only))['probability']

    return {
        'overall': overall_score,
        'sugar':   sugar_score,
        'bp':      bp_score,
    }


# ── public API — trajectory helpers ──────────────────────────────────────────

def _trajectory(obs_sorted, max_n=5, worsening_threshold=None, improving_threshold=None):
    """
    Fit a line through the last max_n readings (time in days as x-axis).
    Returns (predicted_value_180d, trend_label, slope_per_day).
    """
    recent = obs_sorted[:max_n]
    if not recent:
        return None, 'unknown', 0.0

    from datetime import date as _date
    base_date = _to_date(recent[-1].date)
    x, y = [], []
    for o in reversed(recent):
        d = _to_date(o.date)
        try:
            x.append((d - base_date).days)
            y.append(float(o.value))
        except (ValueError, TypeError):
            pass

    if len(x) < 2:
        try:
            return float(recent[0].value), 'stable', 0.0
        except (ValueError, TypeError):
            return None, 'unknown', 0.0

    try:
        coeffs    = np.polyfit(x, y, 1)
        slope     = float(coeffs[0])
        last_x    = x[-1]
        predicted = round(float(np.polyval(coeffs, last_x + 180)), 1)

        slope_per_month = slope * 30
        if   slope_per_month >  (worsening_threshold or 0):
            trend = 'worsening'
        elif slope_per_month < -(improving_threshold or 0):
            trend = 'improving'
        else:
            trend = 'stable'

        return predicted, trend, slope
    except (np.linalg.LinAlgError, ValueError):
        return None, 'unknown', 0.0


def _weighted_projection(obs_sorted, max_n, horizon_days=180):
    """
    Exponentially weighted average of last max_n values, then project forward
    using a linear fit weighted by recency (more recent = higher weight).
    Returns projected float or None.
    """
    recent = obs_sorted[:max_n]
    if not recent:
        return None
    if len(recent) == 1:
        try:
            return round(float(recent[0].value), 1)
        except (ValueError, TypeError):
            return None

    from datetime import date as _date
    base_date = _to_date(recent[-1].date)
    x, y, w = [], [], []
    for i, o in enumerate(reversed(recent)):
        d = _to_date(o.date)
        try:
            xi = (d - base_date).days
            x.append(xi)
            y.append(float(o.value))
            w.append(2 ** i)   # more recent readings get higher weight
        except (ValueError, TypeError):
            pass

    if len(x) < 2:
        return None
    try:
        coeffs    = np.polyfit(x, y, 1, w=w)
        last_x    = x[-1]
        return round(float(np.polyval(coeffs, last_x + horizon_days)), 1)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _quadratic_projection(obs_sorted, max_n, horizon_days=180):
    """
    Quadratic (degree-2) polyfit on last max_n readings.
    Returns projected float or None (falls back to linear if < 3 points).
    """
    recent = obs_sorted[:max_n]
    if not recent:
        return None

    from datetime import date as _date
    base_date = _to_date(recent[-1].date)
    x, y = [], []
    for o in reversed(recent):
        d = _to_date(o.date)
        try:
            x.append((d - base_date).days)
            y.append(float(o.value))
        except (ValueError, TypeError):
            pass

    if len(x) < 2:
        return None
    degree = 2 if len(x) >= 3 else 1
    try:
        coeffs = np.polyfit(x, y, degree)
        last_x = x[-1]
        return round(float(np.polyval(coeffs, last_x + horizon_days)), 1)
    except (np.linalg.LinAlgError, ValueError):
        return None


# ── public API — multi-model trajectory projections ──────────────────────────

def predict_multi_hba1c_trajectory(observations):
    """
    Return 3 HbA1c projections for 6 months out, one per model style.

    Returns
    -------
    {
        'lasso': float or None,   # linear projection (last 3 readings)
        'rf':    float or None,   # quadratic projection (last 5 readings)
        'xgb':   float or None,   # recency-weighted projection (last 5 readings)
        'trend': str,             # overall trend label
        'slope': float,
    }
    """
    from patients.models import Observation as Obs
    hba1c_obs = sorted(
        [o for o in observations if o.code == Obs.LOINC_HBA1C and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    lasso_val, trend, slope = _trajectory(hba1c_obs, max_n=3,
                                          worsening_threshold=0.2,
                                          improving_threshold=0.2)
    rf_val  = _quadratic_projection(hba1c_obs, max_n=5)
    xgb_val = _weighted_projection(hba1c_obs,  max_n=5)

    return {
        'lasso': lasso_val,
        'rf':    rf_val,
        'xgb':   xgb_val,
        'trend': trend,
        'slope': round(slope, 5),
    }


def predict_multi_sbp_trajectory(observations):
    """
    Return 3 SBP projections for 6 months out, one per model style.

    Returns
    -------
    {
        'lasso': float or None,   # linear 3-point projection
        'rf':    float or None,   # recency-weighted projection (last 5)
        'xgb':   float or None,   # linear 5-point projection
        'trend': str,
        'slope': float,
    }
    """
    from patients.models import Observation as Obs
    sbp_obs = sorted(
        [o for o in observations if o.code == Obs.LOINC_SBP and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    lasso_val, trend, slope = _trajectory(sbp_obs, max_n=3,
                                          worsening_threshold=2,
                                          improving_threshold=2)
    rf_val, _, _  = _trajectory(sbp_obs, max_n=5,
                                 worsening_threshold=2,
                                 improving_threshold=2)
    xgb_val = _weighted_projection(sbp_obs, max_n=5)

    return {
        'lasso': lasso_val,
        'rf':    rf_val,
        'xgb':   xgb_val,
        'trend': trend,
        'slope': round(slope, 5),
    }


# ── legacy single-trajectory wrappers (kept for backward compat) ─────────────

def predict_hba1c_trajectory(observations):
    """Returns (predicted_hba1c_6mo, trend_label, slope_per_day)."""
    r = predict_multi_hba1c_trajectory(observations)
    return r['lasso'], r['trend'], r['slope']


def predict_sbp_trajectory(observations):
    """Returns (predicted_sbp_6mo, trend_label, slope_per_day)."""
    r = predict_multi_sbp_trajectory(observations)
    return r['lasso'], r['trend'], r['slope']
