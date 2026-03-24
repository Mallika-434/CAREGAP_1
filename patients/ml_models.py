"""
patients/ml_models.py
─────────────────────
Feature extraction, trajectory prediction, and model I/O for the
Predictive Modeling tab.  No Django ORM queries happen here —
callers pass pre-fetched lists of Patient / Observation / Condition
objects so this module stays pure-Python and fast.

Models directory: <project_root>/models/
  risk_predictor.pkl  — scikit-learn Pipeline (StandardScaler + LR)
"""

from pathlib import Path
import numpy as np

MODELS_DIR      = Path(__file__).resolve().parent.parent / 'models'
RISK_MODEL_PATH = MODELS_DIR / 'risk_predictor.pkl'

FEATURE_NAMES = [
    'latest_hba1c',
    'latest_sbp',
    'age',
    'has_diabetes',
    'has_hypertension',
    'hba1c_trend',
    'bp_trend',
    'days_since_last_visit',
    'care_gaps_count',
]


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


# ── public API ────────────────────────────────────────────────────────────────

def extract_features(patient, observations, conditions):
    """
    Build the 9-element feature vector for a patient.

    Parameters
    ----------
    patient      : patients.models.Patient instance
    observations : iterable of Observation (pre-fetched, any order)
    conditions   : iterable of Condition   (pre-fetched, any order)

    Returns
    -------
    (feature_dict, numpy_array)  — both contain the same 9 values.
    """
    from patients.models import Observation as Obs, Condition as Cond
    from datetime import date as _date

    today = _date.today()
    obs_list  = list(observations)
    cond_list = list(conditions)

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

    # ── Age ───────────────────────────────────────────────────────────
    age = patient.age or 0

    # ── Conditions ───────────────────────────────────────────────────
    active_codes = {c.code for c in cond_list if c.stop is None}
    has_diabetes     = any(c in active_codes for c in Cond.DIABETES_CODES)
    has_hypertension = any(c in active_codes for c in Cond.HYPERTENSION_CODES)

    # ── Trends (polyfit slope over last 3 readings) ───────────────────
    hba1c_trend = _poly_slope(hba1c_obs, max_n=3)
    bp_trend    = _poly_slope(sbp_obs,   max_n=3)

    # ── Days since last visit (proxy: latest observation date) ───────
    all_dates = [o.date for o in obs_list if o.date is not None]
    if all_dates:
        days_since_last_visit = (today - _to_date(max(all_dates))).days
    else:
        days_since_last_visit = 999

    # ── Care gaps count (0-3) ─────────────────────────────────────────
    care_gaps = 0
    if latest_hba1c >= 8.0:
        care_gaps += 1
    if latest_sbp >= 140:
        care_gaps += 1
    if hba1c_obs:
        last_hba1c_days = (today - _to_date(hba1c_obs[0].date)).days
        if last_hba1c_days > 365:
            care_gaps += 1
    elif has_diabetes or has_hypertension:
        care_gaps += 1

    feature_dict = {
        'latest_hba1c':          latest_hba1c,
        'latest_sbp':            latest_sbp,
        'age':                   age,
        'has_diabetes':          int(has_diabetes),
        'has_hypertension':      int(has_hypertension),
        'hba1c_trend':           hba1c_trend,
        'bp_trend':              bp_trend,
        'days_since_last_visit': min(days_since_last_visit, 999),
        'care_gaps_count':       care_gaps,
    }
    feature_arr = np.array([feature_dict[k] for k in FEATURE_NAMES], dtype=float)
    return feature_dict, feature_arr


def _trajectory(obs_sorted, max_n=5, worsening_threshold=None, improving_threshold=None):
    """
    Fit a line through the last max_n readings (time in days as x-axis).
    Returns (predicted_value_180d, trend_label, slope_per_day).
    """
    recent = obs_sorted[:max_n]
    if not recent:
        return None, 'unknown', 0.0

    # Use actual day offsets so the slope has real units (units/day)
    from datetime import date as _date
    base_date = _to_date(recent[-1].date)   # oldest of the selection
    x, y = [], []
    for o in reversed(recent):              # chronological order
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
        coeffs  = np.polyfit(x, y, 1)
        slope   = float(coeffs[0])                    # units per day
        last_x  = x[-1]
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


def predict_hba1c_trajectory(observations):
    """
    Returns (predicted_hba1c_6mo, trend_label, slope_per_day).
    trend_label: 'improving' | 'stable' | 'worsening' | 'unknown'
    """
    from patients.models import Observation as Obs
    hba1c_obs = sorted(
        [o for o in observations if o.code == Obs.LOINC_HBA1C and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    return _trajectory(hba1c_obs, max_n=5,
                       worsening_threshold=0.2,
                       improving_threshold=0.2)


def predict_sbp_trajectory(observations):
    """
    Returns (predicted_sbp_6mo, trend_label, slope_per_day).
    """
    from patients.models import Observation as Obs
    sbp_obs = sorted(
        [o for o in observations if o.code == Obs.LOINC_SBP and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    return _trajectory(sbp_obs, max_n=5,
                       worsening_threshold=2,
                       improving_threshold=2)


def load_risk_model():
    """Load the trained Pipeline from disk. Returns None if not yet trained."""
    if not RISK_MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(RISK_MODEL_PATH)
    except Exception as exc:
        print(f'[ml] could not load risk model: {exc}')
        return None
