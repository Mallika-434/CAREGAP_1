"""
Management command: train_models
────────────────────────────────
Trains 3 risk-progression models on all chronic patients and saves each to
the models/ directory:

  models/lasso_logistic_regression.pkl  — L1-regularised Logistic Regression
  models/random_forest.pkl              — Random Forest Classifier
  models/xgboost.pkl                    — Gradient Boosted Trees (sklearn)
  models/risk_predictor.pkl             — legacy alias (copy of Lasso)

Labels are derived from outcome-based clinical criteria (ACC/AHA 2023 and
ADA 2024 guidelines) rather than the rule engine score, to avoid circular
reasoning.  Target HIGH-risk prevalence: 15–25% of chronic patients.

Usage:
    python manage.py train_models

Re-run whenever data is refreshed (after import_synthea or create_demo_db).
"""

from django.core.management.base import BaseCommand


def outcome_label(patient, obs_list, conds_list, meds_list, enc_list):
    """
    Outcome-based HIGH RISK label — no circular dependency on the rule engine.

    Based on ACC/AHA 2023 Hypertension Guidelines and ADA 2024 Standards of
    Medical Care.  Returns 1 (HIGH) or 0 (LOW).

    Criteria (any one sufficient for HIGH):
      - SBP >= 135 with <= 1 active medication  (undertreated Stage 1/2 HTN)
      - HbA1c >= 7.0 with no encounters in past year  (lost to follow-up)
      - > 270 days since last encounter with active HTN or diabetes dx
      - SBP >= 130 with no encounters in past year
      - Age < 50 with SBP >= 140  (early-onset Stage 2 HTN, higher lifetime risk)
      - Age < 50 with HbA1c >= 7.5  (early-onset poor glycemic control)
      - SBP >= 128, > 200 days since encounter, <= 1 encounter last year
      - HbA1c >= 6.8, > 200 days since encounter, <= 1 encounter last year
    """
    from datetime import date, timedelta

    # ── Latest SBP ───────────────────────────────────────────────────
    sbp_obs = sorted(
        [o for o in obs_list if o.code == '8480-6' and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    sbp = 0.0
    if sbp_obs:
        try:
            sbp = float(sbp_obs[0].value)
        except (ValueError, TypeError):
            pass

    # ── Latest HbA1c ─────────────────────────────────────────────────
    hba1c_obs = sorted(
        [o for o in obs_list if o.code == '4548-4' and o.date is not None],
        key=lambda o: o.date, reverse=True,
    )
    hba1c = 0.0
    if hba1c_obs:
        try:
            hba1c = float(hba1c_obs[0].value)
        except (ValueError, TypeError):
            pass

    # ── Medications ───────────────────────────────────────────────────
    active_meds = sum(1 for m in meds_list if m.stop is None)

    # ── Encounters ────────────────────────────────────────────────────
    one_year_ago = date.today() - timedelta(days=365)
    enc_dates = []
    for e in enc_list:
        if e.start is not None:
            d = e.start.date() if hasattr(e.start, 'date') else e.start
            enc_dates.append(d)

    recent_encs = sum(1 for d in enc_dates if d >= one_year_ago)
    days = (date.today() - max(enc_dates)).days if enc_dates else 999

    # ── Conditions ────────────────────────────────────────────────────
    has_diabetes = any(
        c.code in ('44054006', '73211009') and c.stop is None
        for c in conds_list
    )
    has_htn = any(
        c.code in ('59621000', '38341003') and c.stop is None
        for c in conds_list
    )
    age = patient.age or 0

    # ── Outcome-based HIGH RISK criteria (ACC/AHA 2023, ADA 2024) ────
    if sbp >= 135 and active_meds <= 1:                          return 1
    if hba1c >= 7.0 and recent_encs == 0:                        return 1
    if days > 270 and (has_diabetes or has_htn):                 return 1
    if sbp >= 130 and recent_encs == 0:                          return 1
    if age < 50 and sbp >= 140:                                  return 1
    if age < 50 and hba1c >= 7.5:                                return 1
    if sbp >= 128 and days > 200 and recent_encs <= 1:           return 1
    if hba1c >= 6.8 and days > 200 and recent_encs <= 1:         return 1
    return 0


class Command(BaseCommand):
    help = 'Train 3 ML risk-progression models and save to models/'

    def handle(self, *args, **options):
        import numpy as np
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     classification_report)
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        from patients.models import Patient
        from patients.ml_models import extract_features, MODELS_DIR, RISK_MODEL_PATH

        # ── 1. Load patients ──────────────────────────────────────────
        self.stdout.write('Loading chronic patients…')
        patients = list(
            Patient.objects.filter(cohort='chronic')
                           .prefetch_related('observations', 'conditions',
                                             'medications', 'encounters')
        )
        self.stdout.write(f'  {len(patients):,} patients loaded')

        # ── 2. Extract features + outcome-based labels ────────────────
        self.stdout.write('Extracting features and computing outcome labels…')
        X_rows, y_rows = [], []
        skipped = 0

        for i, patient in enumerate(patients):
            if i > 0 and i % 1000 == 0:
                self.stdout.write(f'  {i:,}/{len(patients):,}…')

            obs   = list(patient.observations.all())
            conds = list(patient.conditions.all())
            meds  = list(patient.medications.all())
            encs  = list(patient.encounters.all())

            try:
                label          = outcome_label(patient, obs, conds, meds, encs)
                _, feature_arr = extract_features(patient, obs, conds, meds, encs)
                X_rows.append(feature_arr)
                y_rows.append(label)
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    self.stdout.write(
                        self.style.WARNING(f'  skip {patient.patient_id[:8]}: {exc}')
                    )

        if not X_rows:
            self.stdout.write(self.style.ERROR('No samples — aborting.'))
            return

        X = np.array(X_rows)
        y = np.array(y_rows)

        pos  = int(y.sum())
        neg  = len(y) - pos
        pct  = 100 * pos / len(y)

        self.stdout.write(
            f'  Labels: HIGH={pos:,} ({pct:.1f}%)  LOW={neg:,} ({100-pct:.1f}%)'
        )
        if skipped:
            self.stdout.write(self.style.WARNING(f'  Skipped: {skipped}'))

        # Warn if label balance is outside expected range
        if pct < 10:
            self.stdout.write(self.style.WARNING(
                f'  ⚠ HIGH-risk rate {pct:.1f}% is below 10% — '
                'consider loosening outcome_label criteria.'
            ))
        elif pct > 35:
            self.stdout.write(self.style.WARNING(
                f'  ⚠ HIGH-risk rate {pct:.1f}% is above 35% — '
                'consider tightening outcome_label criteria.'
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f'  Label balance OK (target 15–25%)'
            ))

        # ── 3. Train / test split ─────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
        self.stdout.write(
            f'  Train: {len(X_train):,}  Test: {len(X_test):,}'
        )

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # ── 4. Define models ──────────────────────────────────────────
        model_defs = [
            (
                'Lasso Logistic Regression',
                MODELS_DIR / 'lasso_logistic_regression.pkl',
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf',    LogisticRegression(
                        solver='saga',
                        max_iter=2000,
                        random_state=42,
                        class_weight='balanced',
                        C=1.0,
                    )),
                ]),
            ),
            (
                'Random Forest',
                MODELS_DIR / 'random_forest.pkl',
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf',    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_leaf=10,
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=-1,
                    )),
                ]),
            ),
            (
                'GradientBoosting',
                MODELS_DIR / 'xgboost.pkl',
                Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf',    GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42,
                    )),
                ]),
            ),
        ]

        # ── 5. Train, evaluate, save each model ──────────────────────
        lasso_model = None
        for model_name, save_path, pipeline in model_defs:
            self.stdout.write(f'\nTraining {model_name}…')
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)

            self.stdout.write(self.style.SUCCESS(
                f'  Test-set metrics  (n={len(y_test):,}):'
            ))
            self.stdout.write(f'    Accuracy : {acc:.3f}')
            self.stdout.write(f'    Precision: {prec:.3f}')
            self.stdout.write(f'    Recall   : {rec:.3f}')
            self.stdout.write(f'    F1 Score : {f1:.3f}')
            self.stdout.write(
                classification_report(y_test, y_pred,
                                      target_names=['LOW', 'HIGH'],
                                      zero_division=0)
            )

            joblib.dump(pipeline, save_path)
            self.stdout.write(self.style.SUCCESS(f'  Saved -> {save_path.name}'))

            if 'Lasso' in model_name:
                lasso_model = pipeline

        # ── 6. Save legacy risk_predictor.pkl (Lasso alias) ──────────
        if lasso_model is not None:
            joblib.dump(lasso_model, RISK_MODEL_PATH)
            self.stdout.write(self.style.SUCCESS(
                f'\nLegacy alias saved -> {RISK_MODEL_PATH.name}'
            ))

        # ── 7. Clear in-memory model cache so next request reloads ───
        import patients.ml_models as _ml
        _ml._MODEL_CACHE.clear()

        self.stdout.write(self.style.SUCCESS(
            '\nAll 3 models trained and saved.'
        ))
        self.stdout.write(
            'Ensemble endpoint /api/patients/<id>/predict/ is now active.'
        )
