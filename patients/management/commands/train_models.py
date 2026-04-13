"""
Management command: train_models
────────────────────────────────
Trains 3 risk-progression models on all chronic patients and saves each to
the models/ directory:

  models/lasso_logistic_regression.pkl  — L1-regularised Logistic Regression
  models/random_forest.pkl              — Random Forest Classifier
  models/xgboost.pkl                    — Gradient Boosted Trees (sklearn)
  models/risk_predictor.pkl             — legacy alias (copy of Lasso)

Usage:
    python manage.py train_models

Re-run whenever data is refreshed (after import_synthea or create_demo_db).
"""

from django.core.management.base import BaseCommand


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
        from patients.risk_engine import assess_risk
        from patients.ml_models import extract_features, MODELS_DIR, RISK_MODEL_PATH

        # ── 1. Load patients ──────────────────────────────────────────
        self.stdout.write('Loading chronic patients…')
        patients = list(
            Patient.objects.filter(cohort='chronic')
                           .prefetch_related('observations', 'conditions')
        )
        self.stdout.write(f'  {len(patients):,} patients loaded')

        # ── 2. Extract features + labels ─────────────────────────────
        self.stdout.write('Extracting features…')
        X_rows, y_rows = [], []
        skipped = 0

        for i, patient in enumerate(patients):
            if i > 0 and i % 1000 == 0:
                self.stdout.write(f'  {i:,}/{len(patients):,}…')

            obs   = list(patient.observations.all())
            conds = list(patient.conditions.all())

            try:
                result         = assess_risk(patient, obs, conds)
                label          = 1 if result.score >= 60 else 0
                _, feature_arr = extract_features(patient, obs, conds)
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

        pos = int(y.sum())
        neg = len(y) - pos
        self.stdout.write(
            f'  Samples: {len(y):,}  (HIGH={pos:,} {100*pos/len(y):.1f}%,'
            f'  LOW={neg:,} {100*neg/len(y):.1f}%)'
        )
        if skipped:
            self.stdout.write(self.style.WARNING(f'  Skipped: {skipped}'))

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
                        penalty='l1',
                        solver='liblinear',
                        max_iter=1000,
                        random_state=42,
                        class_weight='balanced',
                        C=0.5,
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
        for label, save_path, pipeline in model_defs:
            self.stdout.write(f'\nTraining {label}…')
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

            if 'Lasso' in label:
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
