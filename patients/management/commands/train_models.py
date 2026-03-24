"""
Management command: train_models
────────────────────────────────
Trains a LogisticRegression risk-progression model on all chronic patients
and saves it to models/risk_predictor.pkl.

Usage:
    python manage.py train_models

Re-run whenever data is refreshed (after import_synthea).
Training on 6 267 chronic patients typically takes < 30 s.
"""

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Train ML risk-progression model and save to models/risk_predictor.pkl'

    def handle(self, *args, **options):
        import numpy as np
        import joblib
        from sklearn.linear_model import LogisticRegression
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
                result           = assess_risk(patient, obs, conds)
                label            = 1 if result.score >= 60 else 0
                _, feature_arr   = extract_features(patient, obs, conds)
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

        # ── 4. Train pipeline (scaler → LR) ──────────────────────────
        self.stdout.write('Training LogisticRegression…')
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',   # handles label imbalance
                solver='lbfgs',
            )),
        ])
        model.fit(X_train, y_train)

        # ── 5. Evaluate ───────────────────────────────────────────────
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        self.stdout.write(self.style.SUCCESS(
            f'\nTest-set metrics  (n={len(y_test):,}):'
        ))
        self.stdout.write(f'  Accuracy : {acc:.3f}')
        self.stdout.write(f'  Precision: {prec:.3f}')
        self.stdout.write(f'  Recall   : {rec:.3f}')
        self.stdout.write(f'  F1 Score : {f1:.3f}')
        self.stdout.write('')
        self.stdout.write(classification_report(y_test, y_pred,
                                                target_names=['LOW', 'HIGH']))

        # ── 6. Save ───────────────────────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, RISK_MODEL_PATH)
        self.stdout.write(self.style.SUCCESS(
            f'Model saved → {RISK_MODEL_PATH}'
        ))
        self.stdout.write(
            'Predictive endpoint /api/patients/<id>/predict/ is now active.'
        )
