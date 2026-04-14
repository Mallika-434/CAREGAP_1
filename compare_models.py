import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'caregap.settings')
django.setup()

import numpy as np
import joblib
import sqlite3
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)

print("=" * 65)
print("MODEL COMPARISON: LOCAL vs COLAB")
print("=" * 65)

# ── Load test data from SQLite ────────────────────────────────
print("\nLoading test data...")

conn = sqlite3.connect('db.sqlite3')
cursor = conn.cursor()

query = """
SELECT 
    p.patient_id,
    CAST(
        (strftime('%Y', 'now') - strftime('%Y', p.birthdate)) -
        (strftime('%m-%d', 'now') < strftime('%m-%d', p.birthdate))
    AS INTEGER) as age,
    p.gender,
    CASE WHEN d.patient_id IS NOT NULL THEN 1 ELSE 0 END as has_diabetes,
    CASE WHEN h.patient_id IS NOT NULL THEN 1 ELSE 0 END as has_hypertension,
    COALESCE(CAST(hba1c.value AS REAL), 0.0) as latest_hba1c,
    COALESCE(CAST(sbp.value AS REAL), 0.0) as latest_sbp,
    COALESCE(CAST(dbp.value AS REAL), 0.0) as latest_dbp,
    COALESCE(CAST(bmi.value AS REAL), 0.0) as latest_bmi,
    COALESCE(CAST(chol.value AS REAL), 0.0) as latest_cholesterol,
    COALESCE(enc_count.total_encounters, 0) as total_encounters,
    COALESCE(enc_count.recent_encounters, 0) as encounters_last_year,
    COALESCE(med_count.med_count, 0) as active_medications,
    COALESCE(cond_count.cond_count, 0) as active_conditions,
    COALESCE(CAST(
        (julianday('now') - julianday(last_enc.last_enc_date))
    AS INTEGER), 999) as days_since_last_encounter
FROM patients_patient p
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '4548-4'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id
             AND o.date = latest.max_date
    WHERE o.code = '4548-4'
) hba1c ON p.patient_id = hba1c.patient_id
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '8480-6'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id
             AND o.date = latest.max_date
    WHERE o.code = '8480-6'
) sbp ON p.patient_id = sbp.patient_id
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '8462-4'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id
             AND o.date = latest.max_date
    WHERE o.code = '8462-4'
) dbp ON p.patient_id = dbp.patient_id
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '39156-5'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id
             AND o.date = latest.max_date
    WHERE o.code = '39156-5'
) bmi ON p.patient_id = bmi.patient_id
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '2093-3'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id
             AND o.date = latest.max_date
    WHERE o.code = '2093-3'
) chol ON p.patient_id = chol.patient_id
LEFT JOIN (
    SELECT DISTINCT patient_id FROM patients_condition
    WHERE code IN ('44054006', '73211009') AND stop IS NULL
) d ON p.patient_id = d.patient_id
LEFT JOIN (
    SELECT DISTINCT patient_id FROM patients_condition
    WHERE code IN ('59621000', '38341003') AND stop IS NULL
) h ON p.patient_id = h.patient_id
LEFT JOIN (
    SELECT patient_id,
        COUNT(*) as total_encounters,
        SUM(CASE WHEN start >= date('now', '-365 days')
            THEN 1 ELSE 0 END) as recent_encounters
    FROM patients_encounter GROUP BY patient_id
) enc_count ON p.patient_id = enc_count.patient_id
LEFT JOIN (
    SELECT patient_id, COUNT(*) as med_count
    FROM patients_medication WHERE stop IS NULL
    GROUP BY patient_id
) med_count ON p.patient_id = med_count.patient_id
LEFT JOIN (
    SELECT patient_id, COUNT(*) as cond_count
    FROM patients_condition WHERE stop IS NULL
    GROUP BY patient_id
) cond_count ON p.patient_id = cond_count.patient_id
LEFT JOIN (
    SELECT patient_id, MAX(start) as last_enc_date
    FROM patients_encounter GROUP BY patient_id
) last_enc ON p.patient_id = last_enc.patient_id
WHERE p.cohort = 'chronic'
"""

cursor.execute(query)
rows = cursor.fetchall()
conn.close()

print(f"Loaded {len(rows):,} chronic patients")

# ── Build feature matrix ──────────────────────────────────────
import pandas as pd

df = pd.DataFrame(rows, columns=[
    'patient_id', 'age', 'gender',
    'has_diabetes', 'has_hypertension',
    'latest_hba1c', 'latest_sbp', 'latest_dbp',
    'latest_bmi', 'latest_cholesterol',
    'total_encounters', 'encounters_last_year',
    'active_medications', 'active_conditions',
    'days_since_last_encounter'
])

# Engineer same features as ml_models.py
for col in ['latest_hba1c', 'latest_sbp', 'latest_dbp',
            'latest_bmi', 'latest_cholesterol']:
    df[col] = df[col].replace(0, np.nan)

df['missing_hba1c'] = df['latest_hba1c'].isna().astype(int)

for col in ['latest_hba1c', 'latest_sbp', 'latest_dbp',
            'latest_bmi', 'latest_cholesterol']:
    df[col] = df[col].fillna(df[col].median())

df['gender_m']   = (df['gender'] == 'M').astype(int)
df['age_group']  = pd.cut(df['age'],
                           bins=[0,35,50,65,120],
                           labels=[1,2,3,4]).astype(int)
df['is_comorbid'] = ((df['has_diabetes']==1) &
                     (df['has_hypertension']==1)).astype(int)
df['low_engagement'] = (df['encounters_last_year']==0).astype(int)
df['undertreated']   = ((df['active_medications']<=1) &
                        (df['has_hypertension']==1)).astype(int)
df['high_condition_burden'] = (df['active_conditions']>=10).astype(int)
df['days_since_last_encounter'] = df[
    'days_since_last_encounter'].clip(upper=999)

FEATURES = [
    'latest_hba1c', 'latest_sbp', 'latest_dbp',
    'latest_bmi', 'latest_cholesterol',
    'age', 'gender_m', 'age_group',
    'has_diabetes', 'has_hypertension', 'is_comorbid',
    'total_encounters', 'encounters_last_year',
    'active_medications', 'active_conditions',
    'low_engagement', 'undertreated',
    'high_condition_burden', 'missing_hba1c',
]

X = df[FEATURES].fillna(0).values

# ── Create labels for both approaches ────────────────────────

# Label 1: Local (rule engine)
def local_label(row):
    sbp   = row['latest_sbp']
    hba1c = row['latest_hba1c']
    if sbp >= 160:   return 1
    if hba1c >= 9.0: return 1
    return 0

# Label 2: Colab (outcome based)
def colab_label(row):
    sbp   = row['latest_sbp']
    hba1c = row['latest_hba1c']
    meds  = row['active_medications']
    encs  = row['encounters_last_year']
    days  = row['days_since_last_encounter']
    age   = row['age']
    diab  = row['has_diabetes']
    htn   = row['has_hypertension']
    if sbp >= 135 and meds <= 1:                return 1
    if hba1c >= 7.0 and encs == 0:              return 1
    if days > 270 and (diab==1 or htn==1):      return 1
    if sbp >= 130 and encs == 0:                return 1
    if age < 50 and sbp >= 140:                 return 1
    if age < 50 and hba1c >= 7.5:               return 1
    if sbp >= 128 and days > 200 and encs <= 1: return 1
    if hba1c >= 6.8 and days > 200 and encs<=1: return 1
    return 0

y_local = df.apply(local_label, axis=1).values
y_colab = df.apply(colab_label, axis=1).values

print(f"\nLocal labels:  HIGH={y_local.sum():,} ({y_local.mean()*100:.1f}%)")
print(f"Colab labels:  HIGH={y_colab.sum():,} ({y_colab.mean()*100:.1f}%)")

# ── Load and evaluate all models ─────────────────────────────
model_files = {
    'Local Lasso':   'models/lasso_logistic_regression.pkl',
    'Local RF':      'models/random_forest.pkl',
    'Local GB':      'models/xgboost.pkl',
    'Colab Lasso':   'models/lasso_colab.pkl',
    'Colab RF':      'models/random_forest_colab.pkl',
    'Colab GB':      'models/xgboost_colab.pkl',
}

print(f"\n{'='*65}")
print(f"{'Model':<20} {'Label':<8} {'Acc':>6} {'Prec':>6} "
      f"{'Rec':>6} {'F1':>6} {'AUC':>6}")
print(f"{'-'*65}")

for model_name, model_path in model_files.items():
    if not os.path.exists(model_path):
        print(f"  {model_name:<18}: FILE NOT FOUND — skip")
        continue

    try:
        model = joblib.load(model_path)

        # Try both label sets
        for label_name, y in [('Local', y_local),
                               ('Colab', y_colab)]:
            try:
                y_prob = model.predict_proba(X)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                acc  = accuracy_score(y, y_pred)
                prec = precision_score(y, y_pred, zero_division=0)
                rec  = recall_score(y, y_pred, zero_division=0)
                f1   = f1_score(y, y_pred, zero_division=0)
                auc  = roc_auc_score(y, y_prob)

                print(f"  {model_name:<18} {label_name:<8} "
                      f"{acc:>6.3f} {prec:>6.3f} "
                      f"{rec:>6.3f} {f1:>6.3f} {auc:>6.3f}")
            except Exception as e:
                print(f"  {model_name:<18} {label_name:<8} ERROR: {e}")

    except Exception as e:
        print(f"  {model_name:<18}: LOAD ERROR — {e}")

print(f"\n{'='*65}")
print("RECOMMENDATION:")
print("  Best recall      → use for triage (catch most high risk)")
print("  Best AUC         → use for research paper")
print("  Best precision   → use for resource forecasting")