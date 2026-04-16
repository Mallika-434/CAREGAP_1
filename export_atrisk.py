import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'caregap.settings')
django.setup()

import csv
import sqlite3

print("Exporting at_risk patients for disease onset prediction...")

conn = sqlite3.connect(
    r'C:\Users\malli\OneDrive\Desktop\MRP NEW\caregap-main\db.sqlite3'
)
cursor = conn.cursor()

query = """
SELECT 
    p.patient_id,
    CAST(
        (strftime('%Y', 'now') - strftime('%Y', p.birthdate)) -
        (strftime('%m-%d', 'now') < strftime('%m-%d', p.birthdate))
    AS INTEGER) as age,
    p.gender,
    p.ethnicity,
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
    AS INTEGER), 999) as days_since_last_encounter,
    CASE WHEN CAST(sbp.value AS REAL) >= 130 THEN 1 ELSE 0 END as developed_htn,
    CASE WHEN CAST(bmi.value AS REAL) >= 30 THEN 1 ELSE 0 END as developed_diabetes
FROM patients_patient p
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '8480-6'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id AND o.date = latest.max_date
    WHERE o.code = '8480-6'
) sbp ON p.patient_id = sbp.patient_id
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '8462-4'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id AND o.date = latest.max_date
    WHERE o.code = '8462-4'
) dbp ON p.patient_id = dbp.patient_id
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '39156-5'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id AND o.date = latest.max_date
    WHERE o.code = '39156-5'
) bmi ON p.patient_id = bmi.patient_id
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '2093-3'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id AND o.date = latest.max_date
    WHERE o.code = '2093-3'
) chol ON p.patient_id = chol.patient_id
LEFT JOIN (
    SELECT patient_id,
        COUNT(*) as total_encounters,
        SUM(CASE WHEN start >= date('now', '-365 days') THEN 1 ELSE 0 END) as recent_encounters
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
WHERE p.cohort = 'at_risk'
"""

print("Running query...")
cursor.execute(query)
rows = cursor.fetchall()
print(f"Got {len(rows)} at_risk patients")

with open('caregap_atrisk_v2.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'patient_id', 'age', 'gender', 'ethnicity',
        'latest_sbp', 'latest_dbp', 'latest_bmi', 'latest_cholesterol',
        'total_encounters', 'encounters_last_year',
        'active_medications', 'active_conditions',
        'days_since_last_encounter',
        'developed_htn', 'developed_diabetes'
    ])
    writer.writerows(rows)

conn.close()

htn_pos = sum(1 for r in rows if r[13] == 1)
t2d_pos = sum(1 for r in rows if r[14] == 1)
total   = len(rows)

print(f"\nDone! Saved caregap_atrisk_data.csv")
print(f"Total at_risk patients: {total:,}")
print(f"\nLabel distribution:")
print(f"  HTN onset  (SBP>=130): {htn_pos:,} ({100*htn_pos/total:.1f}%) positive")
print(f"  T2D onset  (BMI>=30):  {t2d_pos:,} ({100*t2d_pos/total:.1f}%) positive")
print(f"\nUpload caregap_atrisk_data.csv to Google Colab and run Sections 13-19.")