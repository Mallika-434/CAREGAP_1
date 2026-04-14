import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'caregap.settings')
django.setup()

import csv
import sqlite3

print("Exporting enhanced training data v2...")

db_path = os.environ.get(
    'DB_PATH',
    r'C:\Users\malli\OneDrive\Desktop\MRP NEW\caregap-main\db.sqlite3'
)

print(f"Using database: {db_path}")

conn = sqlite3.connect(db_path)
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
    AS INTEGER), 999) as days_since_last_encounter,
    CASE WHEN 
        COALESCE(CAST(sbp.value AS REAL), 0) >= 160 OR 
        COALESCE(CAST(hba1c.value AS REAL), 0) >= 9.0 
    THEN 1 ELSE 0 END as label
FROM patients_patient p
LEFT JOIN (
    SELECT o.patient_id, o.value FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation WHERE code = '4548-4'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id AND o.date = latest.max_date
    WHERE o.code = '4548-4'
) hba1c ON p.patient_id = hba1c.patient_id
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

print("Running query...")
cursor.execute(query)
rows = cursor.fetchall()
print(f"Got {len(rows)} patients")

with open('caregap_training_data_v2.csv', 'w',
          newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'patient_id', 'age', 'gender', 'ethnicity',
        'has_diabetes', 'has_hypertension',
        'latest_hba1c', 'latest_sbp', 'latest_dbp',
        'latest_bmi', 'latest_cholesterol',
        'total_encounters', 'encounters_last_year',
        'active_medications', 'active_conditions',
        'days_since_last_encounter', 'label'
    ])
    writer.writerows(rows)

conn.close()
print(f"Done! Saved caregap_training_data_v2.csv")