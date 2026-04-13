import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'caregap.settings')
django.setup()

import csv
import sqlite3

print("Direct SQLite export starting...")

# Connect directly to SQLite - no Django ORM
conn = sqlite3.connect(
    r'C:\Users\malli\OneDrive\Desktop\MRP NEW\caregap-main\db.sqlite3'
)
cursor = conn.cursor()

# One single SQL query - no Python loops needed
query = """
SELECT 
    p.patient_id,
    CAST(
        (strftime('%Y', 'now') - strftime('%Y', p.birthdate)) -
        (strftime('%m-%d', 'now') < strftime('%m-%d', p.birthdate))
    AS INTEGER) as age,
    CASE WHEN d.patient_id IS NOT NULL THEN 1 ELSE 0 END as has_diabetes,
    CASE WHEN h.patient_id IS NOT NULL THEN 1 ELSE 0 END as has_hypertension,
    COALESCE(CAST(hba1c.value AS REAL), 0.0) as latest_hba1c,
    COALESCE(CAST(sbp.value AS REAL), 0.0) as latest_sbp,
    CASE WHEN 
        COALESCE(CAST(sbp.value AS REAL), 0) >= 160 OR 
        COALESCE(CAST(hba1c.value AS REAL), 0) >= 9.0 
    THEN 1 ELSE 0 END as label
FROM patients_patient p
-- Latest HbA1c
LEFT JOIN (
    SELECT o.patient_id, o.value
    FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation
        WHERE code = '4548-4'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id 
             AND o.date = latest.max_date
    WHERE o.code = '4548-4'
) hba1c ON p.patient_id = hba1c.patient_id
-- Latest SBP
LEFT JOIN (
    SELECT o.patient_id, o.value
    FROM patients_observation o
    INNER JOIN (
        SELECT patient_id, MAX(date) as max_date
        FROM patients_observation
        WHERE code = '8480-6'
        GROUP BY patient_id
    ) latest ON o.patient_id = latest.patient_id 
             AND o.date = latest.max_date
    WHERE o.code = '8480-6'
) sbp ON p.patient_id = sbp.patient_id
-- Has diabetes
LEFT JOIN (
    SELECT DISTINCT patient_id
    FROM patients_condition
    WHERE code IN ('44054006', '73211009')
    AND stop IS NULL
) d ON p.patient_id = d.patient_id
-- Has hypertension
LEFT JOIN (
    SELECT DISTINCT patient_id
    FROM patients_condition
    WHERE code IN ('59621000', '38341003')
    AND stop IS NULL
) h ON p.patient_id = h.patient_id
WHERE p.cohort = 'chronic'
"""

print("Running query...")
cursor.execute(query)
rows = cursor.fetchall()
print(f"Query done! Got {len(rows)} patients")

with open('caregap_training_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'patient_id', 'age', 'has_diabetes', 'has_hypertension',
        'latest_hba1c', 'latest_sbp', 'label'
    ])
    writer.writerows(rows)

conn.close()
print(f"Done! Exported {len(rows)} patients")
print("File saved: caregap_training_data.csv")