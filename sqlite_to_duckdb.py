"""
Converts db_demo.sqlite3 → synthea_california.duckdb
Run once: python sqlite_to_duckdb.py
"""
import sqlite3
import duckdb
import pandas as pd
import os

SQLITE_PATH = 'db_demo.sqlite3'
DUCKDB_PATH = 'synthea_california.duckdb'

TABLES = [
    'patients_patient',
    'patients_observation',
    'patients_condition',
    'patients_encounter',
    'patients_medication',
]

# DuckDB column name mapping to match what duckdb_client.py expects
COLUMN_MAP = {
    'patients_patient': {
        'patient_id': 'Id',
        'first': 'FIRST',
        'last': 'LAST',
        'birthdate': 'BIRTHDATE',
        'gender': 'GENDER',
        'race': 'RACE',
        'ethnicity': 'ETHNICITY',
        'city': 'CITY',
        'state': 'STATE',
        'zip_code': 'ZIP',
        'lat': 'LAT',
        'lon': 'LON',
        'is_deceased': 'DEATHDATE',
        'cohort': 'COHORT',
    },
    'patients_observation': {
        'patient_id': 'PATIENT',
        'date': 'DATE',
        'code': 'CODE',
        'description': 'DESCRIPTION',
        'value': 'VALUE',
        'units': 'UNITS',
    },
    'patients_condition': {
        'patient_id': 'PATIENT',
        'start': 'START',
        'stop': 'STOP',
        'code': 'CODE',
        'description': 'DESCRIPTION',
    },
    'patients_encounter': {
        'patient_id': 'PATIENT',
        'encounter_id': 'Id',
        'start': 'START',
        'stop': 'STOP',
        'encounter_class': 'ENCOUNTERCLASS',
        'description': 'DESCRIPTION',
    },
    'patients_medication': {
        'patient_id': 'PATIENT',
        'start': 'START',
        'stop': 'STOP',
        'code': 'CODE',
        'description': 'DESCRIPTION',
    },
}

# Final DuckDB table names duckdb_client.py queries
TABLE_NAME_MAP = {
    'patients_patient':     'patients',
    'patients_observation': 'observations',
    'patients_condition':   'conditions',
    'patients_encounter':   'encounters',
    'patients_medication':  'medications',
}

if os.path.exists(DUCKDB_PATH):
    os.remove(DUCKDB_PATH)
    print(f'Removed existing {DUCKDB_PATH}')

sqlite_conn = sqlite3.connect(SQLITE_PATH)
duck_conn = duckdb.connect(DUCKDB_PATH)

for sqlite_table in TABLES:
    print(f'Converting {sqlite_table}...', end=' ', flush=True)
    df = pd.read_sql_query(f'SELECT * FROM {sqlite_table}', sqlite_conn)

    # Drop Django's auto-increment PK to avoid case-collision with mapped 'Id' columns
    df = df.drop(columns=['id'], errors='ignore')

    col_map = COLUMN_MAP.get(sqlite_table, {})
    df = df.rename(columns=col_map)

    # For patients table: set DEATHDATE to NULL for living patients
    if sqlite_table == 'patients_patient':
        df['DEATHDATE'] = df['DEATHDATE'].apply(lambda x: None if not x else x)

    duck_table = TABLE_NAME_MAP[sqlite_table]
    duck_conn.execute(f'DROP TABLE IF EXISTS {duck_table}')
    duck_conn.register('df_temp', df)
    duck_conn.execute(f'CREATE TABLE {duck_table} AS SELECT * FROM df_temp')
    count = duck_conn.execute(f'SELECT COUNT(*) FROM {duck_table}').fetchone()[0]
    print(f'done ({count:,} rows)')

sqlite_conn.close()
duck_conn.close()
print(f'\nDone. {DUCKDB_PATH} is ready.')
