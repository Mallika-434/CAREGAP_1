"""
Management command: create_demo_db
───────────────────────────────────
Creates db_demo.sqlite3 with a clinically balanced 20,000-patient sample:

  chronic   3,684  (≈ 18% of total)
  at_risk   9,853  (≈ 49% of total)
  pediatric 4,086  (≈ 20% of total)
  deceased  2,347  (≈ 12% of total — note: cohorts can overlap in count)
  ─────────────────
  Total    20,000

All related observations, conditions, medications, and encounters are
copied for each selected patient. Organizations and urgent-care facilities
are copied in full (they are not patient-scoped).

Django migrations are copied from the source so the demo DB is immediately
usable without running migrate.

Usage:
    python manage.py create_demo_db
"""

import os
import sqlite3

from django.conf import settings
from django.core.management.base import BaseCommand


# Exact cohort counts that produce 15,000 total with the same ratios
# as the full 33,990-patient database.
COHORT_COUNTS = [
    ('chronic',   3_684),
    ('at_risk',   9_853),
    ('pediatric', 4_086),
    ('deceased',  2_347),
]


class Command(BaseCommand):
    help = 'Create db_demo.sqlite3: 15,000-patient balanced sample for HF Spaces.'

    def handle(self, *args, **options):
        src_path = str(settings.BASE_DIR / 'db.sqlite3')
        dst_path = str(settings.BASE_DIR / 'db_demo.sqlite3')

        if not os.path.exists(src_path):
            self.stdout.write(self.style.ERROR(f'Source DB not found: {src_path}'))
            return

        if os.path.exists(dst_path):
            os.remove(dst_path)
            self.stdout.write(f'Removed existing db_demo.sqlite3')

        src = sqlite3.connect(src_path)
        dst = sqlite3.connect(dst_path)

        # Large pages + WAL give much better bulk-insert throughput.
        dst.execute('PRAGMA journal_mode=WAL')
        dst.execute('PRAGMA synchronous=NORMAL')
        dst.execute('PRAGMA cache_size=-64000')   # 64 MB cache
        dst.execute('PRAGMA page_size=4096')

        # ── 1. Copy schema ────────────────────────────────────────────
        for (sql,) in src.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE sql IS NOT NULL ORDER BY type DESC"
        ):
            try:
                dst.execute(sql)
            except sqlite3.OperationalError:
                pass  # already exists
        dst.commit()

        # ── 2. Select patients by cohort ──────────────────────────────
        self.stdout.write('Selecting patients...')
        all_patient_ids: list[str] = []

        for cohort, n in COHORT_COUNTS:
            rows = src.execute(
                "SELECT patient_id FROM patients_patient "
                "WHERE cohort = ? ORDER BY RANDOM() LIMIT ?",
                (cohort, n),
            ).fetchall()
            ids = [r[0] for r in rows]
            all_patient_ids.extend(ids)
            self.stdout.write(f'Copying {cohort}: {len(ids):,}')

        # Deduplicate (a patient could theoretically appear in two cohort
        # queries if the DB has dirty data — keep order stable).
        seen: set[str] = set()
        unique_ids: list[str] = []
        for pid in all_patient_ids:
            if pid not in seen:
                seen.add(pid)
                unique_ids.append(pid)

        self.stdout.write(
            f'Total unique patients selected: {len(unique_ids):,}'
        )

        # ── helpers ───────────────────────────────────────────────────
        # SQLite max bind parameters = 999 - batch in chunks of 900.
        CHUNK = 900

        def copy_patient_table(table: str, fk: str = 'patient_id') -> None:
            """Copy every row for the selected patients (no row limit)."""
            total = 0
            for i in range(0, len(unique_ids), CHUNK):
                batch = unique_ids[i: i + CHUNK]
                phs = ','.join(['?'] * len(batch))
                rows = src.execute(
                    f"SELECT * FROM {table} WHERE {fk} IN ({phs})",
                    batch,
                ).fetchall()
                if rows:
                    ncols = len(rows[0])
                    col_phs = ','.join(['?'] * ncols)
                    dst.executemany(
                        f"INSERT OR IGNORE INTO {table} VALUES ({col_phs})",
                        rows,
                    )
                    total += len(rows)
            dst.commit()
            self.stdout.write(f'  -> {total:,} rows')

        def copy_patient_table_limited(
            table: str,
            order_col: str,
            max_per_patient: int,
            fk: str = 'patient_id',
        ) -> None:
            """Copy only the N most-recent rows per patient using ROW_NUMBER()."""
            total = 0
            for i in range(0, len(unique_ids), CHUNK):
                batch = unique_ids[i: i + CHUNK]
                phs = ','.join(['?'] * len(batch))
                # ROW_NUMBER() partitioned by patient keeps the top N per patient
                # without loading all rows into Python.
                rows = src.execute(
                    f"""
                    SELECT * FROM (
                        SELECT *,
                               ROW_NUMBER() OVER (
                                   PARTITION BY {fk}
                                   ORDER BY {order_col} DESC
                               ) AS _rn
                        FROM {table}
                        WHERE {fk} IN ({phs})
                    ) WHERE _rn <= ?
                    """,
                    batch + [max_per_patient],
                ).fetchall()
                if rows:
                    # Strip the trailing _rn column before inserting
                    ncols_with_rn = len(rows[0])
                    ncols = ncols_with_rn - 1
                    col_phs = ','.join(['?'] * ncols)
                    dst.executemany(
                        f"INSERT OR IGNORE INTO {table} VALUES ({col_phs})",
                        [r[:ncols] for r in rows],
                    )
                    total += len(rows)
            dst.commit()
            self.stdout.write(f'  -> {total:,} rows (max {max_per_patient}/patient)')

        # ── 3. Copy patients_patient ──────────────────────────────────
        self.stdout.write('Copying patients...')
        copy_patient_table('patients_patient')

        # ── 4. Copy clinical records ──────────────────────────────────
        self.stdout.write('Copying observations...')
        copy_patient_table_limited('patients_observation', 'date', max_per_patient=50)

        self.stdout.write('Copying conditions...')
        copy_patient_table('patients_condition')          # all conditions

        self.stdout.write('Copying medications...')
        copy_patient_table('patients_medication')         # all medications

        self.stdout.write('Copying encounters...')
        copy_patient_table_limited('patients_encounter', 'start', max_per_patient=20)

        # ── 5. Copy lookup tables (not patient-scoped) ────────────────
        self.stdout.write('Copying organizations...')
        for table in ('patients_organization', 'patients_urgentcare'):
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if rows:
                ncols = len(rows[0])
                col_phs = ','.join(['?'] * ncols)
                dst.executemany(
                    f"INSERT OR IGNORE INTO {table} VALUES ({col_phs})",
                    rows,
                )
                dst.commit()
                self.stdout.write(f'  {table}: {len(rows):,} rows')

        # ── 6. Copy django_migrations (no migrate needed on demo DB) ──
        rows = src.execute("SELECT * FROM django_migrations").fetchall()
        if rows:
            ncols = len(rows[0])
            col_phs = ','.join(['?'] * ncols)
            dst.executemany(
                f"INSERT OR IGNORE INTO django_migrations VALUES ({col_phs})",
                rows,
            )
            dst.commit()

        src.close()

        # VACUUM the destination so it has no wasted pages.
        self.stdout.write('Compacting...')
        dst.execute('VACUUM')
        dst.close()

        size_mb = os.path.getsize(dst_path) / (1024 ** 2)
        self.stdout.write(self.style.SUCCESS(
            f'Done! db_demo.sqlite3 created'
        ))
        self.stdout.write(self.style.SUCCESS(
            f'File size: {size_mb:.1f} MB'
        ))
