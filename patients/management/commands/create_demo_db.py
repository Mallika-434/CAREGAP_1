"""
Management command: create_demo_db
───────────────────────────────────
Creates db_demo.sqlite3 containing 500 randomly-sampled chronic patients
plus all their related records and the full urgent-care / organisations
lookup data.  Designed to produce a <10 MB file suitable for HF Spaces.

Usage:
    python manage.py create_demo_db
    python manage.py create_demo_db --patients 200
"""

import os
import sqlite3
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Create a small demo SQLite database with N chronic patients.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--patients', type=int, default=500,
            help='Number of chronic patients to include (default: 500)',
        )

    def handle(self, *args, **options):
        n = options['patients']
        src_path = str(settings.BASE_DIR / 'db.sqlite3')
        dst_path = str(settings.BASE_DIR / 'db_demo.sqlite3')

        if not os.path.exists(src_path):
            self.stdout.write(self.style.ERROR(f'Source not found: {src_path}'))
            return

        if os.path.exists(dst_path):
            os.remove(dst_path)
            self.stdout.write(f'Removed existing {dst_path}')

        self.stdout.write(f'Creating demo DB with {n} chronic patients…')

        src = sqlite3.connect(src_path)
        dst = sqlite3.connect(dst_path)
        src.row_factory = sqlite3.Row

        # ── 1. Copy schema (tables + indexes) ────────────────────────
        for row in src.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE sql IS NOT NULL ORDER BY type DESC"
        ):
            try:
                dst.execute(row[0])
            except sqlite3.OperationalError:
                pass   # table/index already exists
        dst.commit()
        self.stdout.write('  Schema copied.')

        # ── 2. Sample N chronic patients ─────────────────────────────
        patient_ids = [
            r[0] for r in src.execute(
                "SELECT patient_id FROM patients_patient "
                "WHERE cohort = 'chronic' AND is_deceased = 0 "
                "ORDER BY RANDOM() LIMIT ?",
                (n,),
            ).fetchall()
        ]
        self.stdout.write(f'  Sampled {len(patient_ids)} patient IDs.')

        phs = ','.join(['?'] * len(patient_ids))

        def copy_fk_table(table, fk='patient_id'):
            rows = src.execute(
                f"SELECT * FROM {table} WHERE {fk} IN ({phs})",
                patient_ids,
            ).fetchall()
            if not rows:
                self.stdout.write(f'  {table}: 0 rows (skipped)')
                return
            ncols = len(rows[0])
            dst.executemany(
                f"INSERT OR IGNORE INTO {table} VALUES ({','.join(['?']*ncols)})",
                rows,
            )
            dst.commit()
            self.stdout.write(f'  {table}: {len(rows):,} rows')

        # ── 3. Copy patient-scoped tables ─────────────────────────────
        copy_fk_table('patients_patient')
        copy_fk_table('patients_observation')
        copy_fk_table('patients_condition')
        copy_fk_table('patients_medication')
        copy_fk_table('patients_encounter')

        # ── 4. Copy lookup tables (not patient-scoped) ────────────────
        for table in ('patients_urgentcare', 'patients_organization'):
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if rows:
                ncols = len(rows[0])
                dst.executemany(
                    f"INSERT OR IGNORE INTO {table} "
                    f"VALUES ({','.join(['?']*ncols)})",
                    rows,
                )
                dst.commit()
                self.stdout.write(f'  {table}: {len(rows):,} rows (full copy)')

        # ── 5. Copy django_migrations so manage.py migrate is a no-op ─
        rows = src.execute("SELECT * FROM django_migrations").fetchall()
        if rows:
            dst.executemany(
                "INSERT OR IGNORE INTO django_migrations VALUES (?,?,?,?)",
                rows,
            )
            dst.commit()
            self.stdout.write(f'  django_migrations: {len(rows)} rows')

        src.close()
        dst.close()

        size_mb = os.path.getsize(dst_path) / (1024 ** 2)
        self.stdout.write(self.style.SUCCESS(
            f'\ndb_demo.sqlite3 created — {size_mb:.1f} MB'
        ))
        self.stdout.write(
            'Next: update Dockerfile to use db_demo.sqlite3 for HF Spaces.'
        )
