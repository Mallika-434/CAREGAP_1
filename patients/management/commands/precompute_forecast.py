"""
Management command: precompute_forecast
────────────────────────────────────────
Pre-computes triage queue and resource forecast, stores results in:
  - Django cache  (triage_list key, TTL 300 s)
  - patients/data/triage_cache.json  + triage_cache.pkl
  - patients/data/forecast_cache.json + forecast_cache.pkl

The .pkl files are read by the resource_forecast view for fast,
cache-expiry-proof serving.

Usage:
    python manage.py precompute_forecast
"""

import json
import os
import pickle
import time

from django.core.management.base import BaseCommand


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')


class Command(BaseCommand):
    help = 'Pre-computes triage queue and resource forecast; saves to patients/data/.'

    def handle(self, *args, **options):
        from patients.triage_services import get_triage_payload, get_resource_forecast_payload

        os.makedirs(DATA_DIR, exist_ok=True)

        # ── Triage ────────────────────────────────────────────────────
        self.stdout.write('Computing triage queue…', ending=' ')
        self.stdout.flush()
        t0 = time.monotonic()
        triage = get_triage_payload()
        elapsed = time.monotonic() - t0

        emergency_count = len(triage.get('emergency_patients', []))
        urgent_count = len(triage.get('urgent_patients', []))
        self.stdout.write(self.style.SUCCESS(
            f'done ({elapsed:.1f}s) — {emergency_count} emergency, {urgent_count} urgent'
        ))

        triage_path = os.path.join(DATA_DIR, 'triage_cache.json')
        with open(triage_path, 'w') as f:
            json.dump(triage, f, default=str, indent=2)
        self.stdout.write(f'  Saved -> {triage_path}')

        triage_pkl = os.path.join(DATA_DIR, 'triage_cache.pkl')
        with open(triage_pkl, 'wb') as f:
            pickle.dump(triage, f)
        self.stdout.write(f'  Saved -> {triage_pkl}')

        # ── Forecast ──────────────────────────────────────────────────
        self.stdout.write('Computing resource forecast…', ending=' ')
        self.stdout.flush()
        t0 = time.monotonic()
        forecast = get_resource_forecast_payload()
        elapsed = time.monotonic() - t0

        hvol = forecast.get('high_risk_volume', 0)
        res = forecast.get('resources', {})
        beds = res.get('beds', {}).get('count', 0)
        icu = res.get('icu', {}).get('count', 0)
        nurses = res.get('nurses', {}).get('count', 0)
        self.stdout.write(self.style.SUCCESS(
            f'done ({elapsed:.1f}s) — high_risk_volume={hvol}, '
            f'beds={beds}, icu={icu}, nurses={nurses}'
        ))

        forecast_path = os.path.join(DATA_DIR, 'forecast_cache.json')
        with open(forecast_path, 'w') as f:
            json.dump(forecast, f, indent=2)
        self.stdout.write(f'  Saved -> {forecast_path}')

        forecast_pkl = os.path.join(DATA_DIR, 'forecast_cache.pkl')
        with open(forecast_pkl, 'wb') as f:
            pickle.dump(forecast, f)
        self.stdout.write(f'  Saved -> {forecast_pkl}')

        self.stdout.write(self.style.SUCCESS(
            '\nForecast pre-computed. Dashboard Action Required section will show live numbers.'
        ))
