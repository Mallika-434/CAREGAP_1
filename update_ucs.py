import os
import django
import random

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'caregap.settings')
django.setup()

from patients.models import UrgentCare

# Mock some Massachusetts locations
ma_locations = [
    {"city": "Boston", "state": "MA", "lat": 42.3601, "lon": -71.0589},
    {"city": "Cambridge", "state": "MA", "lat": 42.3736, "lon": -71.1097},
    {"city": "Worcester", "state": "MA", "lat": 42.2626, "lon": -71.8023},
    {"city": "Springfield", "state": "MA", "lat": 42.1015, "lon": -72.5898},
    {"city": "Lowell", "state": "MA", "lat": 42.6334, "lon": -71.3162},
]

ucs = UrgentCare.objects.all()
for i, uc in enumerate(ucs):
    loc = ma_locations[i % len(ma_locations)]
    uc.city = loc["city"]
    uc.state = loc["state"]
    # Add some slight random jitter to lat/lon so they aren't all exactly on top of each other
    uc.lat = loc["lat"] + random.uniform(-0.02, 0.02)
    uc.lon = loc["lon"] + random.uniform(-0.02, 0.02)
    uc.save()
    print(f"Updated {uc.name} -> {uc.city}, {uc.state}")
