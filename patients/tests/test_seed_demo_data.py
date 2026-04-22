from django.core.management import call_command
from django.test import TestCase

from patients.models import Condition, Encounter, Observation, Organization, Patient, UrgentCare


class SeedDemoDataTests(TestCase):
    def test_seed_demo_data_creates_core_demo_records(self):
        call_command("seed_demo_data")

        self.assertEqual(Patient.objects.count(), 8)
        self.assertGreaterEqual(Observation.objects.count(), 20)
        self.assertGreaterEqual(Encounter.objects.count(), 6)
        self.assertGreaterEqual(Condition.objects.count(), 4)
        self.assertGreaterEqual(Organization.objects.count(), 2)
        self.assertGreaterEqual(UrgentCare.objects.count(), 2)
        self.assertTrue(Patient.objects.filter(cohort="chronic").exists())
        self.assertTrue(Patient.objects.filter(cohort="at_risk").exists())
        self.assertTrue(Patient.objects.filter(cohort="pediatric").exists())
        self.assertTrue(Patient.objects.filter(cohort="deceased").exists())
