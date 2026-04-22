from datetime import date, timedelta

from django.test import TestCase

from patients.models import Patient, UrgentCare
from patients.urgent_care_matcher import find_urgent_cares, normalize_insurance


class UrgentCareMatcherTests(TestCase):
    def setUp(self):
        self.patient = Patient.objects.create(
            patient_id="patient-urgent-001",
            first="Grace",
            last="Hopper",
            birthdate=date.today() - timedelta(days=67 * 365),
            gender="F",
            race="White",
            ethnicity="Not Hispanic or Latino",
            city="Los Angeles",
            state="CA",
            zip_code="90001",
            insurance="Medicare Advantage",
            lat=34.05,
            lon=-118.25,
        )

    def test_normalize_insurance_maps_known_values(self):
        self.assertEqual(normalize_insurance("Medicare Advantage"), "medicare")
        self.assertEqual(normalize_insurance("Blue Cross Blue Shield PPO"), "private")
        self.assertEqual(normalize_insurance("Self Pay"), "uninsured")
        self.assertEqual(normalize_insurance(""), "unknown")

    def test_find_urgent_cares_prefers_matching_insurance_and_nearest_distance(self):
        closest = UrgentCare.objects.create(
            name="Nearby Medicare Clinic",
            city="Los Angeles",
            state="CA",
            address="100 Main St",
            lat=34.051,
            lon=-118.251,
            rating=4.0,
            accepts_medicare=True,
            accepts_private=False,
        )
        farther = UrgentCare.objects.create(
            name="Farther Medicare Clinic",
            city="Los Angeles",
            state="CA",
            address="200 Main St",
            lat=34.10,
            lon=-118.30,
            rating=5.0,
            accepts_medicare=True,
            accepts_private=False,
        )
        UrgentCare.objects.create(
            name="Private Only Clinic",
            city="Los Angeles",
            state="CA",
            address="300 Main St",
            lat=34.049,
            lon=-118.249,
            rating=5.0,
            accepts_medicare=False,
            accepts_private=True,
        )

        facilities = find_urgent_cares(self.patient, max_results=5)

        self.assertEqual(facilities[0]["name"], closest.name)
        self.assertEqual(facilities[1]["name"], farther.name)
        self.assertTrue(all(item["accepts"]["medicare"] for item in facilities))
        self.assertTrue(all(item["insurance_accepted"] == "medicare" for item in facilities))

    def test_find_urgent_cares_falls_back_to_all_facilities_when_no_insurance_match(self):
        UrgentCare.objects.create(
            name="Private Clinic",
            city="Los Angeles",
            state="CA",
            address="111 Main St",
            lat=34.06,
            lon=-118.26,
            rating=4.1,
            accepts_private=True,
            accepts_medicare=False,
        )
        uninsured_patient = Patient.objects.create(
            patient_id="patient-urgent-002",
            first="Alan",
            last="Turing",
            birthdate=date.today() - timedelta(days=50 * 365),
            gender="M",
            race="White",
            ethnicity="Not Hispanic or Latino",
            city="Los Angeles",
            state="CA",
            zip_code="90002",
            insurance="No insurance",
            lat=34.06,
            lon=-118.26,
        )

        facilities = find_urgent_cares(uninsured_patient, max_results=5)

        self.assertEqual(len(facilities), 1)
        self.assertEqual(facilities[0]["name"], "Private Clinic")
