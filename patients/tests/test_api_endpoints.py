from datetime import date, timedelta
from unittest.mock import patch

from django.core.cache import cache
from django.test import Client, TestCase
from django.utils import timezone

from patients.models import Condition, Encounter, Medication, Observation, Patient


class PatientApiEndpointTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = Client()

    def _make_patient(self, patient_id, first, cohort="chronic", is_deceased=False, age_years=50, gender="F"):
        return Patient.objects.create(
            patient_id=patient_id,
            first=first,
            last="Tester",
            birthdate=date.today() - timedelta(days=age_years * 365),
            gender=gender,
            race="White",
            ethnicity="Not Hispanic or Latino",
            city="Los Angeles",
            state="CA",
            zip_code="90001",
            insurance="Medicare",
            cohort=cohort,
            is_deceased=is_deceased,
            lat=34.05,
            lon=-118.25,
        )

    def _add_observation(self, patient, code, value, days_ago=0, description="Observation"):
        return Observation.objects.create(
            patient=patient,
            code=code,
            value=str(value),
            description=description,
            units="",
            date=timezone.now() - timedelta(days=days_ago),
        )

    def _add_condition(self, patient, code, description, stop=None):
        return Condition.objects.create(
            patient=patient,
            code=code,
            description=description,
            start=date.today() - timedelta(days=365),
            stop=stop,
        )

    def _add_encounter(self, patient, days_ago=0):
        start = timezone.now() - timedelta(days=days_ago)
        return Encounter.objects.create(
            patient=patient,
            encounter_id=f"enc-{patient.patient_id}-{days_ago}",
            start=start,
            stop=start + timedelta(hours=1),
            encounter_class="ambulatory",
            description="Follow-up visit",
        )

    def test_patient_search_excludes_deceased_by_default(self):
        alive = self._make_patient("alive-001", "Alice", cohort="chronic", is_deceased=False)
        self._make_patient("dead-001", "Alice", cohort="deceased", is_deceased=True)

        response = self.client.get("/api/patients/search/", {"q": "Alice"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["patient_id"], alive.patient_id)

    def test_patient_search_honors_explicit_deceased_cohort(self):
        self._make_patient("alive-002", "Bob", cohort="chronic", is_deceased=False)
        deceased = self._make_patient("dead-002", "Bob", cohort="deceased", is_deceased=True)

        response = self.client.get("/api/patients/search/", {"q": "Bob", "cohort": "deceased"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["patient_id"], deceased.patient_id)

    def test_patient_risk_returns_expected_payload(self):
        patient = self._make_patient("risk-001", "Carla", age_years=58)
        self._add_condition(patient, Condition.DIABETES_CODES[0], "Type 2 diabetes mellitus")
        self._add_observation(patient, Observation.LOINC_HBA1C, 8.2, days_ago=15, description="HbA1c")
        self._add_observation(patient, Observation.LOINC_SBP, 145, days_ago=3, description="Systolic BP")

        response = self.client.get(f"/api/patients/{patient.patient_id}/risk/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["patient_id"], patient.patient_id)
        self.assertEqual(payload["tier"], "HIGH")
        self.assertTrue(payload["has_diabetes"])
        self.assertFalse(payload["has_hypertension"])
        self.assertEqual(payload["hba1c_value"], 8.2)
        self.assertEqual(payload["latest_sbp"], 145.0)
        self.assertIn("recommended_action", payload)

    def test_patient_predict_rejects_pediatric_patients(self):
        patient = self._make_patient("ped-001", "Dina", cohort="pediatric", age_years=12)

        response = self.client.get(f"/api/patients/{patient.patient_id}/predict/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["error"], "not_applicable")
        self.assertEqual(payload["cohort"], "pediatric")

    def test_patient_predict_returns_fallback_prediction_for_chronic_patient_without_models(self):
        patient = self._make_patient("pred-001", "Evan", cohort="chronic", age_years=52, gender="M")
        self._add_condition(patient, Condition.HYPERTENSION_CODES[0], "Hypertension")
        self._add_observation(patient, Observation.LOINC_SBP, 162, days_ago=1, description="Systolic BP")
        self._add_observation(patient, Observation.LOINC_HBA1C, 6.1, days_ago=30, description="HbA1c")
        self._add_encounter(patient, days_ago=20)
        Medication.objects.create(
            patient=patient,
            start=date.today() - timedelta(days=90),
            stop=None,
            code="med-001",
            description="Lisinopril",
            reason_code="rx-001",
            reason_description="Hypertension",
        )

        with patch("patients.ml_models.load_risk_models", return_value={}):
            response = self.client.get(f"/api/patients/{patient.patient_id}/predict/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["patient_id"], patient.patient_id)
        self.assertFalse(payload["model_available"])
        self.assertEqual(payload["confidence"], "low")
        self.assertGreater(payload["progression_probability"], 0)
        self.assertIn("recommendation", payload)
        self.assertIn("features", payload)

    def test_analytics_filters_population_and_top_conditions(self):
        chronic = self._make_patient("ana-001", "Fiona", cohort="chronic", age_years=48, gender="F")
        self._make_patient("ana-002", "George", cohort="at_risk", age_years=35, gender="M")
        self._add_condition(chronic, Condition.HYPERTENSION_CODES[0], "Hypertension")
        self._add_observation(chronic, Observation.LOINC_SBP, 142, days_ago=5, description="Systolic BP")
        self._add_observation(chronic, Observation.LOINC_HBA1C, 5.8, days_ago=10, description="HbA1c")

        response = self.client.get(
            "/api/patients/analytics/",
            {"cohort": "chronic", "gender": "F", "condition": "hypertension"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["filters"]["cohort"], "chronic")
        self.assertEqual(payload["filters"]["gender"], "F")
        self.assertEqual(payload["filters"]["condition"], "hypertension")
        self.assertEqual(payload["bp_dist"]["stage2"], 1)
        self.assertTrue(payload["top_conditions"])
        self.assertEqual(payload["top_conditions"][0]["name"], "Hypertension")

    def test_dashboard_stats_basic_returns_population_counts(self):
        chronic = self._make_patient("stats-basic-001", "Lena", cohort="chronic", age_years=54)
        self._make_patient("stats-basic-002", "Milo", cohort="pediatric", age_years=12)
        self._make_patient("stats-basic-003", "Nora", cohort="deceased", is_deceased=True, age_years=70)
        self._add_condition(chronic, Condition.HYPERTENSION_CODES[0], "Hypertension")

        response = self.client.get("/api/patients/stats/basic/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total_active"], 2)
        self.assertEqual(payload["total_deceased"], 1)
        self.assertEqual(payload["cohort_counts"]["chronic"], 1)
        self.assertEqual(payload["cohort_counts"]["pediatric"], 1)
        self.assertEqual(payload["cohort_counts"]["deceased"], 1)
        self.assertGreaterEqual(payload["hypertension_rate"], 0)

    def test_dashboard_stats_returns_expected_sections(self):
        chronic = self._make_patient("stats-full-001", "Owen", cohort="chronic", age_years=59)
        self._add_condition(chronic, Condition.HYPERTENSION_CODES[0], "Hypertension")
        self._add_condition(chronic, Condition.DIABETES_CODES[0], "Type 2 diabetes mellitus")
        self._add_observation(chronic, Observation.LOINC_SBP, 152, days_ago=1, description="Systolic BP")
        self._add_observation(chronic, Observation.LOINC_HBA1C, 8.1, days_ago=2, description="HbA1c")

        response = self.client.get("/api/patients/stats/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("hba1c_dist", payload)
        self.assertIn("bp_dist", payload)
        self.assertIn("risk_overlap", payload)
        self.assertIn("care_gap_cascade", payload)
        self.assertIn("insurance_breakdown", payload)
        self.assertIn("cohort_counts", payload)
        self.assertEqual(payload["cohort_counts"]["chronic"], 1)

    def test_triage_list_separates_emergency_and_urgent_patients(self):
        emergency = self._make_patient("triage-001", "Hazel", cohort="chronic", age_years=62)
        urgent = self._make_patient("triage-002", "Iris", cohort="chronic", age_years=57)
        self._add_observation(emergency, Observation.LOINC_SBP, 165, days_ago=1, description="Systolic BP")
        self._add_observation(urgent, Observation.LOINC_SBP, 145, days_ago=2, description="Systolic BP")
        self._add_encounter(urgent, days_ago=200)

        response = self.client.get("/api/patients/triage/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        emergency_ids = {row["patient_id"] for row in payload["emergency_patients"]}
        urgent_ids = {row["patient_id"] for row in payload["urgent_patients"]}
        self.assertIn(emergency.patient_id, emergency_ids)
        self.assertIn(urgent.patient_id, urgent_ids)
        self.assertNotIn(emergency.patient_id, urgent_ids)

    def test_resource_forecast_uses_cached_triage_breakdown(self):
        cache.set(
            "triage_list",
            {
                "emergency_patients": [{"patient_id": "a"}, {"patient_id": "b"}],
                "urgent_patients": [{"patient_id": "c"}],
                "warning_patients": [{"patient_id": "d"}, {"patient_id": "e"}, {"patient_id": "f"}],
                "stable_patients": [{"patient_id": "g"}],
            },
            300,
        )

        response = self.client.get("/api/patients/resources/forecast/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["risk_breakdown"]["emergency"], 2)
        self.assertEqual(payload["risk_breakdown"]["high"], 1)
        self.assertEqual(payload["risk_breakdown"]["moderate"], 3)
        self.assertEqual(payload["risk_breakdown"]["elevated"], 1)
        self.assertIn("generated_at", payload)
        self.assertIn("resources", payload)

    @patch("patients.ml_models.predict_onset_risk")
    def test_onset_risk_endpoint_returns_mocked_scores_for_at_risk_patient(self, mock_predict):
        patient = self._make_patient("onset-001", "Jade", cohort="at_risk", age_years=44)
        mock_predict.return_value = {
            "available": True,
            "htn": {"ensemble": 61.2},
            "t2d": {"ensemble": 34.8},
        }

        response = self.client.get(f"/api/patients/{patient.patient_id}/onset-risk/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["patient_id"], patient.patient_id)
        self.assertEqual(payload["cohort"], "at_risk")
        self.assertEqual(payload["onset_risk"]["htn"]["ensemble"], 61.2)
        self.assertEqual(payload["onset_risk"]["t2d"]["ensemble"], 34.8)

    def test_bmi_assessment_returns_category_for_pediatric_patient(self):
        patient = self._make_patient("bmi-001", "Kai", cohort="pediatric", age_years=10, gender="M")
        self._add_observation(patient, "39156-5", 23.5, days_ago=1, description="BMI")

        response = self.client.get(f"/api/patients/{patient.patient_id}/bmi-assessment/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["patient_id"], patient.patient_id)
        self.assertEqual(payload["category"], "Obese")
        self.assertEqual(payload["color"], "red")
        self.assertIn("thresholds", payload)
