from datetime import date, timedelta

from django.test import TestCase
from django.utils import timezone

from patients.models import Condition, Observation, Patient
from patients.risk_engine import assess_risk


class RiskEngineTests(TestCase):
    def _make_patient(self, **overrides):
        defaults = {
            "patient_id": "patient-001",
            "first": "Ada",
            "last": "Lovelace",
            "birthdate": date.today() - timedelta(days=55 * 365),
            "gender": "F",
            "race": "White",
            "ethnicity": "Not Hispanic or Latino",
            "city": "Los Angeles",
            "state": "CA",
            "zip_code": "90001",
            "insurance": "Medicare",
        }
        defaults.update(overrides)
        return Patient.objects.create(**defaults)

    def _make_observation(self, patient, code, value, days_ago=0, description="Observation"):
        return Observation.objects.create(
            patient=patient,
            code=code,
            value=str(value),
            description=description,
            units="",
            date=timezone.now() - timedelta(days=days_ago),
        )

    def _make_condition(self, patient, code, description, stop=None):
        return Condition.objects.create(
            patient=patient,
            code=code,
            description=description,
            start=date.today() - timedelta(days=365),
            stop=stop,
        )

    def test_assess_risk_returns_emergency_for_critical_vitals(self):
        patient = self._make_patient()
        self._make_condition(patient, Condition.DIABETES_CODES[0], "Type 2 diabetes mellitus")
        self._make_condition(patient, Condition.HYPERTENSION_CODES[0], "Hypertension")
        self._make_observation(patient, Observation.LOINC_HBA1C, 9.4, days_ago=20, description="HbA1c")
        self._make_observation(patient, Observation.LOINC_SBP, 165, days_ago=2, description="Systolic BP")

        result = assess_risk(
            patient,
            patient.observations.all(),
            patient.conditions.all(),
        )

        self.assertEqual(result.tier, "EMERGENCY")
        self.assertEqual(result.followup_urgency_days, 0)
        self.assertTrue(result.has_diabetes)
        self.assertTrue(result.has_hypertension)
        self.assertGreaterEqual(result.score, 80)
        self.assertEqual(result.hba1c_value, 9.4)
        self.assertEqual(result.latest_sbp, 165.0)

    def test_assess_risk_returns_normal_for_stable_patient(self):
        patient = self._make_patient(
            patient_id="patient-002",
            birthdate=date.today() - timedelta(days=30 * 365),
        )
        self._make_observation(patient, Observation.LOINC_HBA1C, 5.5, days_ago=45, description="HbA1c")
        self._make_observation(patient, Observation.LOINC_SBP, 118, days_ago=10, description="Systolic BP")

        result = assess_risk(
            patient,
            patient.observations.all(),
            patient.conditions.all(),
        )

        self.assertEqual(result.tier, "NORMAL")
        self.assertEqual(result.followup_urgency_days, 180)
        self.assertFalse(result.has_diabetes)
        self.assertFalse(result.has_hypertension)
        self.assertLess(result.score, 10)

    def test_assess_risk_flags_missing_hba1c_for_diabetic_patient(self):
        patient = self._make_patient(patient_id="patient-003")
        self._make_condition(patient, Condition.DIABETES_CODES[0], "Type 2 diabetes mellitus")

        result = assess_risk(
            patient,
            patient.observations.all(),
            patient.conditions.all(),
        )

        self.assertIn(result.tier, {"MODERATE", "HIGH"})
        self.assertIsNone(result.hba1c_days_gap)
        self.assertIsNone(result.hba1c_value)
        self.assertTrue(any("no HbA1c test on record" in reason for reason in result.reasons))
