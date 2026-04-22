from datetime import date, timedelta

from django.core.cache import cache
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from patients.models import (
    Condition,
    Encounter,
    Medication,
    Observation,
    Organization,
    Patient,
    UrgentCare,
)


class Command(BaseCommand):
    help = "Seed a tiny demo dataset for free/demo deployments."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete existing seeded records before inserting demo data.",
        )

    def handle(self, *args, **options):
        if Patient.objects.exists() and not options["reset"]:
            self.stdout.write("Patients already exist; skipping demo seed.")
            return

        if options["reset"]:
            self.stdout.write("Resetting demo data...")
            Medication.objects.all().delete()
            Observation.objects.all().delete()
            Encounter.objects.all().delete()
            Condition.objects.all().delete()
            Patient.objects.all().delete()
            Organization.objects.all().delete()
            UrgentCare.objects.all().delete()

        now = timezone.now()

        with transaction.atomic():
            orgs = [
                Organization(
                    org_id="org-demo-001",
                    name="CareGap Community Clinic",
                    address="101 Health Way",
                    city="St. Louis",
                    state="MO",
                    zip_code="63110",
                    lat=38.636,
                    lon=-90.261,
                    phone="314-555-0101",
                ),
                Organization(
                    org_id="org-demo-002",
                    name="CareGap Family Health Center",
                    address="250 Wellness Ave",
                    city="St. Louis",
                    state="MO",
                    zip_code="63108",
                    lat=38.644,
                    lon=-90.251,
                    phone="314-555-0110",
                ),
            ]
            Organization.objects.bulk_create(orgs, ignore_conflicts=True)

            urgent_cares = [
                UrgentCare(
                    name="Northside Urgent Care",
                    city="St. Louis",
                    state="MO",
                    address="800 Olive St",
                    phone="314-555-0201",
                    lat=38.628,
                    lon=-90.199,
                    accepts_medicaid=True,
                    accepts_medicare=True,
                    accepts_private=True,
                    accepts_uninsured=True,
                    rating=4.4,
                    open_24h=False,
                ),
                UrgentCare(
                    name="Gateway Walk-In Clinic",
                    city="St. Louis",
                    state="MO",
                    address="455 Forest Park Ave",
                    phone="314-555-0202",
                    lat=38.637,
                    lon=-90.236,
                    accepts_medicaid=True,
                    accepts_medicare=True,
                    accepts_private=True,
                    accepts_uninsured=False,
                    rating=4.7,
                    open_24h=True,
                ),
            ]
            UrgentCare.objects.bulk_create(urgent_cares, ignore_conflicts=True)

            patients = [
                Patient(
                    patient_id="demo-chr-001",
                    first="Maria",
                    last="Gomez",
                    birthdate=date(1972, 5, 14),
                    gender="F",
                    race="White",
                    ethnicity="Hispanic",
                    city="St. Louis",
                    state="MO",
                    zip_code="63110",
                    insurance="Medicaid",
                    lat=38.631,
                    lon=-90.241,
                    is_deceased=False,
                    cohort="chronic",
                ),
                Patient(
                    patient_id="demo-chr-002",
                    first="James",
                    last="Carter",
                    birthdate=date(1961, 9, 2),
                    gender="M",
                    race="Black",
                    ethnicity="Non-Hispanic",
                    city="St. Louis",
                    state="MO",
                    zip_code="63108",
                    insurance="Medicare",
                    lat=38.641,
                    lon=-90.253,
                    is_deceased=False,
                    cohort="chronic",
                ),
                Patient(
                    patient_id="demo-chr-003",
                    first="Samuel",
                    last="Lee",
                    birthdate=date(1980, 1, 25),
                    gender="M",
                    race="Asian",
                    ethnicity="Non-Hispanic",
                    city="St. Louis",
                    state="MO",
                    zip_code="63103",
                    insurance="Private",
                    lat=38.627,
                    lon=-90.197,
                    is_deceased=False,
                    cohort="chronic",
                ),
                Patient(
                    patient_id="demo-risk-001",
                    first="Alicia",
                    last="Nguyen",
                    birthdate=date(1989, 11, 9),
                    gender="F",
                    race="Asian",
                    ethnicity="Non-Hispanic",
                    city="St. Louis",
                    state="MO",
                    zip_code="63104",
                    insurance="Private",
                    lat=38.615,
                    lon=-90.214,
                    is_deceased=False,
                    cohort="at_risk",
                ),
                Patient(
                    patient_id="demo-risk-002",
                    first="Robert",
                    last="Patel",
                    birthdate=date(1978, 3, 30),
                    gender="M",
                    race="Asian",
                    ethnicity="Non-Hispanic",
                    city="St. Louis",
                    state="MO",
                    zip_code="63116",
                    insurance="No Insurance",
                    lat=38.598,
                    lon=-90.258,
                    is_deceased=False,
                    cohort="at_risk",
                ),
                Patient(
                    patient_id="demo-risk-003",
                    first="Elena",
                    last="Rivera",
                    birthdate=date(1994, 7, 11),
                    gender="F",
                    race="White",
                    ethnicity="Hispanic",
                    city="St. Louis",
                    state="MO",
                    zip_code="63118",
                    insurance="Private",
                    lat=38.612,
                    lon=-90.221,
                    is_deceased=False,
                    cohort="at_risk",
                ),
                Patient(
                    patient_id="demo-ped-001",
                    first="Mia",
                    last="Johnson",
                    birthdate=date(2013, 8, 19),
                    gender="F",
                    race="Black",
                    ethnicity="Non-Hispanic",
                    city="St. Louis",
                    state="MO",
                    zip_code="63112",
                    insurance="Medicaid",
                    lat=38.658,
                    lon=-90.284,
                    is_deceased=False,
                    cohort="pediatric",
                ),
                Patient(
                    patient_id="demo-dec-001",
                    first="Helen",
                    last="Brooks",
                    birthdate=date(1950, 2, 6),
                    gender="F",
                    race="White",
                    ethnicity="Non-Hispanic",
                    city="St. Louis",
                    state="MO",
                    zip_code="63109",
                    insurance="Medicare",
                    lat=38.582,
                    lon=-90.291,
                    is_deceased=True,
                    cohort="deceased",
                ),
            ]
            Patient.objects.bulk_create(patients, ignore_conflicts=True)

            patient_map = {patient.patient_id: patient for patient in Patient.objects.all()}

            conditions = [
                Condition(
                    patient=patient_map["demo-chr-001"],
                    start=date(2021, 1, 10),
                    stop=None,
                    code=Condition.DIABETES_CODES[0],
                    description="Type 2 diabetes mellitus",
                ),
                Condition(
                    patient=patient_map["demo-chr-001"],
                    start=date(2020, 6, 15),
                    stop=None,
                    code=Condition.HYPERTENSION_CODES[0],
                    description="Essential hypertension",
                ),
                Condition(
                    patient=patient_map["demo-chr-002"],
                    start=date(2019, 4, 3),
                    stop=None,
                    code=Condition.HYPERTENSION_CODES[0],
                    description="Essential hypertension",
                ),
                Condition(
                    patient=patient_map["demo-chr-003"],
                    start=date(2022, 8, 9),
                    stop=None,
                    code=Condition.DIABETES_CODES[0],
                    description="Type 2 diabetes mellitus",
                ),
                Condition(
                    patient=patient_map["demo-dec-001"],
                    start=date(2018, 1, 1),
                    stop=date(2024, 5, 1),
                    code=Condition.HYPERTENSION_CODES[0],
                    description="Essential hypertension",
                ),
            ]
            Condition.objects.bulk_create(conditions, ignore_conflicts=True)

            observations = [
                self._obs(patient_map["demo-chr-001"], now - timedelta(days=12), Observation.LOINC_HBA1C, "HbA1c", "8.4", "%"),
                self._obs(patient_map["demo-chr-001"], now - timedelta(days=12), Observation.LOINC_SBP, "Systolic blood pressure", "152", "mmHg"),
                self._obs(patient_map["demo-chr-001"], now - timedelta(days=12), "8462-4", "Diastolic blood pressure", "94", "mmHg"),
                self._obs(patient_map["demo-chr-001"], now - timedelta(days=12), "39156-5", "Body mass index", "33.2", "kg/m2"),
                self._obs(patient_map["demo-chr-001"], now - timedelta(days=12), "2093-3", "Cholesterol", "221", "mg/dL"),
                self._obs(patient_map["demo-chr-002"], now - timedelta(days=40), Observation.LOINC_HBA1C, "HbA1c", "5.6", "%"),
                self._obs(patient_map["demo-chr-002"], now - timedelta(days=40), Observation.LOINC_SBP, "Systolic blood pressure", "166", "mmHg"),
                self._obs(patient_map["demo-chr-002"], now - timedelta(days=40), "8462-4", "Diastolic blood pressure", "98", "mmHg"),
                self._obs(patient_map["demo-chr-002"], now - timedelta(days=40), "39156-5", "Body mass index", "29.1", "kg/m2"),
                self._obs(patient_map["demo-chr-003"], now - timedelta(days=180), Observation.LOINC_HBA1C, "HbA1c", "10.2", "%"),
                self._obs(patient_map["demo-chr-003"], now - timedelta(days=180), Observation.LOINC_SBP, "Systolic blood pressure", "142", "mmHg"),
                self._obs(patient_map["demo-chr-003"], now - timedelta(days=180), "8462-4", "Diastolic blood pressure", "88", "mmHg"),
                self._obs(patient_map["demo-chr-003"], now - timedelta(days=180), "39156-5", "Body mass index", "31.8", "kg/m2"),
                self._obs(patient_map["demo-risk-001"], now - timedelta(days=20), Observation.LOINC_HBA1C, "HbA1c", "6.1", "%"),
                self._obs(patient_map["demo-risk-001"], now - timedelta(days=20), Observation.LOINC_SBP, "Systolic blood pressure", "134", "mmHg"),
                self._obs(patient_map["demo-risk-001"], now - timedelta(days=20), "8462-4", "Diastolic blood pressure", "84", "mmHg"),
                self._obs(patient_map["demo-risk-001"], now - timedelta(days=20), "39156-5", "Body mass index", "28.7", "kg/m2"),
                self._obs(patient_map["demo-risk-001"], now - timedelta(days=20), "2093-3", "Cholesterol", "204", "mg/dL"),
                self._obs(patient_map["demo-risk-002"], now - timedelta(days=420), Observation.LOINC_HBA1C, "HbA1c", "5.8", "%"),
                self._obs(patient_map["demo-risk-002"], now - timedelta(days=420), Observation.LOINC_SBP, "Systolic blood pressure", "128", "mmHg"),
                self._obs(patient_map["demo-risk-002"], now - timedelta(days=420), "8462-4", "Diastolic blood pressure", "82", "mmHg"),
                self._obs(patient_map["demo-risk-002"], now - timedelta(days=420), "39156-5", "Body mass index", "31.0", "kg/m2"),
                self._obs(patient_map["demo-risk-003"], now - timedelta(days=7), Observation.LOINC_HBA1C, "HbA1c", "5.4", "%"),
                self._obs(patient_map["demo-risk-003"], now - timedelta(days=7), Observation.LOINC_SBP, "Systolic blood pressure", "118", "mmHg"),
                self._obs(patient_map["demo-risk-003"], now - timedelta(days=7), "8462-4", "Diastolic blood pressure", "76", "mmHg"),
                self._obs(patient_map["demo-risk-003"], now - timedelta(days=7), "39156-5", "Body mass index", "24.6", "kg/m2"),
                self._obs(patient_map["demo-ped-001"], now - timedelta(days=14), "39156-5", "Body mass index", "27.4", "kg/m2"),
                self._obs(patient_map["demo-ped-001"], now - timedelta(days=14), Observation.LOINC_SBP, "Systolic blood pressure", "112", "mmHg"),
                self._obs(patient_map["demo-dec-001"], now - timedelta(days=500), Observation.LOINC_HBA1C, "HbA1c", "7.2", "%"),
                self._obs(patient_map["demo-dec-001"], now - timedelta(days=500), Observation.LOINC_SBP, "Systolic blood pressure", "148", "mmHg"),
            ]
            Observation.objects.bulk_create(observations, ignore_conflicts=True)

            encounters = [
                self._enc(patient_map["demo-chr-001"], "enc-demo-001", now - timedelta(days=18), "ambulatory", "Chronic care follow-up"),
                self._enc(patient_map["demo-chr-002"], "enc-demo-002", now - timedelta(days=120), "ambulatory", "Blood pressure check"),
                self._enc(patient_map["demo-chr-003"], "enc-demo-003", now - timedelta(days=210), "outpatient", "Diabetes management visit"),
                self._enc(patient_map["demo-risk-001"], "enc-demo-004", now - timedelta(days=30), "wellness", "Annual wellness visit"),
                self._enc(patient_map["demo-risk-002"], "enc-demo-005", now - timedelta(days=420), "wellness", "Preventive visit"),
                self._enc(patient_map["demo-risk-003"], "enc-demo-006", now - timedelta(days=10), "ambulatory", "Routine follow-up"),
                self._enc(patient_map["demo-ped-001"], "enc-demo-007", now - timedelta(days=15), "wellness", "Pediatric wellness check"),
                self._enc(patient_map["demo-dec-001"], "enc-demo-008", now - timedelta(days=530), "inpatient", "Prior hospitalization"),
            ]
            Encounter.objects.bulk_create(encounters, ignore_conflicts=True)

            medications = [
                Medication(
                    patient=patient_map["demo-chr-001"],
                    start=date(2021, 1, 10),
                    stop=None,
                    code="860975",
                    description="Metformin 500 MG",
                    reason_code=Condition.DIABETES_CODES[0],
                    reason_description="Type 2 diabetes mellitus",
                ),
                Medication(
                    patient=patient_map["demo-chr-001"],
                    start=date(2020, 6, 15),
                    stop=None,
                    code="316049",
                    description="Losartan 50 MG",
                    reason_code=Condition.HYPERTENSION_CODES[0],
                    reason_description="Essential hypertension",
                ),
                Medication(
                    patient=patient_map["demo-chr-002"],
                    start=date(2019, 4, 3),
                    stop=None,
                    code="197361",
                    description="Amlodipine 10 MG",
                    reason_code=Condition.HYPERTENSION_CODES[0],
                    reason_description="Essential hypertension",
                ),
            ]
            Medication.objects.bulk_create(medications, ignore_conflicts=True)

        cache.clear()
        self.stdout.write(self.style.SUCCESS("Demo seed data created."))

    @staticmethod
    def _obs(patient, when, code, description, value, units):
        return Observation(
            patient=patient,
            date=when,
            code=code,
            description=description,
            value=value,
            units=units,
        )

    @staticmethod
    def _enc(patient, encounter_id, start, encounter_class, description):
        return Encounter(
            patient=patient,
            encounter_id=encounter_id,
            start=start,
            stop=start + timedelta(hours=1),
            encounter_class=encounter_class,
            description=description,
        )
