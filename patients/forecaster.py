import math


def forecast_resources(risk_breakdown):
    """
    Calculates 30-day hospital resource requirements based on
    risk tier breakdown (emergency, high, moderate, elevated).

    Hospitalization rates:
      - Emergency: 90% (extreme vitals, immediate acuity)
      - High:      50% (unstable chronic)
      - Moderate:  25% (rising risk)
      - Elevated:  10% (early warning, Stage 2 HTN)

    ICU rate: 15% of total admissions
    Nurse ratios: 1:4 general, 1:1 ICU
    """
    emergency = risk_breakdown.get('emergency', 0)
    high      = risk_breakdown.get('high', 0)
    moderate  = risk_breakdown.get('moderate', 0)
    elevated  = risk_breakdown.get('elevated', 0)

    h_emergency = int(emergency * 0.90)
    h_high      = int(high      * 0.50)
    h_moderate  = int(moderate  * 0.25)
    h_elevated  = int(elevated  * 0.10)

    total_beds = h_emergency + h_high + h_moderate + h_elevated
    icu_beds   = int(total_beds * 0.15)
    nurses     = math.ceil((total_beds - icu_beds) / 4) + icu_beds

    return {
        'risk_breakdown': risk_breakdown,
        'period_days': 30,
        'resources': {
            'beds': {
                'count': max(1, total_beds),
                'label': 'General Hospital Beds',
                'description': 'Stabilization for patients with HbA1c > 8.0 or SBP > 140.'
            },
            'icu': {
                'count': max(0, icu_beds),
                'label': 'ICU Rooms',
                'description': 'Reserved for critical care cohort with ensemble risk score above 75%.'
            },
            'nurses': {
                'count': max(1, nurses),
                'label': 'Nurse Staffing',
                'description': '24/7 staffing at 4:1 general and 1:1 critical care ratio.'
            }
        }
    }
