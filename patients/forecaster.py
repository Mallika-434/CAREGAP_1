import math


def forecast_resources(high_risk_count):
    predicted_beds = math.ceil(high_risk_count * 0.15)
    predicted_icu = math.ceil(predicted_beds * 0.10)
    nurses_needed = math.ceil(predicted_beds / 4) + predicted_icu

    return {
        'high_risk_volume': high_risk_count,
        'period_days': 30,
        'resources': {
            'beds': {
                'count': predicted_beds,
                'label': 'General Hospital Beds',
                'description': 'Estimated 30-day capacity based on 15% admission rate',
            },
            'icu': {
                'count': predicted_icu,
                'label': 'ICU Rooms',
                'description': 'Estimated Critical Care needs (10% of admissions)',
            },
            'nurses': {
                'count': nurses_needed,
                'label': 'Nurse Staffing',
                'description': 'Staffing requirements (1:4 general, 1:1 ICU ratio)',
            },
        },
    }
