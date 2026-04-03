# Discharge Summary API Reference

**Project:** Apex Discharge Summary Backend  
**Version:** 1.0  
**Last Updated:** April 2026

## Base URL
http://localhost:5001
text*(Your backend is currently running on port 5001)*

## Available Endpoints

### 1. GET /api/patients
**Description:** Returns list of recent patients for selection screen.

**Response Example:**
```json
[
  {
    "discharge_id": 1,
    "name": "Priya Sharma"
  },
  {
    "discharge_id": 2,
    "name": "Rahul Verma"
  }
]
2. GET /api/discharge/{discharge_id}
Description: Returns complete structured discharge summary.
Note: If AI fields (HPI, Hospital Course, Restrictions) are empty, the backend automatically generates them.
Response Structure:
JSON{
  "document_type": "PATIENT DISCHARGE SUMMARY",
  "sections": {
    "section_1": {
      "fields": {
        "patient_name": "Priya Sharma",
        "mrn_hospital_id": "MRN-12345",
        "age": 45,
        "gender": "Female",
        "admission_date_time": "2026-03-15T10:30:00",
        "discharge_date_time": "2026-03-22T09:15:00",
        "length_of_stay": 7,
        "admitting_physician": "Dr. Suresh",
        "discharging_physician": "Dr. Radha"
      }
    },
    "section_2": {
      "fields": {
        "primary_diagnosis": "Cataract",
        "primary_diagnosis_icd10_code": "H26.9"
      }
    },
    "section_3": {
      "fields": {
        "chief_complaint": "Eye pain, Redness, Blurred vision",
        "history_of_present_illness": "AI generated paragraph here..."
      }
    },
    "section_4": {
      "fields": {
        "vital_signs": "Temp: 98.6 | HR: 82 | BP: 120/80 | RR: 16 | SpO2: 98 | GCS: 15",
        "systemic_examination": "Cardiovascular: Normal | Respiratory: Clear | ..."
      }
    },
    "section_5": {
      "laboratory_investigations": [
        {
          "test_name": "Hemoglobin",
          "admission_value": "14",
          "discharge_value": "13",
          "reference_range": "12-16",
          "interpretation": "Normal"
        }
      ]
    },
    "section_6": {
      "fields": {
        "summary_of_hospital_course": "AI generated summary here..."
      }
    },
    "section_7": {
      "fields": {
        "medications_on_discharge": [
          {
            "drug_name": "Moxifloxacin",
            "dose": "0.5%",
            "route": "Eye",
            "frequency": "QID",
            "duration": "7 days",
            "special_instructions": "Use regularly"
          }
        ]
      }
    },
    "section_8": {
      "fields": {
        "functional_status": "Mobility Status: Ambulatory | Diet: Oral | ..."
      }
    },
    "section_9": {
      "fields": {
        "activity_dietary_restrictions": "AI generated discharge instructions here..."
      }
    },
    "section_10": { "fields": {} },
    "section_11": { "fields": {} },
    "section_12": { "fields": {} },
    "section_13": { "fields": {} }
  }
}
3. POST /api/save-summary/{discharge_id}
Description: Save edited AI-generated fields back to the database.
Request Body:
JSON{
  "hpi": "Updated History of Present Illness...",
  "course": "Updated Summary of Hospital Course...",
  "restrictions": "Updated activity and dietary restrictions..."
}
Success Response:
JSON{
  "message": "Saved successfully"
}