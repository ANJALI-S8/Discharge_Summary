
import random
from faker import Faker
from datetime import timedelta, datetime
import psycopg2
from psycopg2.extras import execute_values, Json

fake = Faker('en_IN')

NUM_PATIENTS = 10000
BATCH_SIZE = 1000

conn = psycopg2.connect(
    dbname="discharge_ai",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432
)
cur = conn.cursor()

print("Cleaning old data...")
cur.execute("""
TRUNCATE patients, discharge_summaries, diagnoses,
lab_results, medications_discharge, follow_up_appointments
RESTART IDENTITY CASCADE;
""")

PHYSICIANS = ["Dr. Suresh", "Dr. Radha", "Dr. Amit", "Dr. Meera"]

for batch in range(0, NUM_PATIENTS, BATCH_SIZE):

    patients = []
    discharges = []
    diagnoses = []
    labs = []
    meds = []
    followups = []

    for i in range(BATCH_SIZE):

        pid = batch + i + 1

        age = random.randint(20,80)
        gender = random.choice(["Male","Female"])

        # ---------------- PATIENT ----------------
        patients.append((
            pid,
            fake.name(),
            f"MRN-{pid}",
            fake.date_of_birth(minimum_age=age, maximum_age=age),
            age,
            gender,
            fake.aadhaar_id(),
            fake.address(),
            fake.name(),
            fake.phone_number()
        ))

        # ---------------- DATES ----------------
        adm = datetime.now() - timedelta(days=random.randint(5,30))
        dis = adm + timedelta(days=random.randint(1,7))

        # ---------------- JSON ----------------
        vital_signs = Json({
            "temperature": str(round(random.uniform(97,101),1)),
            "heart_rate": str(random.randint(60,100)),
            "blood_pressure": f"{random.randint(100,140)}/{random.randint(60,90)}",
            "respiratory_rate": str(random.randint(12,20)),
            "spo2": str(random.randint(95,100)),
            "gcs": "15"
        })

        anthropometric = Json({
            "height_cm": random.randint(150,180),
            "weight_kg": random.randint(50,90),
            "bmi": round(random.uniform(18,30),1),
            "bsa": round(random.uniform(1.5,2.2),2)
        })

        systemic_exam = Json({
            "cardiovascular": "Normal",
            "respiratory": "Clear",
            "gastrointestinal": "Soft",
            "neurological": "Normal",
            "musculoskeletal": "Normal",
            "genitourinary": "Normal",
            "skin_lymph_nodes": "Normal"
        })

        functional_status = Json({
            "mobility_status": "Ambulatory",
            "diet_nutrition": "Oral",
            "continence": "Continent",
            "cognition_mmse": "30"
        })

        infection_control = Json({
            "cauti": {"status": "No"},
            "clabsi": {"status": "No"},
            "vap": {"status": "No"},
            "ssi": {"status": "No"}
        })

        quality_indicators = Json({
            "vte_prophylaxis": {"complied": "Yes"},
            "fall_risk_assessment": {"complied": "Yes"},
            "pressure_ulcer": {"complied": "Yes"}
        })

        signatures = Json({
            "doctor": random.choice(PHYSICIANS),
            "patient": fake.name(),
            "nurse": fake.name()
        })

        administrative = Json({
            "ehr_updated": "Yes",
            "claim_submitted": "Yes",
            "copy_given": "Yes"
        })

        restrictions = None
        hospital_course = None
        hpi = None

        # ---------------- DISCHARGE ----------------
        discharges.append((
            pid, pid,
            adm, dis, (dis-adm).days,
            random.choice(PHYSICIANS),
            random.choice(PHYSICIANS),
            random.choice(["Emergency","OPD"]),
            random.choice(["Routine","LAMA"]),
            random.choice(["Eye pain","Redness","Blurred vision"]),
            hpi,
            "No","No","No","IT job",
            vital_signs,
            anthropometric,
            systemic_exam,
            "Stable",
            Json({"bp":"120/80","hr":"80","spo2":"98"}),
            "Healed",
            functional_status,
            Json({"education":"Given"}),
            restrictions,
            "None",
            infection_control,
            quality_indicators,
            Json({"advance_directive":"No"}),
            signatures,
            administrative,
            hospital_course
        ))

        # ---------------- DIAG ----------------
        diagnoses.append((pid,"Primary","Cataract","H26.9","ICD10"))

        # ---------------- LAB ----------------
        labs.append((pid,"Hemoglobin","14","13","12-16","Normal"))

        # ---------------- MED ----------------
        meds.append((pid,"Moxifloxacin","0.5%","Eye","QID","7 days","Use regularly"))

        # ---------------- FOLLOWUP ----------------
        followups.append((pid,"Ophthalmology",random.choice(PHYSICIANS),dis+timedelta(days=7),"Review"))

    # ================= INSERTS (FIXED ONLY HERE) =================

    execute_values(cur, "INSERT INTO patients VALUES %s", patients)

    execute_values(cur, "INSERT INTO discharge_summaries VALUES %s", discharges)

    execute_values(cur, """
    INSERT INTO diagnoses 
    (discharge_id, diagnosis_type, diagnosis_text, icd10_code, code_type)
    VALUES %s
    """, diagnoses)

    execute_values(cur, """
    INSERT INTO lab_results 
    (discharge_id, test_name, admission_value, discharge_value, reference_range, interpretation)
    VALUES %s
    """, labs)

    execute_values(cur, """
    INSERT INTO medications_discharge
    (discharge_id, drug_name, dose, route, frequency, duration, special_instructions)
    VALUES %s
    """, meds)

    execute_values(cur, """
    INSERT INTO follow_up_appointments
    (discharge_id, department, physician, date_time, purpose)
    VALUES %s
    """, followups)

    conn.commit()
    print(f"Inserted {batch+BATCH_SIZE}")

cur.close()
conn.close()

print("✅ DONE — FULL DATA INSERTED")