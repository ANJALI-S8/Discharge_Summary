

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI

app = FastAPI(title="Apex Discharge Summary Backend")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LLM =================
client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="EMPTY"
)

def call_ai(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("LLM Error:", e)
        return "AI generation failed"


# ================= DB =================
def get_conn():
    return psycopg2.connect(
        dbname="discharge_ai",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432,
        cursor_factory=RealDictCursor
    )


# ================= HELPER: Format JSON fields nicely =================
def format_field(field):
    if isinstance(field, dict):
        return " | ".join(f"{k}: {v}" for k, v in field.items())
    return field if field else "N/A"

def format_vitals(vitals):
    if isinstance(vitals, dict):
        mapping = {
            "temperature": "Temp", "heart_rate": "HR", "blood_pressure": "BP",
            "respiratory_rate": "RR", "spo2": "SpO2", "gcs": "GCS"
        }
        items = [f"{mapping.get(k, k)}: {v}" for k, v in vitals.items()]
        return " | ".join(items)
    return str(vitals) if vitals else "N/A"


# ================= AI GENERATION =================
def generate_ai(discharge_id: int):
    # (your existing generate_ai function - unchanged)
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            p.name, p.age, p.gender,
            ds.chief_complaint, ds.admission_date, ds.discharge_date, ds.length_of_stay,
            ds.mode_of_admission, ds.discharge_type,
            ds.vital_signs_admission, ds.systemic_examination,
            ds.functional_status,
            array_agg(DISTINCT d.diagnosis_text || ' (' || COALESCE(d.icd10_code,'') || ')') FILTER (WHERE d.diagnosis_text IS NOT NULL) AS diagnoses,
            array_agg(DISTINCT l.test_name || ': ' || l.admission_value || ' → ' || l.discharge_value) FILTER (WHERE l.test_name IS NOT NULL) AS labs,
            array_agg(DISTINCT m.drug_name || ' ' || m.dose || ' ' || m.route || ' ' || m.frequency) FILTER (WHERE m.drug_name IS NOT NULL) AS meds
        FROM discharge_summaries ds
        JOIN patients p ON ds.patient_id = p.patient_id
        LEFT JOIN diagnoses d ON ds.discharge_id = d.discharge_id
        LEFT JOIN lab_results l ON ds.discharge_id = l.discharge_id
        LEFT JOIN medications_discharge m ON ds.discharge_id = m.discharge_id
        WHERE ds.discharge_id = %s
        GROUP BY p.name, p.age, p.gender, ds.chief_complaint, ds.admission_date, 
                 ds.discharge_date, ds.length_of_stay, ds.mode_of_admission, 
                 ds.discharge_type, ds.vital_signs_admission, ds.systemic_examination,
                 ds.functional_status
    """, (discharge_id,))

    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return None

    context = f"""
Patient: {row['name']}, {row['age']} years, {row['gender']}
Chief Complaint: {row['chief_complaint'] or 'Not recorded'}
Admission Date: {row['admission_date']}
Discharge Date: {row['discharge_date']}
Length of Stay: {row['length_of_stay']} days
Mode: {row['mode_of_admission']} | Type: {row['discharge_type']}
Diagnoses: {", ".join(row['diagnoses'] or ['None recorded'])}
Labs: {", ".join(row['labs'] or ['None recorded'])}
Medications: {", ".join(row['meds'] or ['None recorded'])}
Vitals: {row['vital_signs_admission']}
Systemic Exam: {row['systemic_examination']}
Functional Status: {row['functional_status']}
"""

    prompt_hpi = f"""{context}\n\nWrite History of Present Illness.\nRULES:\n- Use ONLY given data\n- Do NOT assume anything\n- No headings\n- Paragraph format\n\nHPI:"""
    prompt_course = f"""{context}\n\nWrite Summary of Hospital Course.\nRULES:\n- Chronological flow\n- Use ONLY given data\n- No headings\n- No placeholders\n\nSummary:"""
    prompt_restrictions = f"""{context}\n\nWrite discharge instructions.\nRULES:\n- Diagnosis-based only\n- No generic advice\n- Bullet points\n\nRestrictions:"""

    hpi = call_ai(prompt_hpi)
    course = call_ai(prompt_course)
    restrictions = call_ai(prompt_restrictions)

    cur.close()
    conn.close()

    return {"hpi": hpi, "course": course, "restrictions": restrictions} if hpi and course and restrictions else None


# ================= GET PATIENTS =================
@app.get("/api/patients")
def get_patients():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT ds.discharge_id, p.name
        FROM discharge_summaries ds
        JOIN patients p ON ds.patient_id = p.patient_id
        ORDER BY ds.discharge_id
        LIMIT 100
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"discharge_id": r["discharge_id"], "name": r["name"]} for r in rows]


# ================= GET DISCHARGE (FIXED) =================
@app.get("/api/discharge/{discharge_id}")
def get_discharge(discharge_id: int):
    conn = get_conn()
    cur = conn.cursor()

    # AI generation check
    cur.execute("""
        SELECT history_of_present_illness, summary_of_hospital_course, activity_restrictions
        FROM discharge_summaries WHERE discharge_id = %s
    """, (discharge_id,))
    chk = cur.fetchone()

    ai_data = None
    if chk and (not chk["history_of_present_illness"] or not chk["summary_of_hospital_course"] or not chk["activity_restrictions"]):
        ai_data = generate_ai(discharge_id)

    # Main patient data
    cur.execute("""
        SELECT p.*, ds.*
        FROM discharge_summaries ds
        JOIN patients p ON ds.patient_id = p.patient_id
        WHERE ds.discharge_id = %s
    """, (discharge_id,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return JSONResponse({"error": "Not found"}, status_code=404)

    cur.execute("SELECT * FROM diagnoses WHERE discharge_id=%s", (discharge_id,))
    diagnoses = cur.fetchall()

    cur.execute("SELECT * FROM lab_results WHERE discharge_id=%s", (discharge_id,))
    labs = cur.fetchall()

    cur.execute("SELECT * FROM medications_discharge WHERE discharge_id=%s", (discharge_id,))
    meds = cur.fetchall()

    cur.close()
    conn.close()

    return {
        "document_type": "PATIENT DISCHARGE SUMMARY",
        "sections": {
            "section_1": {"fields": {
                "patient_name": row["name"],
                "mrn_hospital_id": row.get("mrn", "N/A"),
                "age": row.get("age", "N/A"),
                "gender": row.get("gender", "N/A"),
                "admission_date_time": str(row["admission_date"]) if row.get("admission_date") else "N/A",
                "discharge_date_time": str(row["discharge_date"]) if row.get("discharge_date") else "N/A",
                "length_of_stay": row.get("length_of_stay", "N/A"),          # ← raw number (UI will add " days")
                "admitting_physician": row.get("admitting_physician", "N/A"),
                "discharging_physician": row.get("discharging_physician", "N/A")
            }},

            "section_2": {"fields": {
                "primary_diagnosis": diagnoses[0]["diagnosis_text"] if diagnoses else "N/A",
                "primary_diagnosis_icd10_code": diagnoses[0]["icd10_code"] if diagnoses else "N/A"
            }},

            "section_3": {"fields": {
                "chief_complaint": row.get("chief_complaint", "N/A"),
                "history_of_present_illness": ai_data["hpi"] if ai_data else (row.get("history_of_present_illness") or "N/A")
            }},

            "section_4": {"fields": {
                "vital_signs": format_vitals(row.get("vital_signs_admission")),
                "systemic_examination": format_field(row.get("systemic_examination"))
            }},

            "section_5": {"laboratory_investigations": labs},

            "section_6": {"fields": {
                "summary_of_hospital_course": ai_data["course"] if ai_data else (row.get("summary_of_hospital_course") or "N/A")
            }},

            "section_7": {"fields": {"medications_on_discharge": meds}},

            "section_8": {"fields": {
                "functional_status": format_field(row.get("functional_status"))
            }},

            "section_9": {"fields": {
                "activity_dietary_restrictions": ai_data["restrictions"] if ai_data else (row.get("activity_restrictions") or "N/A")
            }},

            "section_10": {"fields": row.get("infection_control", {}) or {}},
            "section_11": {"fields": row.get("quality_indicators", {}) or {}},
            "section_12": {"fields": row.get("signatures", {}) or {}},
            "section_13": {"fields": row.get("administrative", {}) or {}}
        }
    }


# ================= SAVE =================
@app.post("/api/save-summary/{discharge_id}")
def save_summary(discharge_id: int, payload: dict = Body(...)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE discharge_summaries
        SET history_of_present_illness = %s,
            summary_of_hospital_course = %s,
            activity_restrictions = %s
        WHERE discharge_id = %s
    """, (payload.get("hpi"), payload.get("course"), payload.get("restrictions"), discharge_id))
    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Saved successfully"}