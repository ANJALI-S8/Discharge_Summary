CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE patients (
    patient_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    mrn VARCHAR(50) UNIQUE NOT NULL,
    dob DATE NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(20) NOT NULL,
    aadhaar VARCHAR(20),
    address TEXT,
    emergency_contact_name VARCHAR(100),
    emergency_contact_phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE discharge_summaries (
    discharge_id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(patient_id) ON DELETE CASCADE,
    admission_date TIMESTAMP NOT NULL,
    discharge_date TIMESTAMP NOT NULL,
    length_of_stay INT NOT NULL,
    admitting_physician VARCHAR(100),
    discharging_physician VARCHAR(100),
    mode_of_admission VARCHAR(50),
    discharge_type VARCHAR(50),
    chief_complaint TEXT,
    history_of_present_illness TEXT,
    tobacco_use VARCHAR(20),
    alcohol_use VARCHAR(20),
    substance_use VARCHAR(50),
    occupation_exposure TEXT,
    vital_signs_admission JSONB,
    anthropometric JSONB,
    systemic_examination JSONB,
    general_condition_discharge VARCHAR(50),
    vital_signs_discharge JSONB,
    wound_drain_status VARCHAR(100),
    functional_status JSONB,
    patient_education JSONB,
    activity_restrictions TEXT,
    tests_pending TEXT,
    infection_control JSONB,
    quality_indicators JSONB,
    advance_directives JSONB,
    signatures JSONB,
    administrative JSONB,
    summary_of_hospital_course TEXT, -- ← ONLY this field is NULL initially
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Child tables for repeating sections
CREATE TABLE diagnoses (
    diag_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    diagnosis_type VARCHAR(20) NOT NULL,
    diagnosis_text TEXT NOT NULL,
    icd10_code VARCHAR(20),
    code_type VARCHAR(30)
);

CREATE TABLE lab_results (
    lab_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    test_name VARCHAR(100) NOT NULL,
    admission_value VARCHAR(50),
    discharge_value VARCHAR(50),
    reference_range VARCHAR(100),
    interpretation VARCHAR(50)
);

CREATE TABLE microbiology (
    micro_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    specimen_type VARCHAR(100),
    date_collected DATE,
    organism TEXT,
    sensitivity TEXT,
    action_taken TEXT
);

CREATE TABLE imaging (
    imaging_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    study_type VARCHAR(100),
    date DATE,
    key_findings TEXT,
    report_ref TEXT
);

CREATE TABLE consultations (
    consult_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    specialty VARCHAR(100),
    consultant VARCHAR(100),
    date DATE,
    recommendation TEXT
);

CREATE TABLE procedures_performed (
    proc_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    procedure_name TEXT,
    date DATE,
    surgeon VARCHAR(100),
    anaesthesia VARCHAR(20)
);

CREATE TABLE medications_discharge (
    med_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    drug_name VARCHAR(100),
    dose VARCHAR(50),
    route VARCHAR(30),
    frequency VARCHAR(50),
    duration VARCHAR(50),
    special_instructions TEXT
);

CREATE TABLE allergies (
    allergy_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    allergen TEXT,
    reaction TEXT,
    severity VARCHAR(20)
);

CREATE TABLE follow_up_appointments (
    fu_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    department VARCHAR(100),
    physician VARCHAR(100),
    date_time TIMESTAMP,
    purpose TEXT
);

CREATE TABLE referrals (
    ref_id SERIAL PRIMARY KEY,
    discharge_id INT REFERENCES discharge_summaries(discharge_id) ON DELETE CASCADE,
    service VARCHAR(100),
    contact VARCHAR(100),
    reason TEXT
);
