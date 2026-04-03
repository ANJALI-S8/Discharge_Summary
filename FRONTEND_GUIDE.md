# Frontend Development Guide - Discharge Summary Application

## Project Overview
This backend provides structured data for a **Patient Discharge Summary** web application.

## Important Information

### Main Identifier
- Use **`discharge_id`** as the primary key for all API calls.

### Main API Endpoint
- **`GET /api/discharge/{discharge_id}`** — This is the most important endpoint.
- It returns data already organized into **section_1 to section_13** for easy UI rendering.
- AI content is automatically generated on first load if fields are missing.

### Editable Fields (Doctor can edit these)
- `history_of_present_illness` → Section 3
- `summary_of_hospital_course` → Section 6
- `activity_dietary_restrictions` → Section 9

After editing, call:
```http
POST /api/save-summary/{discharge_id}
with body: { "hpi": "...", "course": "...", "restrictions": "..." }
Data Formatting Tips

length_of_stay: Show as {value} days
Vitals and other JSONB fields are already formatted into readable strings in most sections.
Labs and Medications come as arrays → display them in tables.
Dates come in ISO format → format them nicely in UI.

Suggested UI Flow

Patient List Screen → Use GET /api/patients
Discharge Detail Screen → Use GET /api/discharge/{discharge_id}
Make Section 3, 6, and 9 editable (textarea or rich text)
Add Save button that calls the POST save endpoint

Database Reference
Refer to schema.sql for complete table structure and relationships.