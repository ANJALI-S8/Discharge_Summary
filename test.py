# import pandas as pd
# import psycopg2

# conn = psycopg2.connect(
#     dbname="discharge_ai",
#     user="postgres",
#     password="postgres",
#     host="localhost",
#     port=5432
# )

# tables = [
#     "patients",
#     "discharge_summaries",
#     "diagnoses",
#     "lab_results",
#     "medications_discharge",
#     "follow_up_appointments"
# ]

# # Create Excel with multiple sheets
# with pd.ExcelWriter("hospital_full_data.xlsx", engine="openpyxl") as writer:
#     for table in tables:
#         df = pd.read_sql(f"SELECT * FROM {table}", conn)
#         df.to_excel(writer, sheet_name=table, index=False)
#         print(f"✅ Exported {table}")

# conn.close()

# print("\n🎉 Multi-sheet Excel created: hospital_full_data.xlsx")
import pandas as pd
import psycopg2

conn = psycopg2.connect(
    dbname="discharge_ai",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432
)

tables = [
    "patients",
    "discharge_summaries",
    "diagnoses",
    "lab_results",
    "medications_discharge",
    "follow_up_appointments"
]

with pd.ExcelWriter("hospital_full_data.xlsx", engine="openpyxl") as writer:
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        df.to_excel(writer, sheet_name=table, index=False)
        print(f"✅ Exported {table}")

conn.close()

print("\n🎉 Excel created with ALL DB tables")
