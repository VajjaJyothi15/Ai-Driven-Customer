import pandas as pd
import sqlite3
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "customer_data.csv"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "database",
    "customer_analytics.db"
)

df = pd.read_csv(CSV_PATH)

conn = sqlite3.connect(DB_PATH)

df.to_sql(
    "customers",
    conn,
    if_exists="replace",
    index=False
)

conn.commit()
conn.close()

print("Database Created Successfully")
print(f"Rows Inserted: {len(df)}")