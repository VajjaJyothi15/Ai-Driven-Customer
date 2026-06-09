import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DB_PATH = os.path.join(
    BASE_DIR,
    "database",
    "customer_analytics.db"
)


def get_data():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM customers",
        conn
    )

    conn.close()

    return df