import sqlite3
import pandas as pd

conn = sqlite3.connect("feedback.db")

df = pd.read_sql_query("SELECT * FROM feedback", conn)
print(df)

conn.close()
