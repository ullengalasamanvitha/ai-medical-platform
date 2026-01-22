import sqlite3

conn = sqlite3.connect("app.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM payments")
rows = cursor.fetchall()

print("Payments table data:")
print(rows)

conn.close()
