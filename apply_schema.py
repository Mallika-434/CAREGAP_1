import sqlite3

def apply_schema():
    conn = sqlite3.connect('db.sqlite3')
    cur = conn.cursor()
    
    # Check if is_deceased column exists
    cur.execute("PRAGMA table_info(patients_patient)")
    columns = [row[1] for row in cur.fetchall()]
    
    if 'is_deceased' not in columns:
        print("Adding 'is_deceased' column to 'patients_patient' table...")
        cur.execute("ALTER TABLE patients_patient ADD COLUMN is_deceased bool NOT NULL DEFAULT 0")
        conn.commit()
        print("Column added successfully.")
    else:
        print("'is_deceased' column already exists.")
        
    conn.close()

if __name__ == '__main__':
    apply_schema()
