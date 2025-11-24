import sqlite3

def get_gesture_name(gesture_id):
    try:
        conn = sqlite3.connect("gesture_db.db")
        cursor = conn.execute(f"SELECT g_name FROM gesture WHERE g_id={gesture_id}")
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            return f"UNKNOWN_{gesture_id}"
    except Exception as e:
        print(f"Error: {e}")
        return f"ERROR_{gesture_id}"

# Test all gestures
conn = sqlite3.connect("gesture_db.db")
cursor = conn.execute("SELECT g_id FROM gesture ORDER BY g_id")
gesture_ids = [row[0] for row in cursor]
conn.close()

print("Testing gesture name retrieval:")
print("-" * 40)
for g_id in gesture_ids:
    name = get_gesture_name(g_id)
    print(f"Gesture {g_id} -> '{name}'")
