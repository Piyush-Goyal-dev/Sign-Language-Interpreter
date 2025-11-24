import sqlite3
import os

print("=" * 60)
print("GESTURE DATABASE CHECKER")
print("=" * 60)

if not os.path.exists("gesture_db.db"):
    print("\n❌ Database file not found!")
    print("Please run: python create_gestures.py")
    exit(1)

try:
    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.execute("SELECT * FROM gesture ORDER BY g_id")
    
    gestures = cursor.fetchall()
    
    if len(gestures) == 0:
        print("\n⚠ Database is empty!")
        print("\nYou need to add gestures. Run:")
        print("  python create_gestures.py")
    else:
        print(f"\n✓ Found {len(gestures)} gesture(s) in database:\n")
        print("ID | Gesture Name")
        print("-" * 30)
        for row in gestures:
            print(f"{row[0]:2d} | {row[1]}")
        
        print("\n" + "=" * 60)
        print("CHECKING GESTURE FOLDERS")
        print("=" * 60)
        
        for row in gestures:
            g_id = row[0]
            g_name = row[1]
            folder_path = f"gestures/{g_id}"
            
            if os.path.exists(folder_path):
                num_images = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
                if num_images > 0:
                    print(f"✓ Gesture {g_id} ({g_name}): {num_images} images")
                else:
                    print(f"⚠ Gesture {g_id} ({g_name}): No images found!")
            else:
                print(f"❌ Gesture {g_id} ({g_name}): Folder not found!")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("SUGGESTIONS")
    print("=" * 60)
    
    if len(gestures) < 5:
        print("⚠ You have less than 5 gestures.")
        print("  For better accuracy, create at least 5-10 gestures.")
        print("  Run: python create_gestures.py")
    
    print("\n✓ Database check complete!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()