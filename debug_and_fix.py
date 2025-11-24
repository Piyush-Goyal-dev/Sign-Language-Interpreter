import sqlite3
import os
import cv2
from glob import glob

print("=" * 70)
print("SIGN LANGUAGE INTERPRETER - COMPLETE DIAGNOSTIC & FIX")
print("=" * 70)

issues = []
fixes = []

# Check 1: Database exists
print("\n[CHECK 1] Database File")
print("-" * 70)
if os.path.exists("gesture_db.db"):
    print("✓ gesture_db.db exists")
else:
    print("❌ gesture_db.db NOT found")
    issues.append("Database file missing")

# Check 2: Database contents
print("\n[CHECK 2] Database Contents")
print("-" * 70)
try:
    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.execute("SELECT g_id, g_name FROM gesture ORDER BY g_id")
    gestures = cursor.fetchall()
    
    if len(gestures) == 0:
        print("❌ Database is EMPTY - no gestures found")
        issues.append("Empty database")
    else:
        print(f"✓ Found {len(gestures)} gesture(s):\n")
        for g_id, g_name in gestures:
            print(f"   ID: {g_id} -> Name: '{g_name}'")
            if not g_name or g_name.strip() == "":
                issues.append(f"Gesture {g_id} has empty name")
    
    conn.close()
except Exception as e:
    print(f"❌ Error reading database: {e}")
    issues.append("Database error")
    gestures = []

# Check 3: Gesture folders
print("\n[CHECK 3] Gesture Folders & Images")
print("-" * 70)
if os.path.exists("gestures"):
    folders = [f for f in os.listdir("gestures") if os.path.isdir(os.path.join("gestures", f))]
    folders.sort()
    
    if len(folders) == 0:
        print("❌ No gesture folders found")
        issues.append("No gesture folders")
    else:
        print(f"✓ Found {len(folders)} gesture folder(s):\n")
        for folder in folders:
            folder_path = f"gestures/{folder}"
            images = glob(f"{folder_path}/*.jpg")
            print(f"   Folder '{folder}': {len(images)} images")
            
            if len(images) == 0:
                issues.append(f"Folder {folder} has no images")
else:
    print("❌ 'gestures' folder NOT found")
    issues.append("Gestures folder missing")

# Check 4: Model file
print("\n[CHECK 4] Trained Model")
print("-" * 70)
if os.path.exists("cnn_model_keras2.h5"):
    print("✓ cnn_model_keras2.h5 exists")
else:
    print("❌ cnn_model_keras2.h5 NOT found")
    issues.append("Model not trained")

# Check 5: Hand histogram
print("\n[CHECK 5] Hand Histogram")
print("-" * 70)
if os.path.exists("hist"):
    print("✓ hist file exists")
else:
    print("❌ hist file NOT found")
    issues.append("Hand histogram not set")

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if len(issues) == 0:
    print("✓ All checks passed!")
else:
    print(f"❌ Found {len(issues)} issue(s):\n")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")

# Offer fixes
if len(issues) > 0:
    print("\n" + "=" * 70)
    print("AUTOMATIC FIX OPTIONS")
    print("=" * 70)
    
    # Fix 1: Add gesture names to database if missing
    if gestures and any("empty name" in issue for issue in issues):
        print("\n[FIX 1] Add default names to gestures with empty names")
        fix = input("Apply this fix? (y/n): ")
        if fix.lower() == 'y':
            try:
                conn = sqlite3.connect("gesture_db.db")
                for g_id, g_name in gestures:
                    if not g_name or g_name.strip() == "":
                        new_name = f"Gesture_{g_id}"
                        conn.execute(f"UPDATE gesture SET g_name = '{new_name}' WHERE g_id = {g_id}")
                        print(f"   Updated gesture {g_id} -> '{new_name}'")
                conn.commit()
                conn.close()
                print("✓ Fix applied!")
                fixes.append("Added default gesture names")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Fix 2: Test database query
    print("\n[FIX 2] Test database query function")
    if gestures:
        try:
            conn = sqlite3.connect("gesture_db.db")
            test_id = gestures[0][0]
            cursor = conn.execute(f"SELECT g_name FROM gesture WHERE g_id={test_id}")
            result = cursor.fetchone()
            conn.close()
            
            if result:
                print(f"✓ Query test successful!")
                print(f"   Gesture {test_id} -> '{result[0]}'")
            else:
                print(f"❌ Query returned no results for gesture {test_id}")
        except Exception as e:
            print(f"❌ Query failed: {e}")

# Final recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

if "Database file missing" in issues or "Empty database" in issues:
    print("\n1. Create gestures:")
    print("   python create_gestures.py")
    print("   (Create at least 5 different gestures)")

if "No gesture folders" in issues or any("has no images" in issue for issue in issues):
    print("\n2. Record gesture images:")
    print("   python create_gestures.py")
    print("   (Press 'c' to start capturing)")

if "Model not trained" in issues:
    print("\n3. Train the model:")
    print("   python load_images.py")
    print("   python cnn_model_train.py")

if "Hand histogram not set" in issues:
    print("\n4. Set hand histogram:")
    print("   python set_hand_histogram.py")
    print("   (Press 'c' then 's')")

if len(issues) == 0:
    print("\n✓ Everything looks good!")
    print("\nYou can now run:")
    print("   python final.py")

print("\n" + "=" * 70)

# Create a test script to verify gesture name retrieval
print("\n[BONUS] Creating test script...")
with open("test_gesture_names.py", "w") as f:
    f.write("""import sqlite3

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
""")
print("✓ Created test_gesture_names.py")
print("\nRun it with: python test_gesture_names.py")

print("\n" + "=" * 70)