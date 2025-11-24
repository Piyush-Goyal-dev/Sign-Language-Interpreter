import cv2
import numpy as np
import pickle, os, sqlite3, random

image_x, image_y = 50, 50

# 15 Common Words with descriptions
PREDEFINED_GESTURES = {
    0: {"name": "Hello", "desc": "Open hand, all 5 fingers spread"},
    1: {"name": "Thanks", "desc": "Flat hand near chin"},
    2: {"name": "Yes", "desc": "Thumbs up"},
    3: {"name": "No", "desc": "Closed fist"},
    4: {"name": "Please", "desc": "Flat hand on chest"},
    5: {"name": "Sorry", "desc": "Fist on chest"},
    6: {"name": "Help", "desc": "One hand on top of other"},
    7: {"name": "Good", "desc": "Thumbs up"},
    8: {"name": "Bad", "desc": "Thumbs down"},
    9: {"name": "Stop", "desc": "Palm facing forward"},
    10: {"name": "Go", "desc": "Point forward"},
    11: {"name": "Come", "desc": "Wave towards you"},
    12: {"name": "Me", "desc": "Point to yourself"},
    13: {"name": "You", "desc": "Point forward"},
    14: {"name": "Bye", "desc": "Wave hand"},
}

def get_hand_hist():
    if not os.path.exists("hist"):
        print("\n❌ ERROR: Hand histogram file not found!")
        print("Please run: python set_hand_histogram.py first")
        exit(1)
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def init_create_folder_database():
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (g_id, g_name)
    try:
        conn.execute(cmd)
    except sqlite3.IntegrityError:
        choice = input("Gesture ID already exists. Update? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
            conn.execute(cmd)
        else:
            print("Skipping...")
            conn.close()
            return False
    conn.commit()
    conn.close()
    return True

def store_images(g_id, g_name):
    total_pics = 1200
    hist = get_hand_hist()
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    x, y, w, h = 300, 100, 300, 300
    create_folder("gestures/" + str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    print("\n" + "=" * 70)
    print(f"RECORDING GESTURE {g_id}: {g_name}")
    print("=" * 70)
    print("\n📋 INSTRUCTIONS:")
    print("  1. Position your hand in the GREEN RECTANGLE")
    print("  2. Make the gesture clearly")
    print("  3. Press 'C' to START capturing")
    print("  4. Hold gesture steady - it captures automatically")
    print("  5. Press 'Q' to quit early")
    print(f"\n🎯 Target: {total_pics} images")
    print("=" * 70 + "\n")

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to capture image")
            break
            
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y + h, x:x + w]
        
        contours_result = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours_result) == 3:
            contours = contours_result[1]
        else:
            contours = contours_result[0]

        if len(contours) > 0:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 0]
            
            if len(valid_contours) > 0:
                contour = max(valid_contours, key=cv2.contourArea)
                
                if cv2.contourArea(contour) > 10000 and frames > 50 and flag_start_capturing:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    save_img = thresh[y1:y1 + h1, x1:x1 + w1]
                    
                    if save_img.size > 0:
                        pic_no += 1
                        
                        if w1 > h1:
                            save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                          cv2.BORDER_CONSTANT, (0, 0, 0))
                        elif h1 > w1:
                            save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                          cv2.BORDER_CONSTANT, (0, 0, 0))
                        
                        save_img = cv2.resize(save_img, (image_x, image_y))
                        
                        rand = random.randint(0, 10)
                        if rand % 2 == 0:
                            save_img = cv2.flip(save_img, 1)
                        
                        cv2.putText(img, "CAPTURING...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0))
                        cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)

        # Display
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Progress
        progress = (pic_no / total_pics) * 100
        cv2.putText(img, f"{pic_no}/{total_pics} ({progress:.1f}%)", (30, 450), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0), 2)
        
        # Gesture info
        cv2.putText(img, f"Gesture: {g_name}", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if not flag_start_capturing:
            cv2.putText(img, "Press 'C' to START", (150, 240), 
                       cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 2)
        
        cv2.imshow("Recording Gesture", img)
        cv2.imshow("Threshold (Your hand should be WHITE)", thresh)
        
        keypress = cv2.waitKey(1)
        if keypress == ord('c') or keypress == ord('C'):
            if not flag_start_capturing:
                flag_start_capturing = True
                print("✓ Capturing started!")
            else:
                flag_start_capturing = False
                frames = 0
                print("⏸ Paused")
                
        if keypress == ord('q') or keypress == ord('Q'):
            print("⚠ Quit by user")
            break
            
        if flag_start_capturing:
            frames += 1
            
        if pic_no == total_pics:
            print(f"\n✓ SUCCESS! Captured {total_pics} images for '{g_name}'")
            break

    cam.release()
    cv2.destroyAllWindows()

def show_gesture_menu():
    """Show menu of predefined gestures"""
    print("\n" + "=" * 70)
    print("15 COMMON WORDS - GESTURE MENU")
    print("=" * 70)
    print("\nID | Gesture Name | Description")
    print("-" * 70)
    for gid, info in PREDEFINED_GESTURES.items():
        print(f"{gid:2d} | {info['name']:12s} | {info['desc']}")
    print("=" * 70)

def main():
    print("\n" + "=" * 70)
    print("SIGN LANGUAGE GESTURE RECORDER")
    print("=" * 70)
    
    init_create_folder_database()
    
    # Show menu
    show_gesture_menu()
    
    # Get gesture ID
    print("\n📝 Enter gesture details:")
    try:
        g_id = int(input("Gesture ID (0-14): "))
        if g_id < 0 or g_id > 14:
            print("❌ Invalid ID. Use 0-14")
            return
    except ValueError:
        print("❌ Invalid input")
        return
    
    # Get gesture name
    if g_id in PREDEFINED_GESTURES:
        default_name = PREDEFINED_GESTURES[g_id]["name"]
        default_desc = PREDEFINED_GESTURES[g_id]["desc"]
        print(f"\n💡 Suggested: {default_name} - {default_desc}")
        use_default = input(f"Use '{default_name}'? (y/n): ").strip().lower()
        
        if use_default in ['y', 'yes', '']:
            g_name = default_name
        else:
            g_name = input("Enter custom name: ").strip()
    else:
        g_name = input("Gesture name: ").strip()
    
    if not g_name:
        print("❌ Name cannot be empty")
        return
    
    # Store in database
    if store_in_db(g_id, g_name):
        # Record images
        store_images(g_id, g_name)
        
        print("\n" + "=" * 70)
        print("✓ GESTURE RECORDING COMPLETE!")
        print("=" * 70)
        print(f"Gesture {g_id}: {g_name}")
        print(f"Location: gestures/{g_id}/")
        print("\n📊 Next steps:")
        print("  1. Record more gestures (repeat this script)")
        print("  2. When done with all gestures, run: python load_images.py")
        print("  3. Then train: python cnn_model_train.py")
        print("=" * 70 + "\n")

if __name__ == "__main__":
    main()