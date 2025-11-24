"""
SIGN LANGUAGE INTERPRETER - MASTER SETUP
Complete automated setup for 15 common words
"""

import os
import sys

print("=" * 80)
print(" " * 20 + "SIGN LANGUAGE INTERPRETER")
print(" " * 15 + "Complete Setup for 15 Common Words")
print("=" * 80)

# Define 15 common words with distinct gestures
GESTURES = [
    {"id": 0, "name": "Hello", "description": "Open hand, all 5 fingers spread"},
    {"id": 1, "name": "Thanks", "description": "Flat hand near chin, move forward"},
    {"id": 2, "name": "Yes", "description": "Fist with thumb up (thumbs up)"},
    {"id": 3, "name": "No", "description": "Fist (closed hand)"},
    {"id": 4, "name": "Please", "description": "Flat hand on chest, circular motion"},
    {"id": 5, "name": "Sorry", "description": "Fist on chest, circular motion"},
    {"id": 6, "name": "Help", "description": "Both hands, one on top of other"},
    {"id": 7, "name": "Good", "description": "Thumbs up"},
    {"id": 8, "name": "Bad", "description": "Thumbs down"},
    {"id": 9, "name": "Stop", "description": "Palm facing forward"},
    {"id": 10, "name": "Go", "description": "Point forward with index finger"},
    {"id": 11, "name": "Come", "description": "Wave hand towards you"},
    {"id": 12, "name": "Me", "description": "Point to yourself"},
    {"id": 13, "name": "You", "description": "Point forward"},
    {"id": 14, "name": "Bye", "description": "Wave hand side to side"},
]

def print_gesture_guide():
    """Print visual guide for all gestures"""
    print("\n" + "=" * 80)
    print("GESTURE GUIDE - 15 COMMON WORDS")
    print("=" * 80)
    print("\nMake these gestures EXACTLY as described:\n")
    
    for g in GESTURES:
        print(f"  [{g['id']:2d}] {g['name']:10s} - {g['description']}")
    
    print("\n" + "=" * 80)
    print("IMPORTANT TIPS:")
    print("=" * 80)
    print("  1. Each gesture should look COMPLETELY DIFFERENT")
    print("  2. Use GOOD LIGHTING (bright, even light)")
    print("  3. PLAIN BACKGROUND (white wall or solid color)")
    print("  4. Keep hand at SAME DISTANCE from camera")
    print("  5. Make gestures CLEAR and EXAGGERATED")
    print("=" * 80)

def check_requirements():
    """Check if all required files and folders exist"""
    print("\n" + "=" * 80)
    print("CHECKING REQUIREMENTS")
    print("=" * 80)
    
    files_needed = [
        "set_hand_histogram.py",
        "create_gestures.py", 
        "load_images.py",
        "cnn_model_train.py",
        "final.py"
    ]
    
    missing = []
    for f in files_needed:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} - MISSING")
            missing.append(f)
    
    if missing:
        print(f"\n  ⚠ Missing {len(missing)} required file(s)")
        print("  Please ensure all files are in the current directory")
        return False
    
    print("\n  ✓ All required files found!")
    return True

def show_workflow():
    """Show the complete workflow"""
    print("\n" + "=" * 80)
    print("COMPLETE WORKFLOW")
    print("=" * 80)
    
    steps = [
        ("STEP 1", "Set Hand Histogram", "python set_hand_histogram.py", "5 minutes"),
        ("STEP 2", "Create 15 Gestures", "python create_gestures.py (15 times)", "45-60 minutes"),
        ("STEP 3", "Prepare Training Data", "python load_images.py", "2-5 minutes"),
        ("STEP 4", "Train CNN Model", "python cnn_model_train.py", "20-40 minutes"),
        ("STEP 5", "Test Recognition", "python final.py", "Ongoing"),
    ]
    
    for step, name, command, time in steps:
        print(f"\n  {step}: {name}")
        print(f"  Command: {command}")
        print(f"  Time: {time}")
    
    print("\n" + "=" * 80)
    print("TOTAL TIME: ~70-110 minutes")
    print("=" * 80)

def main():
    """Main setup function"""
    
    # Show gesture guide
    print_gesture_guide()
    
    # Check requirements
    if not check_requirements():
        print("\n  ⚠ Setup cannot continue without required files")
        return
    
    # Show workflow
    show_workflow()
    
    # Ask if user wants to start
    print("\n" + "=" * 80)
    print("READY TO START?")
    print("=" * 80)
    print("\nThis will guide you through creating a working sign language system.")
    print("You need:")
    print("  - 70-110 minutes of time")
    print("  - Good lighting")
    print("  - Plain background")
    print("  - Webcam")
    
    choice = input("\nReady to begin? (yes/no): ").strip().lower()
    
    if choice in ['yes', 'y']:
        print("\n" + "=" * 80)
        print("STARTING SETUP...")
        print("=" * 80)
        
        print("\n▶ STEP 1: Set Hand Histogram")
        print("  Run this command now:")
        print("  > python set_hand_histogram.py")
        print("\n  Instructions:")
        print("    1. Position your palm in the green squares")
        print("    2. Press 'c' to capture")
        print("    3. Check threshold window (hand should be white)")
        print("    4. Press 's' to save")
        
        input("\nPress ENTER when Step 1 is complete...")
        
        print("\n▶ STEP 2: Create Gestures")
        print("  For EACH gesture below, run:")
        print("  > python create_gestures.py")
        print("\n  Gestures to create:")
        
        for g in GESTURES:
            print(f"\n  Gesture {g['id']}: {g['name']}")
            print(f"    Description: {g['description']}")
            print(f"    Command: python create_gestures.py")
            print(f"    Enter ID: {g['id']}")
            print(f"    Enter Name: {g['name']}")
        
        print("\n  This will take 45-60 minutes for all 15 gestures")
        print("  Be patient and make each gesture clearly!")
        
        input("\nPress ENTER when all gestures are created...")
        
        print("\n▶ STEP 3: Prepare Training Data")
        print("  Run: python load_images.py")
        
        input("\nPress ENTER when complete...")
        
        print("\n▶ STEP 4: Train Model")
        print("  Run: python cnn_model_train.py")
        print("  This will take 20-40 minutes")
        
        input("\nPress ENTER when training is complete...")
        
        print("\n▶ STEP 5: Test Your System!")
        print("  Run: python final.py")
        print("\n  Controls:")
        print("    'v' - Toggle voice")
        print("    'q' - Quit")
        
        print("\n" + "=" * 80)
        print("SETUP GUIDE COMPLETE!")
        print("=" * 80)
        print("\nYou should now have a working sign language interpreter!")
        print("If accuracy is low (<80%), try:")
        print("  1. Recalibrate hand histogram with better lighting")
        print("  2. Make gestures more distinct/exaggerated")
        print("  3. Add more training images per gesture")
        
    else:
        print("\nSetup cancelled. Run this script again when ready.")

if __name__ == "__main__":
    main()