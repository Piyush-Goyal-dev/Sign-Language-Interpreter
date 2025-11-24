import cv2, pickle
import numpy as np
import os
import sqlite3
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model
from threading import Thread
from collections import deque

# Try to initialize text-to-speech
try:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voice_available = True
except:
    voice_available = False
    engine = None

print("=" * 80)
print(" " * 25 + "SIGN LANGUAGE INTERPRETER")
print(" " * 30 + "15 Common Words")
print("=" * 80)

# Load model
print("\n[1/3] Loading trained model...")
try:
    model = load_model('cnn_model_keras2.h5')
    print("      ✓ Model loaded successfully")
except Exception as e:
    print(f"      ❌ Error: {e}")
    print("\n      Please train the model first:")
    print("      python cnn_model_train.py")
    exit(1)

# Load hand histogram
print("[2/3] Loading hand histogram...")
try:
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    print("      ✓ Hand histogram loaded")
except Exception as e:
    print(f"      ❌ Error: {e}")
    print("\n      Please set hand histogram first:")
    print("      python set_hand_histogram.py")
    exit(1)

# Get image size
print("[3/3] Loading configuration...")
try:
    from glob import glob
    all_images = glob('gestures/*/*.jpg')
    if len(all_images) > 0:
        img = cv2.imread(all_images[0], 0)
        image_x, image_y = img.shape
    print(f"      ✓ Image size: {image_x}x{image_y}")
except Exception as e:
    print(f"      ❌ Error: {e}")
    exit(1)

# Load gesture names
print("\n[INFO] Loading gesture database...")
try:
    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.execute("SELECT g_id, g_name FROM gesture ORDER BY g_id")
    gestures = cursor.fetchall()
    conn.close()
    print(f"       ✓ Loaded {len(gestures)} gestures:")
    for gid, gname in gestures:
        print(f"         {gid}: {gname}")
except Exception as e:
    print(f"       ❌ Error: {e}")

print("\n" + "=" * 80)
print("READY TO START!")
print("=" * 80)
print("\n📖 CONTROLS:")
print("   'V' - Toggle voice on/off")
print("   'C' - Clear current sentence")
print("   'Q' - Quit")
print("\n💡 USAGE:")
print("   1. Make gesture in green rectangle")
print("   2. Hold steady for 2 seconds")
print("   3. Gesture will be added to sentence")
print("   4. Remove hand to speak the sentence")
print("\n" + "=" * 80)

input("\nPress ENTER to start camera...")

x, y, w, h = 300, 100, 300, 300
is_voice_on = voice_available

def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
    try:
        conn = sqlite3.connect("gesture_db.db")
        cursor = conn.execute(f"SELECT g_name FROM gesture WHERE g_id={pred_class}")
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
    except:
        pass
    return f"Gesture{pred_class}"

def get_pred_from_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1+h1, x1:x1+w1]
    
    if save_img.size == 0:
        return "", 0
    
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2), int((w1-h1)/2), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2), int((h1-w1)/2), cv2.BORDER_CONSTANT, (0, 0, 0))
    
    pred_probab, pred_class = keras_predict(model, save_img)
    confidence = pred_probab * 100
    
    if confidence > 60:  # Threshold
        text = get_pred_text_from_db(pred_class)
        return text, confidence
    
    return "", confidence

def get_img_contour_thresh(img):
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    blur = cv2.GaussianBlur(dst, (11,11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh,thresh,thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y+h, x:x+w]
    
    contours_result = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = contours_result[1] if len(contours_result) == 3 else contours_result[0]
    
    return img, contours, thresh

def say_text(text):
    if not is_voice_on or not voice_available or engine is None:
        return
    try:
        while engine._inLoop:
            pass
        engine.say(text)
        engine.runAndWait()
    except:
        pass

def recognize_gestures():
    global is_voice_on
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("\n❌ Error: Could not open camera")
        return
    
    print("\n✓ Camera opened successfully")
    print("✓ Recognition started\n")
    
    sentence = []
    current_word = ""
    last_word = ""
    word_frames = 0
    no_hand_frames = 0
    confidence_history = deque(maxlen=10)
    
    # Prediction smoothing
    WORD_THRESHOLD = 25  # Frames to confirm word
    NO_HAND_THRESHOLD = 15  # Frames without hand to finalize sentence
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
        
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img)
        
        pred_text = ""
        confidence = 0
        
        # Find hand
        if len(contours) > 0:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 0]
            
            if len(valid_contours) > 0:
                contour = max(valid_contours, key=cv2.contourArea)
                
                if cv2.contourArea(contour) > 10000:
                    pred_text, confidence = get_pred_from_contour(contour, thresh)
                    confidence_history.append(confidence)
                    no_hand_frames = 0
                    
                    # Stable prediction
                    if pred_text and pred_text == last_word:
                        word_frames += 1
                    else:
                        word_frames = 0
                        last_word = pred_text
                    
                    # Add word to sentence
                    if word_frames == WORD_THRESHOLD and pred_text:
                        if not sentence or sentence[-1] != pred_text:
                            sentence.append(pred_text)
                            current_word = pred_text
                            Thread(target=say_text, args=(pred_text,)).start()
                            print(f"✓ Added: {pred_text}")
                            word_frames = 0
                else:
                    no_hand_frames += 1
        else:
            no_hand_frames += 1
        
        # Speak sentence when hand removed
        if no_hand_frames == NO_HAND_THRESHOLD and len(sentence) > 0:
            full_sentence = " ".join(sentence)
            print(f"\n📢 Speaking: {full_sentence}\n")
            Thread(target=say_text, args=(full_sentence,)).start()
            sentence = []
            current_word = ""
            no_hand_frames = 0
        
        # Calculate average confidence
        avg_conf = np.mean(confidence_history) if confidence_history else 0
        
        # Create UI
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(blackboard, "Sign Language Interpreter", (120, 40), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 255), 2)
        
        # Current prediction
        pred_color = (0, 255, 0) if confidence > 70 else (0, 165, 255) if confidence > 50 else (0, 0, 255)
        cv2.putText(blackboard, f"Current: {pred_text if pred_text else '---'}", (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, pred_color, 2)
        
        # Confidence
        cv2.putText(blackboard, f"Confidence: {avg_conf:.1f}%", (30, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        
        # Progress bar for word confirmation
        if word_frames > 0:
            progress = min(word_frames / WORD_THRESHOLD, 1.0)
            bar_width = int(580 * progress)
            cv2.rectangle(blackboard, (30, 160), (30 + bar_width, 180), (0, 255, 0), -1)
            cv2.rectangle(blackboard, (30, 160), (610, 180), (100, 100, 100), 2)
        
        # Sentence
        cv2.putText(blackboard, "Sentence:", (30, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        
        sentence_text = " ".join(sentence)
        if len(sentence_text) > 35:
            sentence_text = sentence_text[-35:]
        cv2.putText(blackboard, sentence_text, (30, 270), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(blackboard, "Hold gesture 2 sec to add word", (30, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(blackboard, "Remove hand to speak sentence", (30, 375), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Status
        voice_text = "Voice: ON" if is_voice_on else "Voice: OFF"
        voice_color = (0, 255, 0) if is_voice_on else (0, 0, 255)
        cv2.putText(blackboard, voice_text, (30, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, voice_color, 2)
        
        cv2.putText(blackboard, "Q:Quit | V:Voice | C:Clear", (320, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw hand rectangle
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.putText(img, "Place hand here", (x+50, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine views
        combined = np.hstack((img, blackboard))
        
        cv2.imshow("Sign Language Interpreter", combined)
        cv2.imshow("Hand Detection (should be WHITE)", thresh)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('v') or key == ord('V'):
            is_voice_on = not is_voice_on
            print(f"Voice: {'ON' if is_voice_on else 'OFF'}")
        elif key == ord('c') or key == ord('C'):
            sentence = []
            current_word = ""
            print("Sentence cleared")
    
    cam.release()
    cv2.destroyAllWindows()
    print("\n✓ Application closed")

# Initialize
print("Initializing model...")
keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
print("✓ Ready!\n")

# Start
try:
    recognize_gestures()
except KeyboardInterrupt:
    print("\n\n⚠ Stopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()