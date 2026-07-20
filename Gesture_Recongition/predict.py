#predict.py
import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import Counter
from utils import normalize_landmarks

# Load the trained model
try:
    model = joblib.load('gesture_model.pkl')
except:
    print("Error: Train the model first using model_training.py")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)

# Stability Buffer to handle tremors
buffer = []
BUFFER_SIZE = 15 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    display_text = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            raw_list = []
            for lm in hand_lms.landmark:
                raw_list.extend([lm.x, lm.y, lm.z])
            
            clean_list = normalize_landmarks(raw_list)
            
            # Predict
            raw_pred = model.predict([clean_list])[0]
            buffer.append(raw_pred)
            
            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)
            
            # Get the most frequent prediction in the buffer (Smoothing)
            display_text = Counter(buffer).most_common(1)[0][0]

    # Clean UI Overlay
    cv2.rectangle(frame, (0, 0), (350, 80), (0, 0, 0), -1)
    cv2.putText(frame, display_text, (20, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.imshow("PD Assistive System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()