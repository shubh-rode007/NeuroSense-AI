#collect_data.py
import cv2
import mediapipe as mp
import csv
import os
from utils import normalize_landmarks

# --- CONFIGURATION ---
LABEL_NAME = "WATER"  # Change this for each gesture YES,NO,HELP,COME,GO,PAIN,MEDICINE,TOILET,CALL,THANK YOU,HUNGRY,|| WATER,STOP,SIT,STAND,SLEEP,OK
SAMPLES_NEEDED = 300   # Frames to capture per gesture
# ---------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
data = []
count = 0

print(f"Starting collection for: {LABEL_NAME}. Press 'q' to stop early.")

while count < SAMPLES_NEEDED:
    ret, frame = cap.read()
    if not ret: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            raw_list = []
            for lm in hand_lms.landmark:
                raw_list.extend([lm.x, lm.y, lm.z])
            
            # Apply Normalization
            clean_list = normalize_landmarks(raw_list)
            clean_list.append(LABEL_NAME) # Add label to the row
            data.append(clean_list)
            count += 1
            
            # Show progress on screen
            cv2.putText(frame, f"Capturing {LABEL_NAME}: {count}/{SAMPLES_NEEDED}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Data Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Save to CSV
file_exists = os.path.isfile('dataset.csv')
with open('dataset.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        header = [f'pt_{i}' for i in range(63)] + ['label']
        writer.writerow(header)
    writer.writerows(data)

print(f"Success! {count} samples of '{LABEL_NAME}' saved to dataset.csv")
cap.release()
cv2.destroyAllWindows()