#gesture_engine.py
import cv2
import mediapipe as mp
import joblib
from collections import Counter
from utils import normalize_landmarks

class GestureProcessor:
    def __init__(self, model_path='gesture_model.pkl'):
        # 1. Load the ML Model
        try:
            self.model = joblib.load(model_path)
            print("✅ SUCCESS: Gesture Model Loaded.")
        except Exception as e:
            print(f"❌ ERROR: Could not load model. Reason: {e}")
            self.model = None

        # 2. Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 3. Tremor Stability Buffer
        self.buffer = []
        self.BUFFER_SIZE = 15

    def process_frame(self, frame):
        prediction = "No Hand Detected"
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks and self.model:
            for hand_lms in results.multi_hand_landmarks:
                # Draw landmarks so the user knows the AI sees them
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract and Normalize coordinates
                raw_list = []
                for lm in hand_lms.landmark:
                    raw_list.extend([lm.x, lm.y, lm.z])
                
                clean_list = normalize_landmarks(raw_list)
                
                # Predict
                raw_pred = self.model.predict([clean_list])[0]
                self.buffer.append(raw_pred)
                
                if len(self.buffer) > self.BUFFER_SIZE:
                    self.buffer.pop(0)
                
                prediction = Counter(self.buffer).most_common(1)[0][0]

        return frame, prediction