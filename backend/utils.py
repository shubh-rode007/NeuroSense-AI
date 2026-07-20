#utils.py
import numpy as np

def normalize_landmarks(landmark_list):
    # Convert flat list to (21, 3) array
    temp_landmarks = np.array(landmark_list).reshape(-1, 3)
    
    # 1. Centering: Subtract the wrist (point 0) from all points
    origin = temp_landmarks[0]
    temp_landmarks = temp_landmarks - origin
    
    # 2. Scaling: Normalize by the maximum distance found
    max_val = np.max(np.abs(temp_landmarks))
    if max_val != 0:
        temp_landmarks = temp_landmarks / max_val
        
    return temp_landmarks.flatten().tolist()