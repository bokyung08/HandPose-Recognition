import numpy as np
import cv2
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
except:
    hands_detector = None

def extract_landmarks(frame, detector, num_joints):
    if detector:
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            return [lm.x for lm in hand_landmarks.landmark] + \
                   [lm.y for lm in hand_landmarks.landmark] + \
                   [lm.z for lm in hand_landmarks.landmark]
    return np.zeros(num_joints * 3)
