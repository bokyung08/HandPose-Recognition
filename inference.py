import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from utils.mediapipe_utils import extract_landmarks, hands_detector
from config import IMG_SIZE, SEQUENCE_LENGTH, NUM_JOINTS, MODEL_SAVE_PATH

model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'Attention': tf.keras.layers.Layer})

cap = cv2.VideoCapture(0)
frames, landmarks = deque(maxlen=SEQUENCE_LENGTH), deque(maxlen=SEQUENCE_LENGTH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    frames.append(frame_gray)
    landmarks.append(extract_landmarks(frame, hands_detector, NUM_JOINTS))

    if len(frames) == SEQUENCE_LENGTH:
        X_cnn = np.expand_dims(np.array(frames)/255.0, (0,-1))
        X_ts = np.expand_dims(np.array(landmarks), 0)
        pred = model.predict([X_cnn, X_ts])
        action = np.argmax(pred)
        cv2.putText(frame, f"Action: {action}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
