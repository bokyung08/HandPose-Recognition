import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from utils.mediapipe_utils import extract_landmarks
from config import IMG_SIZE, SEQUENCE_LENGTH, NUM_JOINTS, DATA_DIR

def extract_frames_and_landmarks(video_path, detector):
    frames, landmarks = [], []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frames.append(frame_gray)

        lm = extract_landmarks(frame, detector, NUM_JOINTS)
        landmarks.append(lm)

    cap.release()
    return np.array(frames), np.array(landmarks)

def load_multimodal_data(detector):
    X_cnn, X_ts, y = [], [], []
    class_folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    class_map = {name: i for i, name in enumerate(class_folders)}

    for folder_name, class_id in class_map.items():
        class_path = os.path.join(DATA_DIR, folder_name)
        for video_file in os.listdir(class_path):
            if video_file.endswith(('.mp4', '.avi','mov','MOV')):
                frames, landmarks = extract_frames_and_landmarks(os.path.join(class_path, video_file), detector)
                if len(frames) >= SEQUENCE_LENGTH:
                    for i in range(len(frames) - SEQUENCE_LENGTH + 1):
                        X_cnn.append(frames[i:i+SEQUENCE_LENGTH])
                        X_ts.append(landmarks[i:i+SEQUENCE_LENGTH])
                        y.append(class_id)

    X_cnn = np.expand_dims(np.array(X_cnn, dtype='float32') / 255.0, -1)
    X_ts = np.array(X_ts, dtype='float32')
    y = np.array(y)

    return train_test_split(X_cnn, X_ts, y, test_size=0.2, random_state=42, stratify=y)
