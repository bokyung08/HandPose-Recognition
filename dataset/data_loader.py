import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from utils.mediapipe_utils import extract_landmarks
from config import IMG_SIZE, SEQUENCE_LENGTH, NUM_JOINTS, DATA_DIR

def extract_frames_and_landmarks(video_path, detector):
    frames, landmarks = [], []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[WARN] 비디오 열기 실패: {video_path}")
        return np.array([]), np.array([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frames.append(frame_gray)

        lm = extract_landmarks(frame, detector, NUM_JOINTS)
        landmarks.append(lm)

    cap.release()

    if len(frames) == 0:
        print(f"[WARN] 프레임 추출 실패: {video_path}")

    return np.array(frames), np.array(landmarks)


def load_multimodal_data(detector):
    X_cnn, X_ts, y = [], [], []

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] DATA_DIR 경로 없음: {DATA_DIR}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    class_folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if len(class_folders) == 0:
        print(f"[ERROR] DATA_DIR에 클래스 폴더가 없음: {DATA_DIR}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    class_map = {name: i for i, name in enumerate(class_folders)}
    print("클래스 매핑:", class_map)

    for folder_name, class_id in class_map.items():
        class_path = os.path.join(DATA_DIR, folder_name)
        video_files = [f for f in os.listdir(class_path) if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]

        if len(video_files) == 0:
            print(f"[WARN] {folder_name} 폴더에 비디오 없음")
            continue

        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            frames, landmarks = extract_frames_and_landmarks(video_path, detector)

            if len(frames) >= SEQUENCE_LENGTH:
                for i in range(len(frames) - SEQUENCE_LENGTH + 1):
                    X_cnn.append(frames[i:i+SEQUENCE_LENGTH])
                    X_ts.append(landmarks[i:i+SEQUENCE_LENGTH])
                    y.append(class_id)
            else:
                print(f"[WARN] 시퀀스 길이 부족 ({len(frames)} frames): {video_file}")

    if len(X_cnn) == 0:
        print("[ERROR] 어떤 비디오도 데이터셋으로 변환되지 않음")

    X_cnn = np.expand_dims(np.array(X_cnn, dtype='float32') / 255.0, -1)
    X_ts = np.array(X_ts, dtype='float32')
    y = np.array(y)

    return train_test_split(X_cnn, X_ts, y, test_size=0.2, random_state=42, stratify=y if len(y) > 0 else None)
