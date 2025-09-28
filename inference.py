import cv2
import numpy as np
import tensorflow as tf
import os
import time
from models.multimodal_model import build_attention_model
from config import IMG_SIZE, SEQUENCE_LENGTH, NUM_JOINTS, NUM_CLASSES, MODEL_SAVE_PATH, DATA_DIR
from utils.mediapipe_utils import hands_detector, extract_landmarks

# 클래스 이름 매핑 (폴더명 기준)
class_folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
class_names = {i: name for i, name in enumerate(class_folders)}
print("클래스 매핑:", class_names)

# 모델 빌드 & 가중치 로드
model = build_attention_model()
model.load_weights(MODEL_SAVE_PATH)
print(">>> 모델 가중치 로드 완료")

# 웹캠 실행
cap = cv2.VideoCapture(0)
frames, landmarks_seq = [], []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    lm = extract_landmarks(frame, hands_detector, NUM_JOINTS)

    frames.append(frame_gray)
    landmarks_seq.append(lm)

    if len(frames) >= SEQUENCE_LENGTH:
        X_cnn = np.array(frames[-SEQUENCE_LENGTH:], dtype="float32") / 255.0
        X_cnn = np.expand_dims(X_cnn, axis=(0, -1))
        X_ts = np.array(landmarks_seq[-SEQUENCE_LENGTH:], dtype="float32")
        X_ts = np.expand_dims(X_ts, axis=0)

        # 추론 + 성능 측정
        start = time.time()
        y_pred = model.predict([X_cnn, X_ts], verbose=0)
        end = time.time()
        latency = (end - start) * 1000
        fps = 1 / (end - start + 1e-8)

        y_class = np.argmax(y_pred, axis=1)[0]
        confidence = np.max(y_pred) * 100
        class_label = class_names.get(y_class, "Unknown")

        text = f"Action: {class_label} ({confidence:.1f}%) | {latency:.1f}ms | {fps:.1f} FPS"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
