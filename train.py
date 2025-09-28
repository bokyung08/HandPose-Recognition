import numpy as np
from tensorflow.keras.utils import to_categorical
from dataset.data_loader import load_multimodal_data
from models.multimodal_model import build_attention_model   # <- 3D 모델 사용
from utils.mediapipe_utils import hands_detector
from config import MODEL_SAVE_PATH, NUM_CLASSES
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(">>> train.py 실행 시작됨")

if __name__ == "__main__":
    print("데이터 로딩 시작")
    X_cnn_train, X_cnn_test, X_ts_train, X_ts_test, y_train, y_test = load_multimodal_data(hands_detector)
    print("훈련 데이터 크기:", len(y_train))
    print("테스트 데이터 크기:", len(y_test))

    # 라벨 원-핫 인코딩
    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)


    # 모델 생성
    model = build_attention_model()
    model.summary()

    # 학습
    history = model.fit([X_cnn_train, X_ts_train], y_train_cat,
                        validation_data=([X_cnn_test, X_ts_test], y_test_cat),
                        epochs=10, batch_size=32)

    # 가중치만 저장
    model.save_weights(MODEL_SAVE_PATH)
    print(f"Weights saved at {MODEL_SAVE_PATH}")

