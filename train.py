import numpy as np
from tensorflow.keras.utils import to_categorical
from dataset.data_loader import load_multimodal_data
from models.multimodal_model import build_attention_model
from utils.mediapipe_utils import hands_detector
from config import MODEL_SAVE_PATH, NUM_CLASSES
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    X_cnn_train, X_cnn_test, X_ts_train, X_ts_test, y_train, y_test = load_multimodal_data(hands_detector)

    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

    model = build_attention_model()
    model.summary()

    history = model.fit([X_cnn_train, X_ts_train], y_train_cat,
                        validation_data=([X_cnn_test, X_ts_test], y_test_cat),
                        epochs=20, batch_size=32)

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")
