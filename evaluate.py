import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from dataset.data_loader import load_multimodal_data
from utils.mediapipe_utils import hands_detector
from config import MODEL_SAVE_PATH, NUM_CLASSES

# --- Load test data ---
print("\n>>> 데이터 로드 중...")
X_cnn_train, X_cnn_test, X_ts_train, X_ts_test, y_train, y_test = load_multimodal_data(hands_detector)
y_test_cat = to_categorical(y_test, num_classes=NUM_CLASSES)

# --- Load model ---
print("\n>>> 모델 불러오는 중...")
custom_objects = {"Attention": tf.keras.layers.Layer}
model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects=custom_objects)

# --- Evaluate ---
print("\n>>> 모델 평가 중...")
loss, acc = model.evaluate([X_cnn_test, X_ts_test], y_test_cat, verbose=1)
print(f"\n테스트 정확도: {acc:.4f}, 테스트 손실: {loss:.4f}")

# --- Predictions ---
y_pred = model.predict([X_cnn_test, X_ts_test])
y_pred_classes = np.argmax(y_pred, axis=1)

# --- Classification Report ---
report = classification_report(y_test, y_pred_classes, digits=4)
print("\nClassification Report:\n", report)

with open("evaluation_report.txt", "w") as f:
    f.write("Classification Report\n")
    f.write(report)
print("\n분류 리포트가 evaluation_report.txt 에 저장되었습니다.")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix 이미지가 confusion_matrix.png 에 저장되었습니다.")
