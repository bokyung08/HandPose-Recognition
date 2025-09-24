from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, TimeDistributed,
    Flatten, Dense, Dropout, LSTM, Concatenate
)
from tensorflow.keras.models import Model
from models.attention_layer import Attention
from config import SEQUENCE_LENGTH, IMG_SIZE, NUM_JOINTS, NUM_CLASSES


def build_attention_model():
    # --- Video Branch (3D CNN + LSTM + Attention) ---
    video_input = Input(
        shape=(SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 1),
        name="video_input"
    )

    # 3D CNN layers
    x = Conv3D(32, kernel_size=(3, 3, 3), activation="relu", padding="same")(video_input)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = Conv3D(128, kernel_size=(3, 3, 3), activation="relu", padding="same")(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    # Flatten only spatial dims, keep time dimension
    x = TimeDistributed(Flatten())(x)   # (batch, time, features)

    # Sequence modeling with LSTM
    video_lstm = LSTM(128, return_sequences=True)(x)
    video_context = Attention()(video_lstm)   # (batch, features)

    # --- Landmark Branch (LSTM + Attention) ---
    ts_input = Input(shape=(SEQUENCE_LENGTH, NUM_JOINTS * 3), name="ts_input")
    ts_lstm = LSTM(128, return_sequences=True)(ts_input)
    ts_context = Attention()(ts_lstm)
    ts_features = Dense(128, activation="relu")(ts_context)

    # --- Fusion ---
    combined = Concatenate()([video_context, ts_features])
    combined = Dense(128, activation="relu")(combined)
    combined = Dropout(0.5)(combined)

    output = Dense(NUM_CLASSES, activation="softmax")(combined)

    model = Model(inputs=[video_input, ts_input], outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model
