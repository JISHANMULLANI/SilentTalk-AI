
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence import normalize_sequence


# Configuration
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "alphabet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "sign_lstm.keras"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

SEQUENCE_LENGTH = 30
FEATURES = 63
EPOCHS = 50
BATCH_SIZE = 16
RANDOM_SEED = 42

LETTERS = list("abcdefikorsuvwy")


def load_dataset():
    """Load and normalize labeled sequences from disk."""
    sequences = []
    labels = []

    print("Loading alphabet dataset...")

    for letter in LETTERS:
        label_dir = DATA_DIR / letter

        files = (
            sorted(label_dir.glob("sequence_*.npy"))
            if label_dir.exists()
            else []
        )

        print(f"{letter.upper()}: {len(files)} samples")

        for file_path in files:
            sample = np.load(file_path, allow_pickle=False)

            if sample.shape != (SEQUENCE_LENGTH, FEATURES):
                raise ValueError(
                    f"{file_path} has shape {sample.shape}; "
                    f"expected (30, 63)."
                )

            if not np.isfinite(sample).all():
                raise ValueError(
                    f"{file_path} contains NaN or infinite values."
                )

            # Must match preprocessing used during live prediction.
            sample = normalize_sequence(sample)

            sequences.append(sample)
            labels.append(letter)

    if not sequences:
        raise ValueError(
            f"No samples found in {DATA_DIR}. "
            "Collect alphabet samples first."
        )

    # Every class must have at least two examples for a stratified split.
    counts = {letter: labels.count(letter) for letter in LETTERS}
    insufficient = [
        f"{letter.upper()} ({count})"
        for letter, count in counts.items()
        if count < 2
    ]

    if insufficient:
        raise ValueError(
            "Every letter needs at least 2 samples. "
            "Insufficient data: " + ", ".join(insufficient)
        )

    X = np.asarray(sequences, dtype=np.float32)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    print(f"\nTotal samples: {len(X)}")
    print(f"Input shape: {X.shape}")
    print(f"Classes: {list(encoder.classes_)}")

    return X, y, encoder


def build_model(number_of_classes):
    """Build the LSTM classifier."""
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, FEATURES)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(number_of_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, encoder = load_dataset()

    # Keep each letter represented in both sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Validation samples:", len(X_val))

    model = build_model(len(encoder.classes_))
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nStarting LSTM training...")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # Evaluate the best saved model.
    best_model = tf.keras.models.load_model(MODEL_PATH)
    loss, accuracy = best_model.evaluate(
        X_val, y_val, verbose=0
    )

    # Save the exact class ordering used by the model.
    with open(ENCODER_PATH, "wb") as file:
        pickle.dump(encoder, file)

    print("\nTraining complete!")
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Label encoder saved: {ENCODER_PATH}")


if __name__ == "__main__":
    main()
