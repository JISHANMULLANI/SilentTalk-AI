
import numpy as np


def extract_hand_landmarks(results, hand_index=0):
    """
    Extract 21 hand landmarks as 63 float32 values.
    Returns zeros if the requested hand is not detected.
    """
    features = np.zeros(63, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return features

    if hand_index >= len(results.multi_hand_landmarks):
        return features

    hand_landmarks = results.multi_hand_landmarks[hand_index]

    features = np.array(
        [
            coordinate
            for landmark in hand_landmarks.landmark
            for coordinate in (landmark.x, landmark.y, landmark.z)
        ],
        dtype=np.float32,
    )

    return features


def normalize_landmarks(features):
    """
    Center landmarks at the wrist and scale by maximum 3D distance.
    Input/output shape: (63,)
    """
    features = np.asarray(features, dtype=np.float32)

    if features.shape != (63,):
        raise ValueError(
            f"Expected feature shape (63,), got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values.")

    landmarks = features.reshape(21, 3).copy()
    landmarks -= landmarks[0].copy()

    distances = np.linalg.norm(landmarks, axis=1)
    scale = float(np.max(distances))

    if scale > 1e-6:
        landmarks /= scale

    return landmarks.reshape(63).astype(np.float32)
