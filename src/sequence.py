
import numpy as np


def generate_sequences(features, sequence_length=30, stride=1):
    """
    Generate fixed-length sequences from frame-level features.

    Args:
        features: Array shaped (num_frames, 63).
        sequence_length: Frames per sequence.
        stride: Distance between the starts of consecutive windows.

    Returns:
        Array shaped (num_sequences, sequence_length, 63).
    """
    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2 or features.shape[1] != 63:
        raise ValueError(
            f"Expected shape (num_frames, 63), got {features.shape}"
        )

    if sequence_length <= 0 or stride <= 0:
        raise ValueError("sequence_length and stride must be positive.")

    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values.")

    if len(features) < sequence_length:
        return np.empty((0, sequence_length, 63), dtype=np.float32)

    sequences = [
        features[start:start + sequence_length]
        for start in range(0, len(features) - sequence_length + 1, stride)
    ]

    return np.stack(sequences).astype(np.float32)


def normalize_sequence(sequence):
    """
    Normalize each frame independently.

    Input/output shape: (sequence_length, 63).
    """
    from src.preprocessing import normalize_landmarks

    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2 or sequence.shape[1] != 63:
        raise ValueError(
            f"Expected shape (frames, 63), got {sequence.shape}"
        )

    return np.stack(
        [normalize_landmarks(frame) for frame in sequence]
    ).astype(np.float32)
