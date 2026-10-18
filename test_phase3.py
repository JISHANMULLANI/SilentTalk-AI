
from pathlib import Path

import numpy as np

from src.preprocessing import normalize_landmarks
from src.sequence import generate_sequences, normalize_sequence


def main():
    # Test one frame.
    frame = np.random.rand(63).astype(np.float32)
    normalized = normalize_landmarks(frame)

    assert normalized.shape == (63,)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()

    # Test one complete sequence.
    sequence = np.random.rand(30, 63).astype(np.float32)
    normalized_seq = normalize_sequence(sequence)

    assert normalized_seq.shape == (30, 63)
    assert np.isfinite(normalized_seq).all()

    # Test generation from 60 frames.
    frames = np.random.rand(60, 63).astype(np.float32)
    sequences = generate_sequences(
        frames,
        sequence_length=30,
        stride=30,
    )

    assert sequences.shape == (2, 30, 63)

    print("Frame normalization: PASS")
    print("Sequence normalization: PASS")
    print("Sequence generation: PASS")
    print("Generated shape:", sequences.shape)

    # Check a real collected sample if one exists.
    sample_path = Path("data/raw/alphabet/a/sequence_000.npy")

    if sample_path.exists():
        sample = np.load(sample_path)

        assert sample.shape == (30, 63), (
            f"Unexpected sample shape: {sample.shape}"
        )

        normalized_sample = normalize_sequence(sample)

        print("Real sample:", sample.shape)
        print("Normalized sample:", normalized_sample.shape)
        print("Real sample check: PASS")

    else:
        print("No real sample found yet; synthetic tests passed.")


if __name__ == "__main__":
    main()
