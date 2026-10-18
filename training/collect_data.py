import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow imports from the SignSpeak-AI project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import Camera
from src.hand_detection import HandDetector
from src.preprocessing import extract_hand_landmarks


# Collection settings
LABELS = list("abcdefghijklmnopqrstuvwxyz")
SEQUENCE_LENGTH = 30
SAMPLES_PER_LABEL = 10

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "alphabet"


def count_samples(label):
    """Count saved sequences for a label."""
    label_dir = DATA_DIR / label
    if not label_dir.exists():
        return 0

    return len(list(label_dir.glob("sequence_*.npy")))


def record_sequence(camera, detector, label):
    """Capture 30 consecutive frames of one hand."""
    sequence = []

    print(f"\nRecording sign '{label.upper()}'...")
    print("Keep your hand visible and perform the sign.")

    for frame_number in range(SEQUENCE_LENGTH):
        frame = camera.read_frame()

        if frame is None:
            print("Camera read failed. Sample cancelled.")
            return None

        display_frame, results = detector.detect_hands(frame)

        if not results.multi_hand_landmarks:
            print("Hand lost. Sample cancelled; please try again.")
            return None

        features = extract_hand_landmarks(results)

        if features.shape != (63,) or not np.isfinite(features).all():
            print("Invalid landmarks. Sample cancelled.")
            return None

        sequence.append(features)

        cv2.putText(
            display_frame,
            f"Recording {label.upper()}: "
            f"{frame_number + 1}/{SEQUENCE_LENGTH}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("SignSpeak AI - Data Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("Recording cancelled.")
            return None

    return np.asarray(sequence, dtype=np.float32)


def save_sequence(label, sequence):
    """Save a sequence without overwriting existing samples."""
    label_dir = DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    sample_number = count_samples(label)
    sample_path = label_dir / f"sequence_{sample_number:03d}.npy"

    np.save(sample_path, sequence)
    print(f"Saved: {sample_path.relative_to(PROJECT_ROOT)}")


def main():
    camera = None
    detector = None
    active_label = None

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        camera = Camera()
        detector = HandDetector(max_num_hands=1)

        print("\n=== SignSpeak AI: Alphabet Sequence Collector ===")
        print("Labels: A-Z")
        print("30 frames per sample | 30 samples per sign")
        print("\nControls:")
        print("  Press A-Z to select a sign.")
        print("  Press SPACE to record one sample.")
        print("  Press Q or ESC to quit.\n")

        while True:
            frame = camera.read_frame()

            if frame is None:
                print("Could not read webcam frame.")
                break

            display_frame, results = detector.detect_hands(frame)

            hand_visible = bool(results.multi_hand_landmarks)

            if active_label is None:
                instruction = "Press A-Z to select a sign"
            else:
                saved = count_samples(active_label)
                instruction = (
                    f"Sign: {active_label.upper()} | "
                    f"Samples: {saved}/{SAMPLES_PER_LABEL} | "
                    "SPACE = record"
                )

            cv2.putText(
                display_frame,
                instruction,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if hand_visible else (0, 165, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Hand detected: {'Yes' if hand_visible else 'No'}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if hand_visible else (0, 0, 255),
                2,
            )

            cv2.imshow("SignSpeak AI - Data Collection", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if ord("a") <= key <= ord("z"):
                selected = chr(key)

                if count_samples(selected) >= SAMPLES_PER_LABEL:
                    print(
                        f"\n{selected.upper()} already has "
                        f"{SAMPLES_PER_LABEL} samples."
                    )
                    active_label = selected
                    continue

                active_label = selected
                print(
                    f"\nSelected {active_label.upper()}: "
                    f"{count_samples(active_label)}/{SAMPLES_PER_LABEL} samples"
                )

            elif key == ord(" "):
                if active_label is None:
                    print("Select a label first by pressing A-Z.")
                    continue

                if count_samples(active_label) >= SAMPLES_PER_LABEL:
                    print("This label already has 30 samples.")
                    continue

                sequence = record_sequence(
                    camera, detector, active_label
                )

                if sequence is not None:
                    if sequence.shape != (30, 63):
                        print("Unexpected shape. Sample not saved.")
                    else:
                        save_sequence(active_label, sequence)

                        saved = count_samples(active_label)
                        print(
                            f"{active_label.upper()} progress: "
                            f"{saved}/{SAMPLES_PER_LABEL}"
                        )

                        if saved >= SAMPLES_PER_LABEL:
                            print(
                                f"Finished collecting "
                                f"{active_label.upper()}!"
                            )

                time.sleep(0.3)

        print("\nCollection session ended.")

    except Exception as error:
        print(f"Collection failed: {error}")
        raise

    finally:
        if camera is not None:
            camera.release()

        if detector is not None:
            detector.close()

        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()