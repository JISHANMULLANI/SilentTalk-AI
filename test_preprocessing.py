
import numpy as np

from src.camera import Camera
from src.hand_detection import HandDetector
from src.preprocessing import extract_hand_landmarks


def main():
    camera = None
    detector = None

    try:
        camera = Camera()
        detector = HandDetector(max_num_hands=1)

        print("SignSpeak AI: Feature extraction test started.")
        print("Show one hand to the camera. Press Q or ESC to exit.")

        while True:
            frame = camera.read_frame()

            if frame is None:
                print("Could not read webcam frame.")
                break

            display_frame, results = detector.detect_hands(frame)

            features = extract_hand_landmarks(results)

            if results.multi_hand_landmarks:
                print(
                    f"Hand detected | Shape: {features.shape} | "
                    f"Data type: {features.dtype} | "
                    f"Finite values: {np.isfinite(features).all()}"
                )
            else:
                print("No hand detected | Feature shape:", features.shape)

            import cv2

            cv2.imshow("SignSpeak AI - Feature Test", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except Exception as error:
        print("Test failed:", error)

    finally:
        if camera is not None:
            camera.release()

        if detector is not None:
            detector.close()

        import cv2
        cv2.destroyAllWindows()
        print("Feature extraction test ended.")


if __name__ == "__main__":
    main()
