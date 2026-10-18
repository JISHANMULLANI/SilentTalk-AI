
import cv2

from src.camera import Camera
from src.hand_detection import HandDetector


def main():
    camera = None
    detector = None

    try:
        camera = Camera(camera_index=0)
        detector = HandDetector(max_num_hands=2)

        print("SignSpeak AI: Webcam started.")
        print("Press Q or ESC to exit.")

        while True:
            frame = camera.read_frame()

            if frame is None:
                print("Failed to read webcam frame.")
                break

            frame, results = detector.detect_hands(frame)

            # Display detection status and landmark count.
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)

                cv2.putText(
                    frame,
                    f"Hands detected: {hand_count}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                for index, hand_landmarks in enumerate(
                    results.multi_hand_landmarks, start=1
                ):
                    count = len(hand_landmarks.landmark)

                    print(
                        f"\rHand {index}: {count}/21 landmarks",
                        end="",
                        flush=True,
                    )
            else:
                cv2.putText(
                    frame,
                    "Show your hand to the camera",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("SignSpeak AI - Hand Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except (RuntimeError, ImportError) as error:
        print(f"Error: {error}")

    finally:
        if detector is not None:
            detector.close()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()
        print("\nWebcam closed.")


if __name__ == "__main__":
    main()
