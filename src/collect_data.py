import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp

from hand_features import extract_features

LABELS = [
    "Open_Palm",
    "Closed_Fist",
    "Thumb_Up",
    "Victory",
    "Pointing_Up",
]

SAMPLES_PER_CLASS = 250
OUTPUT_FILE = Path("data/landmarks.csv")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def ensure_csv():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["label"] + [f"f{i}" for i in range(63)]
            )


def collect_class(label, hands):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    captured = 0
    collecting = False
    last_capture = 0.0

    print(f"\nClass: {label}")
    print("Position your hand. Press SPACE to start/collect samples.")
    print("Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand = (
            results.multi_hand_landmarks[0]
            if results.multi_hand_landmarks
            else None
        )

        if hand:
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
            )

        if not collecting:
            message = "SPACE = START"
        else:
            message = f"COLLECTING {captured}/{SAMPLES_PER_CLASS}"

        cv2.putText(
            frame,
            f"Class: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            message,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        if key == ord(" "):
            collecting = True

        if (
            collecting
            and hand is not None
            and captured < SAMPLES_PER_CLASS
            and time.time() - last_capture > 0.06
        ):
            features = extract_features(hand)

            with OUTPUT_FILE.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([label] + features)

            captured += 1
            last_capture = time.time()

        if captured >= SAMPLES_PER_CLASS:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ensure_csv()

    print("======================================")
    print("  HAND GESTURE DATA COLLECTION")
    print("======================================")
    print("Use ONE hand and keep the same hand throughout.")
    print("The five labels are generic gestures for pipeline testing.")
    print()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        for label in LABELS:
            collect_class(label, hands)

    print("\nCollection complete.")
    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
