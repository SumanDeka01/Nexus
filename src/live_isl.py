from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


MODEL = Path("models/isl_random_forest.joblib")

SEQUENCE_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.45


if not MODEL.exists():
    raise FileNotFoundError(
        "Model not found.\n"
        "Run: python src/train_isl_baseline.py"
    )


model = joblib.load(MODEL)

buffer = deque(maxlen=SEQUENCE_LENGTH)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def normalize_landmarks(hand):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand.landmark],
        dtype=np.float32,
    )

    points -= points[0]

    scale = np.max(np.linalg.norm(points, axis=1))

    if scale > 1e-6:
        points /= scale

    return points.flatten()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        result = hands.process(rgb)

        label = "Show one hand"
        confidence = 0.0

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
            )

            buffer.append(
                normalize_landmarks(hand)
            )

            if len(buffer) == SEQUENCE_LENGTH:

                X = np.asarray(
                    buffer,
                    dtype=np.float32,
                ).reshape(1, -1)

                probabilities = model.predict_proba(X)[0]

                index = int(
                    np.argmax(probabilities)
                )

                label = model.classes_[index]

                confidence = float(
                    probabilities[index]
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    label = "Uncertain"

        else:
            buffer.clear()

        # UI
        cv2.rectangle(
            frame,
            (10, 10),
            (600, 115),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"ISL: {label}",
            (25, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Hold a sign | Q = quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "ISL Recognition - Baseline",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
