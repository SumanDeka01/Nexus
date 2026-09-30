from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from hand_features import extract_features


MODEL_FILE = Path("models/sign_model.joblib")
CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_WINDOW = 7

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "models/sign_model.joblib not found. Run train_model.py first."
    )

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
classes = bundle["classes"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

history = deque(maxlen=SMOOTHING_WINDOW)

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label = "No hand"
        confidence = 0.0

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
            )

            features = np.array(
                extract_features(hand),
                dtype=np.float32,
            ).reshape(1, -1)

            probabilities = model.predict_proba(features)[0]
            index = int(np.argmax(probabilities))

            confidence = float(probabilities[index])
            predicted = classes[index]

            if confidence >= CONFIDENCE_THRESHOLD:
                history.append(predicted)

                # Majority vote over recent predictions.
                if history:
                    label = max(set(history), key=history.count)
            else:
                history.clear()
                label = "Uncertain"

        else:
            history.clear()

        cv2.rectangle(
            frame,
            (10, 10),
            (520, 100),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"Prediction: {label}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (25, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Q = quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Sign Recognition Prototype", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
