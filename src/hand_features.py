import numpy as np


def extract_features(hand_landmarks):
    """Convert 21 MediaPipe landmarks into a normalized 63-value vector.

    We translate the hand so the wrist is the origin and scale by the
    maximum landmark distance from the wrist. This makes the classifier
    less sensitive to where the hand appears in the camera and to hand size.
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    # Wrist becomes origin.
    points = points - points[0]

    # Scale normalization.
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-6:
        points = points / scale

    return points.flatten().tolist()
