from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================
DATASET_DIR = Path("ISL_DATASET_LARGE")
MANIFEST_FILE = DATASET_DIR / "metadata.csv"

OUTPUT_DIR = Path("data")
OUTPUT_NPZ = OUTPUT_DIR / "isl_large_sequences.npz"
OUTPUT_CSV = OUTPUT_DIR / "isl_large_labels.csv"

SEQUENCE_LENGTH = 30


# =========================================================
# MEDIAPIPE
# =========================================================
mp_hands = mp.solutions.hands


# =========================================================
# LANDMARK NORMALIZATION
# =========================================================
def normalize_landmarks(hand_landmarks):
    """
    Convert 21 hand landmarks into 63 normalized values.

    Wrist becomes the origin and the coordinates are scaled
    by the maximum distance from the wrist.
    """

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    # Wrist = origin
    points = points - points[0]

    # Scale normalization
    distances = np.linalg.norm(points, axis=1)
    scale = np.max(distances)

    if scale > 1e-6:
        points = points / scale

    return points.flatten()


# =========================================================
# SAMPLE FRAME INDICES
# =========================================================
def sample_indices(total_frames, n):
    """
    Select n evenly spaced frames from a video.
    """

    if total_frames <= 0:
        return []

    if total_frames <= n:
        return np.linspace(
            0,
            total_frames - 1,
            n
        ).astype(int).tolist()

    return np.linspace(
        0,
        total_frames - 1,
        n
    ).astype(int).tolist()


# =========================================================
# EXTRACT ONE VIDEO
# =========================================================
def extract_video(video_path, hands):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return None

    indices = sample_indices(
        total_frames,
        SEQUENCE_LENGTH
    )

    features = []

    current_index = 0
    target_position = 0

    while target_position < len(indices):

        ret, frame = cap.read()

        if not ret:
            break

        if current_index == indices[target_position]:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = hands.process(rgb)

            if result.multi_hand_landmarks:

                # Use the first detected hand.
                hand = result.multi_hand_landmarks[0]

                feature_vector = normalize_landmarks(
                    hand
                )

            else:

                # No hand detected: zero vector.
                feature_vector = np.zeros(
                    63,
                    dtype=np.float32
                )

            features.append(feature_vector)

            target_position += 1

        current_index += 1

    cap.release()

    # Must contain exactly SEQUENCE_LENGTH frames.
    if len(features) != SEQUENCE_LENGTH:
        return None

    return np.array(
        features,
        dtype=np.float32
    )


# =========================================================
# MAIN
# =========================================================
def main():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = pd.read_csv(
        MANIFEST_FILE
    )

    required_columns = {
        "word",
        "local_path"
    }

    if not required_columns.issubset(
        metadata.columns
    ):
        raise ValueError(
            f"metadata.csv must contain "
            f"{required_columns}. "
            f"Found: {list(metadata.columns)}"
        )

    # Only process MP4 files.
    metadata = metadata[
        metadata["local_path"]
        .astype(str)
        .str.lower()
        .str.endswith(".mp4")
    ].copy()

    print("========================================")
    print("LARGE ISL SEQUENCE EXTRACTION")
    print("========================================")

    print(
        "Videos in manifest:",
        len(metadata)
    )

    print(
        "Sequence length:",
        SEQUENCE_LENGTH
    )

    print(
        "Features/frame: 63"
    )

    print()

    sequences = []
    labels = []
    videos = []

    processed = 0
    skipped = 0

    # -----------------------------------------------------
    # MediaPipe Hands
    # -----------------------------------------------------
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for _, row in metadata.iterrows():

            word = str(row["word"])
            video_path = Path(
                str(row["local_path"])
            )

            # If the manifest path is relative, resolve from
            # the project root.
            if not video_path.is_absolute():
                video_path = Path.cwd() / video_path

            print(
                f"[{processed + skipped + 1}/"
                f"{len(metadata)}] "
                f"{word}: "
                f"{video_path.name}"
            )

            sequence = extract_video(
                video_path,
                hands
            )

            if sequence is None:

                print("    SKIPPED")

                skipped += 1
                continue

            sequences.append(sequence)
            labels.append(word)
            videos.append(
                str(video_path)
            )

            processed += 1

    # -----------------------------------------------------
    # Verify extraction
    # -----------------------------------------------------
    if not sequences:
        raise RuntimeError(
            "No videos were successfully processed."
        )

    X = np.stack(
        sequences
    )

    y = np.array(
        labels
    )

    print()
    print("========================================")
    print("EXTRACTION COMPLETE")
    print("========================================")

    print(
        "Videos processed:",
        processed
    )

    print(
        "Videos skipped:",
        skipped
    )

    print(
        "Dataset shape:",
        X.shape
    )

    print(
        "Expected:",
        f"(videos, {SEQUENCE_LENGTH}, 63)"
    )

    print()
    print("Class counts:")

    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y
    )

    labels_output = pd.DataFrame({
        "label": y,
        "video": videos
    })

    labels_output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("Saved:")
    print(OUTPUT_NPZ)

    print(OUTPUT_CSV)

    print()
    print("NEXT STEP:")
    print(
        "Do NOT train yet. "
        "Send the complete output first."
    )


if __name__ == "__main__":
    main()
