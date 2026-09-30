from pathlib import Path
import csv
import cv2
import mediapipe as mp
import numpy as np

DATASET = Path("ISL_DATASET")
OUTPUT_NPZ = Path("data/isl_sequences.npz")
OUTPUT_CSV = Path("data/isl_labels.csv")

WORDS = ["eat", "go", "hello", "help", "no", "please", "water", "yes"]
SEQUENCE_LENGTH = 20
MAX_VIDEOS_PER_CLASS = 50


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


def sample_indices(total_frames, n):
    if total_frames <= 0:
        return []
    return np.linspace(0, total_frames - 1, n).astype(int)


def extract_video(video_path, hands):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return None

    indices = set(sample_indices(total, SEQUENCE_LENGTH))
    features = []

    for i in range(total):
        ok, frame = cap.read()

        if not ok:
            break

        if i not in indices:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            features.append(
                normalize_landmarks(result.multi_hand_landmarks[0])
            )
        else:
            features.append(np.zeros(63, dtype=np.float32))

    cap.release()

    if len(features) != SEQUENCE_LENGTH:
        return None

    return np.asarray(features, dtype=np.float32)


def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            "ISL_DATASET was not found. Make sure the dataset is in the project root."
        )

    videos = []

    for word in WORDS:
        folder = DATASET / word
        found = sorted(folder.glob("*.mp4"))[:MAX_VIDEOS_PER_CLASS]

        print(f"{word:8s}: {len(found)} videos")
        videos.extend((word, p) for p in found)

    X = []
    y = []
    rows = []

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        for number, (word, video) in enumerate(videos, 1):
            print(f"[{number}/{len(videos)}] {word}: {video.name}")

            sequence = extract_video(video, hands)

            if sequence is None:
                print("    SKIPPED: could not extract 20 frames")
                continue

            X.append(sequence)
            y.append(word)
            rows.append([word, str(video)])

    if not X:
        raise RuntimeError("No usable video sequences were extracted.")

    X = np.stack(X)
    y = np.asarray(y)

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["label", "video"])
        writer.writerows(rows)

    print("\n================================")
    print("ISL SEQUENCE EXTRACTION COMPLETE")
    print("================================")
    print(f"Videos processed: {len(X)}")
    print(f"Dataset shape: {X.shape}")
    print("Expected shape: (videos, 20, 63)")
    print(f"Saved: {OUTPUT_NPZ}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
