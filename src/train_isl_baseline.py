from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


DATA = Path("data/isl_sequences.npz")
MODEL = Path("models/isl_random_forest.joblib")


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            "data/isl_sequences.npz not found.\n"
            "Run: python src/extract_isl_sequences.py"
        )

    data = np.load(DATA)

    X = data["X"]
    y = data["y"]

    print("Original shape:", X.shape)

    # 20 frames × 63 features = 1260 features/video
    X = X.reshape(len(X), -1)

    print("Model input shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n================================")
    print("REAL ISL BASELINE RESULTS")
    print("================================")
    print(f"Total videos : {len(y)}")
    print(f"Train videos : {len(y_train)}")
    print(f"Test videos  : {len(y_test)}")
    print(f"Classes      : {sorted(set(y))}")
    print(f"Accuracy     : {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL)

    print(f"\nModel saved to:")
    print(MODEL)


if __name__ == "__main__":
    main()
