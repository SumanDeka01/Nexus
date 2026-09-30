from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


DATA_FILE = Path("data/landmarks.csv")
MODEL_FILE = Path("models/sign_model.joblib")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "data/landmarks.csv not found. Run collect_data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        raise RuntimeError("Dataset is empty.")

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")
    print(f"Samples: {len(df)}")
    print(f"Classes: {sorted(y.unique())}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "classes": sorted(y.unique().tolist()),
        },
        MODEL_FILE,
    )

    print(f"\nSaved model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
