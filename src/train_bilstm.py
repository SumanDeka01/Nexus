from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =========================================================
# SETTINGS
# =========================================================
SEQUENCE_FILE = Path("data/isl_sequences.npz")
LABEL_FILE = Path("data/isl_labels.csv")
METADATA_FILE = Path("ISL_DATASET/metadata.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.25

EPOCHS = 80
BATCH_SIZE = 8

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# =========================================================
# LOAD DATA
# =========================================================
data = np.load(SEQUENCE_FILE, allow_pickle=True)
X = data["X"].astype(np.float32)

labels_df = pd.read_csv(LABEL_FILE)
metadata_df = pd.read_csv(METADATA_FILE)

print("Sequence shape:", X.shape)
print("Label rows:", len(labels_df))
print("Metadata rows:", len(metadata_df))

if not {"label", "video"}.issubset(labels_df.columns):
    raise ValueError(
        f"Expected label/video columns. Found: {list(labels_df.columns)}"
    )

if not {"video_path", "signer", "dataset"}.issubset(metadata_df.columns):
    raise ValueError(
        "metadata.csv must contain video_path, signer and dataset."
    )

if len(X) != len(labels_df):
    raise ValueError(
        f"Sequence/label mismatch: {len(X)} vs {len(labels_df)}"
    )


# =========================================================
# MATCH VIDEOS TO METADATA BY FILENAME
# =========================================================
def basename(path):
    return str(path).replace("\\", "/").split("/")[-1]


labels_df["filename"] = labels_df["video"].map(basename)
metadata_df["filename"] = metadata_df["video_path"].map(basename)

metadata_lookup = (
    metadata_df
    .drop_duplicates("filename")
    .set_index("filename")
)

signers = []
datasets = []

for filename in labels_df["filename"]:
    if filename not in metadata_lookup.index:
        raise ValueError(
            f"Could not find metadata for filename: {filename}"
        )

    row = metadata_lookup.loc[filename]

    if pd.isna(row["signer"]):
        raise ValueError(
            f"Missing signer for filename: {filename}"
        )

    signers.append(str(row["signer"]))
    datasets.append(str(row["dataset"]))

groups = np.array(signers)
datasets = np.array(datasets)
y_text = labels_df["label"].astype(str).to_numpy()


# =========================================================
# SIGNER-INDEPENDENT SPLIT
# Same split strategy as the RF experiment
# =========================================================
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y_text, groups=groups)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train_text = y_text[train_idx]
y_test_text = y_text[test_idx]

train_signers = sorted(set(groups[train_idx]))
test_signers = sorted(set(groups[test_idx]))

print("\n================================")
print("SIGNER-INDEPENDENT SPLIT")
print("================================")
print("Train videos :", len(train_idx))
print("Test videos  :", len(test_idx))
print("Train signers:", train_signers)
print("Test signers :", test_signers)
print("Signer overlap:", set(train_signers) & set(test_signers))

if set(train_signers) & set(test_signers):
    raise RuntimeError("Signer leakage detected.")


# =========================================================
# LABEL ENCODING
# =========================================================
encoder = LabelEncoder()

# Fit on all known classes so class indices remain consistent.
encoder.fit(y_text)

y_train = encoder.transform(y_train_text)
y_test = encoder.transform(y_test_text)

num_classes = len(encoder.classes_)

print("\nClasses:")
print(list(encoder.classes_))

print("\nTraining class counts:")
print(pd.Series(y_train_text).value_counts().sort_index())

print("\nTesting class counts:")
print(pd.Series(y_test_text).value_counts().sort_index())


# =========================================================
# BUILD BiLSTM
# =========================================================
timesteps = X.shape[1]
features = X.shape[2]

model = Sequential([
    Input(shape=(timesteps, features)),

    Bidirectional(
        LSTM(
            64,
            return_sequences=False,
            dropout=0.25,
            recurrent_dropout=0.0
        )
    ),

    Dense(64, activation="relu"),
    Dropout(0.35),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n================================")
print("BiLSTM MODEL")
print("================================")

model.summary()


# =========================================================
# CALLBACKS
# =========================================================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)


# =========================================================
# TRAIN
# IMPORTANT:
# This uses a validation split from TRAINING data only.
# The signer-independent TEST set remains untouched.
# =========================================================
print("\nTraining BiLSTM...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# =========================================================
# TEST
# =========================================================
test_probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(test_probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("BiLSTM SIGNER-INDEPENDENT RESULTS")
print("================================")

print(f"Accuracy     : {accuracy * 100:.2f}%")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("Confusion matrix:")
print("Classes:", list(encoder.classes_))

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(num_classes)
    )
)


# =========================================================
# SAVE MODEL + LABEL ENCODER
# =========================================================
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "isl_bilstm_signer_independent.keras"
encoder_path = models_dir / "isl_bilstm_label_encoder.npy"

model.save(model_path)
np.save(encoder_path, encoder.classes_)

print("\nModel saved to:")
print(model_path)

print("\nLabel classes saved to:")
print(encoder_path)


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================
history_df = pd.DataFrame(history.history)
history_df.to_csv(
    "data/bilstm_training_history.csv",
    index=False
)

print("\nTraining history saved to:")
print("data/bilstm_training_history.csv")

print("\nExperiment complete.")