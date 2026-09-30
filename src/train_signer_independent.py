from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# =========================================================
# PATHS
# =========================================================
SEQUENCE_FILE = Path("data/isl_sequences.npz")
LABEL_FILE = Path("data/isl_labels.csv")
METADATA_FILE = Path("ISL_DATASET/metadata.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.25

# =========================================================
# LOAD DATA
# =========================================================
data = np.load(SEQUENCE_FILE, allow_pickle=True)
X = data["X"]

labels_df = pd.read_csv(LABEL_FILE)
metadata_df = pd.read_csv(METADATA_FILE)

print("Sequence shape:", X.shape)
print("Label rows:", len(labels_df))
print("Metadata rows:", len(metadata_df))

# =========================================================
# CHECK FILE FORMAT
# =========================================================
required_labels = {"label", "video"}
if not required_labels.issubset(labels_df.columns):
    raise ValueError(
        f"Expected {required_labels} in isl_labels.csv, "
        f"found {list(labels_df.columns)}"
    )

required_metadata = {"video_path", "original_filename", "signer", "dataset"}
if not required_metadata.issubset(metadata_df.columns):
    raise ValueError(
        f"metadata.csv is missing one of: {required_metadata}"
    )

if len(X) != len(labels_df):
    raise ValueError(
        f"Sequence/label mismatch: {len(X)} sequences vs "
        f"{len(labels_df)} labels."
    )

# =========================================================
# MATCH BY UNIQUE FILENAME
#
# isl_labels.csv:
# ISL_DATASET\eat\eat__CISLR__00016__Cgs1MibMXJ0.mp4
#
# metadata.csv:
# original_filename = Cgs1MibMXJ0.mp4
#
# But the metadata video_path contains the complete dataset
# filename, so the safest match is the basename of video_path.
# =========================================================
def basename(path):
    return str(path).replace("\\", "/").split("/")[-1]

labels_df["filename"] = labels_df["video"].map(basename)
metadata_df["filename"] = metadata_df["video_path"].map(basename)

# Check that metadata filenames are unique.
duplicate_metadata = metadata_df[
    metadata_df["filename"].duplicated(keep=False)
]

if len(duplicate_metadata) > 0:
    print("\nWarning: duplicate metadata filenames detected.")
    print(duplicate_metadata[["video_path", "original_filename"]].head(10))

# Build lookup using filename.
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
            f"Could not find metadata for filename:\n{filename}"
        )

    row = metadata_lookup.loc[filename]

    if pd.isna(row["signer"]):
        raise ValueError(
            f"Missing signer information for:\n{filename}"
        )

    signers.append(str(row["signer"]))
    datasets.append(str(row["dataset"]))

groups = np.array(signers)
datasets = np.array(datasets)
y = labels_df["label"].astype(str).to_numpy()

# =========================================================
# DATASET INFORMATION
# =========================================================
print("\n================================")
print("DATASET INFORMATION")
print("================================")

print("\nWord counts:")
print(pd.Series(y).value_counts().sort_index())

print("\nDataset counts:")
print(pd.Series(datasets).value_counts().sort_index())

print("\nSigner counts:")
print(pd.Series(groups).value_counts().sort_index())

# =========================================================
# SIGNER-INDEPENDENT SPLIT
# =========================================================
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

train_signers = sorted(set(groups[train_idx]))
test_signers = sorted(set(groups[test_idx]))

print("\n================================")
print("SIGNER-INDEPENDENT SPLIT")
print("================================")

print("Train videos :", len(train_idx))
print("Test videos  :", len(test_idx))
print("Train signers:", train_signers)
print("Test signers :", test_signers)

overlap = set(train_signers) & set(test_signers)

print("Signer overlap:", overlap)

if overlap:
    raise RuntimeError(
        f"DATA LEAKAGE DETECTED: {overlap}"
    )

# =========================================================
# CLASS DISTRIBUTION
# =========================================================
print("\nTraining classes:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTesting classes:")
print(pd.Series(y_test).value_counts().sort_index())

# =========================================================
# RANDOM FOREST BASELINE
# =========================================================
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("\nTraining shape:", X_train_flat.shape)
print("Testing shape :", X_test_flat.shape)

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train_flat, y_train)

# =========================================================
# EVALUATION
# =========================================================
y_pred = model.predict(X_test_flat)

accuracy = accuracy_score(y_test, y_pred)
classes = sorted(np.unique(y))

print("\n================================")
print("SIGNER-INDEPENDENT RESULTS")
print("================================")

print(f"Accuracy     : {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=classes,
        zero_division=0
    )
)

print("Confusion matrix:")
print("Classes:", classes)
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )
)

# =========================================================
# SAVE MODEL
# =========================================================
output_model = Path(
    "models/isl_random_forest_signer_independent.joblib"
)

output_model.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, output_model)

print("\nModel saved to:")
print(output_model)

# =========================================================
# SAVE EXPERIMENT SPLIT
# =========================================================
split = np.full(len(y), "test", dtype=object)
split[train_idx] = "train"

split_df = pd.DataFrame({
    "video": labels_df["video"],
    "label": y,
    "dataset": datasets,
    "signer": groups,
    "split": split
})

split_df.to_csv(
    "data/signer_independent_split.csv",
    index=False
)

print("\nSplit information saved to:")
print("data/signer_independent_split.csv")

print("\nExperiment complete.")