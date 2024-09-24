import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import load_labels, extract_features

BASE_PATH = Path("/home/ndrw1221/nas/nsynth-dataset")
TRAIN_PATH = BASE_PATH / "nsynth-subtrain"
VALID_PATH = BASE_PATH / "nsynth-valid"
TEST_PATH = BASE_PATH / "nsynth-test"
FEATURE_VERSION = "v1"

# Load labels and data
train_labels, train_data = load_labels(f"{TRAIN_PATH}/examples.json")
valid_labels, valid_data = load_labels(f"{VALID_PATH}/examples.json")
test_labels, test_data = load_labels(f"{TEST_PATH}/examples.json")

# Assert train, valid, and test labels does not overlap
assert len(set(train_labels.keys()) & set(valid_labels.keys())) == 0
assert len(set(train_labels.keys()) & set(test_labels.keys())) == 0
assert len(set(valid_labels.keys()) & set(test_labels.keys())) == 0


def process_dataset(audio_dir, labels):
    features = []
    targets = []

    # Collect all filenames in advance
    filenames = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    # Use tqdm to wrap the filenames iterator
    for filename in tqdm(
        filenames, desc=f"Processing {audio_dir.parent.name}/{audio_dir.name}"
    ):
        file_id = filename[:-4]  # Remove '.wav' extension
        if file_id in labels:
            audio_path = os.path.join(audio_dir, filename)
            mfccs = extract_features(audio_path)
            features.append(mfccs)
            targets.append(labels[file_id])
    return np.array(features), np.array(targets)


# Process datasets with progress bars
X_train, y_train = process_dataset(TRAIN_PATH / "audio", train_labels)
X_valid, y_valid = process_dataset(VALID_PATH / "audio", valid_labels)
X_test, y_test = process_dataset(TEST_PATH / "audio", test_labels)


# Save the processed features and labels
np.save(f"features/X_train_{FEATURE_VERSION}.npy", X_train)
np.save(f"features/y_train_{FEATURE_VERSION}.npy", y_train)
np.save(f"features/X_valid_{FEATURE_VERSION}.npy", X_valid)
np.save(f"features/y_valid_{FEATURE_VERSION}.npy", y_valid)
np.save(f"features/X_test_{FEATURE_VERSION}.npy", X_test)
np.save(f"features/y_test_{FEATURE_VERSION}.npy", y_test)
