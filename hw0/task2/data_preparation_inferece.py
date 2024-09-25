import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import load_labels, extract_features

BASE_PATH = Path("path/to/your/nsynth-dataset")  # Correct the path
TEST_PATH = BASE_PATH / "nsynth-test"
FEATURE_VERSION = "v1"

# Load labels and data
test_labels, test_data = load_labels(f"{TEST_PATH}/examples.json")


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
X_test, y_test = process_dataset(TEST_PATH / "audio", test_labels)

os.makedirs("features", exist_ok=True)

# Save the processed features and labels
np.save(f"features/X_test_{FEATURE_VERSION}.npy", X_test)
np.save(f"features/y_test_{FEATURE_VERSION}.npy", y_test)
