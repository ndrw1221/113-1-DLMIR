import numpy as np
import json
import librosa


# Function to extract MFCC features
def extract_features(audio_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Take mean along the time axis
    return mfccs_mean


def load_labels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    labels = {}
    for key, value in data.items():
        labels[key] = value["instrument_family"]
    return labels, data  # Returning data as well for instrument_family


def create_label_mapping(data):
    idx_to_family = {}
    for key, value in data.items():
        idx = value["instrument_family"]
        family_str = value["instrument_family_str"]
        idx_to_family[idx] = family_str
    return idx_to_family
