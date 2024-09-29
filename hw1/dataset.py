import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class SlakhDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Custom Dataset for loading Slakh data.

        Args:
            root_dir (str): Root directory of the dataset.
            split (str): One of 'train', 'validation', or 'test'.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Paths to data and label files
        self.data_dir = os.path.join(self.root_dir, self.split)
        labels_file = os.path.join(self.root_dir, f"{self.split}_labels.json")

        # Load labels
        with open(labels_file, "r") as f:
            self.labels_dict = json.load(f)

        # List of all data filenames
        self.data_files = list(self.labels_dict.keys())

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Ensure idx is an integer
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get filename
        data_file = self.data_files[idx]
        data_path = os.path.join(self.data_dir, data_file)

        # Load data
        data = np.load(data_path)  # Shape depends on how the audio is stored

        # Load corresponding labels
        labels = np.array(self.labels_dict[data_file], dtype=np.float32)

        # Apply any data transformations
        if self.transform:
            data = self.transform(data)

        # Convert data and labels to tensors
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels)

        return data, labels
