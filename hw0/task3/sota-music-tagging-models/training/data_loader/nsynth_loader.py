# coding: utf-8
import os
import numpy as np
from torch.utils import data


class AudioFolder(data.Dataset):
    def __init__(self, root, split, input_length=None):
        self.root = root
        self.split = split
        self.input_length = input_length
        self.get_songlist()
        self.label = np.load(
            os.path.join(self.root, "nsynth", self.split.lower(), "label.npy")
        )

    def __getitem__(self, index):
        npy, label = self.get_npy(index)
        return npy.astype("float32"), label.astype("float32")

    def get_songlist(self):
        if self.split == "TRAIN":
            self.fl = np.load("./../split/nsynth/train.npy")
        elif self.split == "VALID":
            self.fl = np.load("./../split/nsynth/valid.npy")
        elif self.split == "TEST":
            self.fl = np.load("./../split/nsynth/test.npy")
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")

    def get_npy(self, index):
        # npy_path = os.path.join(self.root, "nsynth", self.split.lower(), "audio.npy")
        # npy = np.load(npy_path, mmap_mode="r")[index]
        # npy = np.load(npy_path)[index]

        # # Ensure the audio length is sufficient
        # if len(npy) < self.input_length:
        #     # Pad the audio if it's too short
        #     npy = np.pad(npy, (0, self.input_length - len(npy)), "constant")

        # # Randomly sample a chunk
        # random_idx = np.random.randint(0, len(npy) - self.input_length + 1)
        # npy = npy[random_idx : random_idx + self.input_length]

        # label = self.label[index]
        # return npy, label

        ix, fn = self.fl[index].split("\t")
        npy_path = (
            os.path.join(self.root, "nsynth", "npy", fn.split("/")[-1][:-3]) + "npy"
        )
        npy = np.load(npy_path, mmap_mode="r")
        random_idx = int(np.floor(np.random.random(1) * (len(npy) - self.input_length)))
        npy = np.array(npy[random_idx : random_idx + self.input_length])
        label = self.label[int(ix)]
        return npy, label

    def __len__(self):
        return len(self.fl)


def get_audio_loader(root, batch_size, split="TRAIN", num_workers=0, input_length=None):
    data_loader = data.DataLoader(
        dataset=AudioFolder(root, split=split, input_length=input_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return data_loader
