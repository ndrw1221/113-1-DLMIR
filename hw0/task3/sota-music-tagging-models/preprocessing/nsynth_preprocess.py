import os
import json
import numpy as np
import tqdm
import argparse


def main(config):
    split = config.split
    data_path = config.data_path

    audio_split = []
    # audios = []
    # labels = []

    with open(os.path.join(config.data_path, "examples.json")) as f:
        examples = json.load(f)

    for idx, key in enumerate(tqdm.tqdm(examples.keys())):
        audio_wav = os.path.join(config.data_path, "audio", key + ".wav")
        # audio_npy = np.load(os.path.join(config.data_path, "npy", key + ".npy"))
        # label = examples[key]["instrument_family"]

        audio_split.append(f"{idx}\t{audio_wav}")
        # audios.append(audio_npy)
        # labels.append(label)

    np.save(os.path.join(f"./../split/nsynth/{split}.npy"), audio_split)
    # np.save(os.path.join(f"./../training/data/nsynth/{split}/audio.npy"), audios)
    # np.save(os.path.join(f"./../training/data/nsynth/{split}/label.npy"), labels)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, required=True)
    args.add_argument(
        "--split", type=str, choices=["train", "valid", "test"], required=True
    )
    config = args.parse_args()
    main(config)
