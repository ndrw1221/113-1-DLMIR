import os
import json
import torchaudio
import torch
import museval
import tqdm
import argparse
import numpy as np
from openunmix import model, transforms
from torch.utils.data import Dataset, DataLoader

import pdb


class UnmixDatasetTestSplit(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.files = os.listdir(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mixture_path = os.path.join(self.root, self.files[idx], "mixture.wav")
        vocals_path = os.path.join(self.root, self.files[idx], "vocals.wav")
        mixture_wav, _ = torchaudio.load(mixture_path)
        vocals_wav, _ = torchaudio.load(vocals_path)
        track_name = self.files[idx]
        return mixture_wav, vocals_wav, track_name


def separate(audio, target, model_path, encoder, decoder, device) -> dict:
    state = torch.load(
        model_path,
        weights_only=True,
        map_location=device,
    )
    unmix = model.OpenUnmix(
        nb_bins=4096 // 2 + 1,
        nb_channels=2,
        hidden_size=512,
        max_bin=state["input_mean"].shape[0],
    ).to(device)
    unmix.load_state_dict(state, strict=False)
    unmix.freeze()

    stft = encoder(audio)
    estimates = unmix(stft)
    estimates = decoder(estimates.squeeze(0))

    return estimates


def save_audio(audio, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(f"{output_dir}/vocals.wav", audio, 44100)


def compute_sdr(estimates, target):
    estimates = estimates.permute(0, 2, 1).cpu().detach().numpy()
    target = target.permute(0, 2, 1).cpu().detach().numpy()
    SDR, _, _, _ = museval.evaluate(target, estimates)
    return np.nanmedian(SDR[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=137)
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    epoch = args.epoch
    print(f"Using epoch: {epoch}")
    # test_split_path = "/home/ndrw1221/nas/datasets/musdb18hq-dataset/test"
    test_split_path = args.root
    model_path = f"../scripts/open-unmix/vocals_epoch_{epoch}.pth"
    save_dir = f"results-griffinlim/epoch_{epoch}"
    os.makedirs(save_dir, exist_ok=True)

    dataset = UnmixDatasetTestSplit(test_split_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stft, _ = transforms.make_filterbanks(n_fft=4096, n_hop=1024, sample_rate=44100)
    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=2 == 1)).to(device)

    griffinlim = torchaudio.transforms.GriffinLim(
        n_fft=4096, win_length=4096, hop_length=1024
    ).to(device)

    sdr_results = {}
    # i = 0
    pbar = tqdm.tqdm(data_loader)
    for mixture, vocals, name in pbar:
        # if i > 2:
        #     break
        name = name[0]
        mixture = mixture.to(device)
        estimates = separate(
            mixture,
            "vocals",
            model_path,
            encoder,
            griffinlim,
            device,
        )
        # pdb.set_trace()
        save_audio(
            estimates.cpu(),
            f"{save_dir}/{name}",
        )
        sdr = compute_sdr(estimates.unsqueeze(0), vocals)
        pbar.write(f"{name}, Median SDR: {sdr:.2f}")
        sdr_results[name] = sdr
        pbar.update()
        # i += 1

    total_median_sdr = np.nanmedian(list(sdr_results.values()))
    sdr_results["Total Median SDR"] = total_median_sdr

    output_file = f"{save_dir}/sdr_results.json"
    with open(output_file, "w") as f:
        json.dump(sdr_results, f, indent=4)

    print(f"Total Median SDR={total_median_sdr:.2f}")


if __name__ == "__main__":
    main()
