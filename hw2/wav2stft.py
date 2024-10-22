import os
from pathlib import Path
import torchaudio
import torch
import numpy as np
import tqdm
from openunmix import transforms


def compute_stft_and_save(input_wav_path, output_npy_path, n_fft, n_hop, sample_rate):
    # Load audio file
    waveform, sr = torchaudio.load(input_wav_path)
    if sr != sample_rate:
        # Resample if necessary
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Move waveform to CPU (if not already)
    waveform = waveform.to("cpu")

    # Add a batch dimension if necessary (for single example case)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # Shape: (1, nb_channels, nb_timesteps)

    # Create the STFT transform
    stft, _ = transforms.make_filterbanks(
        n_fft=n_fft, n_hop=n_hop, sample_rate=sample_rate
    )

    # Apply STFT
    mix_stft = stft(waveform)

    # Compute magnitude spectrogram
    magnitude = torch.sqrt(mix_stft.pow(2).sum(-1))

    # Convert to numpy array and save
    magnitude = magnitude.numpy()

    # Remove the batch dimension
    magnitude = magnitude.squeeze(0)

    np.save(output_npy_path, magnitude)


def process_dataset(dataset_root, output_root, n_fft, n_hop, sample_rate):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    for split in ["train", "test"]:
        print(f"Processing {split} set...")
        split_folder = dataset_root / split
        output_split_folder = output_root / split
        output_split_folder.mkdir(parents=True, exist_ok=True)

        # Iterate over tracks
        for track_folder in tqdm.tqdm(list(split_folder.iterdir())):
            if track_folder.is_dir():
                output_track_folder = output_split_folder / track_folder.name
                output_track_folder.mkdir(parents=True, exist_ok=True)

                # Process each source (e.g., mixture, vocals, drums, bass)
                for source_file in track_folder.glob("*.wav"):
                    source_name = source_file.stem
                    output_npy_file = output_track_folder / f"{source_name}.npy"
                    compute_stft_and_save(
                        input_wav_path=source_file,
                        output_npy_path=output_npy_file,
                        n_fft=n_fft,
                        n_hop=n_hop,
                        sample_rate=sample_rate,
                    )


if __name__ == "__main__":
    dataset_root = "/home/ndrw1221/nas/datasets/musdb18hq-dataset"  # Replace with your dataset path
    output_root = "/home/ndrw1221/nas/datasets/musdb18stft"  # Where you want to save the NPY files
    n_fft = 4096  # Same as args.nfft in your training code
    n_hop = 1024  # Same as args.nhop in your training code
    sample_rate = 44100  # Or whatever sample rate your model uses

    process_dataset(dataset_root, output_root, n_fft, n_hop, sample_rate)
