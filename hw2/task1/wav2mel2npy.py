import os
import numpy as np
import torch
import torchaudio

# Directory path
data_dir = "/home/ndrw1221/nas/musdb18hq-dataset"

# Parameters for STFT
sr = 44100  # Sampling rate
n_fft = 4096  # Number of FFT components
hop_length = 1024  # Number of samples between successive frames


def process_wav_to_stft(file_path):
    # Load the audio file using torchaudio
    waveform, orig_sr = torchaudio.load(file_path)
    # Resample if needed
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sr)
        waveform = resampler(waveform)

    # Define the window function
    window = torch.hann_window(n_fft)

    # Ensure waveform has shape (channels, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Reshape waveform to (nb_samples, nb_channels, nb_timesteps)
    x = waveform.unsqueeze(0)  # Shape: (1, channels, time)

    # Flatten samples and channels for STFT computation
    x = x.view(-1, x.shape[-1])  # Shape: (nb_samples * nb_channels, time)

    # Compute STFT
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        pad_mode="reflect",
        return_complex=True,
    )

    # Reshape back to (nb_samples, nb_channels, freq_bins, time_frames)
    nb_samples = 1
    nb_channels = waveform.shape[0]
    stft = stft.view(nb_samples, nb_channels, stft.shape[-2], stft.shape[-1])

    # Convert complex tensor to real and imaginary parts
    stft_real_imag = torch.view_as_real(
        stft
    )  # Shape: (nb_samples, nb_channels, freq_bins, time_frames, 2)

    return stft_real_imag


def save_stft(stft, output_path):
    # Convert STFT tensor to numpy array
    stft_np = stft.numpy()
    # Save the STFT as an .npy file
    os.makedirs(
        os.path.dirname(output_path), exist_ok=True
    )  # Create the output directory if it doesn't exist
    np.save(output_path, stft_np)


def convert_dataset_to_stft(dataset_dir, output_dir):
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")

                # Convert WAV to STFT
                stft = process_wav_to_stft(file_path)

                # Construct the output path
                relative_path = os.path.relpath(
                    file_path, dataset_dir
                )  # Get the relative path of the file
                output_path = os.path.join(output_dir, relative_path).replace(
                    ".wav", ".npy"
                )

                # Save STFT as .npy file
                save_stft(stft, output_path)
                print(f"Saved STFT to {output_path}")


if __name__ == "__main__":
    # Convert test dataset
    test_input_dir = os.path.join(data_dir, "test")
    test_output_dir = os.path.join(data_dir, "test-stft")
    convert_dataset_to_stft(test_input_dir, test_output_dir)

    # Convert train dataset
    train_input_dir = os.path.join(data_dir, "train")
    train_output_dir = os.path.join(data_dir, "train-stft")
    convert_dataset_to_stft(train_input_dir, train_output_dir)
