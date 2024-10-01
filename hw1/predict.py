# predict.py

import argparse
import torch
import numpy as np
import librosa
from model import Model


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on an audio file.")
    parser.add_argument(
        "--audio_path", type=str, required=True, help="Path to the input audio file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.pth",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="pred_pianoroll.npy",
        help="Path to save the predicted pianoroll.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=9, help="Number of output classes."
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=24000, help="Sampling rate to use."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)."
    )
    parser.add_argument(
        "--time_step", type=float, default=5.0, help="Time step in seconds."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing probabilities.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = Model(num_classes=args.num_classes, sampling_rate=args.sampling_rate)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    # Load the audio file
    audio_path = args.audio_path
    sampling_rate = args.sampling_rate
    audio, sr = librosa.load(audio_path, sr=sampling_rate)

    # Determine total duration
    total_time = len(audio) / sampling_rate  # in seconds
    num_time_steps = int(np.ceil(total_time / args.time_step))

    # Prepare segments
    segments = []
    expected_length = int(args.time_step * sampling_rate)

    for i in range(num_time_steps):
        start_time = i * args.time_step
        end_time = (i + 1) * args.time_step
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)

        segment = audio[start_sample:end_sample]

        # If segment is shorter than expected, pad with zeros
        if len(segment) < expected_length:
            padding = np.zeros(expected_length - len(segment))
            segment = np.concatenate([segment, padding])
        segments.append(segment)

    # Process the segments with the model
    # The model expects a list of numpy arrays as input
    inputs = segments  # List of numpy arrays

    # Prepare processed inputs
    processed = model.processor(
        inputs, sampling_rate=model.sampling_rate, return_tensors="pt", padding=True
    )
    device = next(model.parameters()).device
    processed = processed.to(device)

    # Run the model
    with torch.no_grad():
        outputs = model.feature_extractor(**processed)
        x = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
        x = x.permute(0, 2, 1)  # (batch_size, hidden_size, sequence_length)
        x = model.pooling(x).squeeze(-1)  # (batch_size, hidden_size)
        x = model.classifier(x)  # (batch_size, num_classes)
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(x)
        # Binarize outputs
        preds = (probs >= args.threshold).cpu().numpy()  # (batch_size, num_classes)

    # preds is of shape (batch_size, num_classes)
    # Transpose to (num_classes, num_time_steps)
    pred_pianoroll = preds.T  # (num_classes, num_time_steps)

    # Save pred_pianoroll
    np.save(args.output_path, pred_pianoroll)
    print(f"Saved predicted pianoroll to {args.output_path}")


if __name__ == "__main__":
    main()
