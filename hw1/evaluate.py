import argparse
import torch
import tqdm
from torch.utils.data import DataLoader
from dataset import SlakhDataset
from model import Model
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on the test set."
    )
    parser.add_argument(
        "--root_dir", type=str, default="dataset", help="Root directory of the dataset."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker threads for data loading.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation (cuda or cpu).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.pth",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=9, help="Number of output classes."
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=24000,
        help="Sampling rate of the audio data.",
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

    # Initialize test dataset and dataloader
    test_dataset = SlakhDataset(root_dir=args.root_dir, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Initialize model
    model = Model(num_classes=args.num_classes, sampling_rate=args.sampling_rate)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, labels in tqdm.tqdm(test_loader):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Binarize probabilities to get predictions
            preds = (probs >= args.threshold).int()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    # Concatenate all labels and predictions
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Generate classification report
    class_names = [f"Class_{i}" for i in range(args.num_classes)]
    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )

    print("Classification Report:")
    print(report)

    # Optionally, compute and print macro/micro averages
    # precision_macro = precision_score(
    #     all_labels, all_preds, average="macro", zero_division=0
    # )
    # recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    # f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # print(f"Macro Precision: {precision_macro:.4f}")
    # print(f"Macro Recall: {recall_macro:.4f}")
    # print(f"Macro F1 Score: {f1_macro:.4f}")


if __name__ == "__main__":
    main()
