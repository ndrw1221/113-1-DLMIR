import argparse
import torch
from torch.utils.data import DataLoader
from dataset import SlakhDataset
from model import Model
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a multi-label classification model."
    )
    parser.add_argument(
        "--root_dir", type=str, default="dataset", help="Root directory of the dataset."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer."
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
        help="Device to use for training (cuda or cpu).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="model.pth",
        help="Path to save the trained model.",
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
        "--run_name", type=str, default=None, help="Unique name for the training run."
    )
    parser.add_argument(
        "--full_finetune",
        type=bool,
        default=False,
        help="Whether to full finetune the model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize datasets and dataloaders
    train_dataset = SlakhDataset(root_dir=args.root_dir, split="train")
    val_dataset = SlakhDataset(root_dir=args.root_dir, split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Initialize model
    model = Model(
        num_classes=args.num_classes,
        sampling_rate=args.sampling_rate,
        full_finetune=args.full_finetune,
    )
    model.to(device)

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # Create a unique run name if not provided
    if args.run_name is None:
        import pytz
        from datetime import datetime

        tz = pytz.timezone("Asia/Taipei")
        run_name = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_name = args.run_name

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        run_name=run_name,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
