import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from pathlib import Path


def build_midi_dataset(train=True):
    # Creating a multitrack tokenizer configuration, read the doc to explore other parameters
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=True,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        one_token_stream_for_programs=False,
    )
    tokenizer = REMI(config)

    # Train the tokenizer with Byte Pair Encoding (BPE)
    midi_paths = list(
        Path("/home/ndrw1221/nas/datasets/pop1k7/Pop1K7/midi_analyzed").glob("**/*.mid")
    )
    if train:
        tokenizer.train(vocab_size=30000, files_paths=midi_paths)
        tokenizer.save(Path(".", "tokenizer.json"))

    # Create a Dataset, a DataLoader, and a collator to train a model
    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(
        pad_token_id=tokenizer.pad_token_id,
    )

    return dataset, collator, tokenizer


def train(num_epochs, batch_size, lr):
    dataset, collator, tokenizer = build_midi_dataset(train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Define the model with a custom vocab size to match the tokenizer's vocab size
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    model = GPT2LMHeadModel(config).to(device)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    model.train()

    losses = []
    for epoch in range(num_epochs):
        single_epoch = []
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}") as progress_bar:
            for batch in progress_bar:
                # Send batch data to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
                )
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record loss and update progress bar description
                step_loss = loss.to("cpu").mean().item()
                single_epoch.append(step_loss)
                progress_bar.set_postfix(step_loss=step_loss)  # Display step loss

        # Calculate and store the average loss for this epoch
        losses.append(np.array(single_epoch).mean())
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {losses[-1]:.4f}")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": losses[-1],
            },
            Path("checkpoints", f"epoch_{epoch+1:03}.pkl"),
        )
        np.save(Path("checkpoints", "training_loss"), np.array(losses))


def main():
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    LR = 1e-4
    train(NUM_EPOCHS, BATCH_SIZE, LR)


if __name__ == "__main__":
    main()
