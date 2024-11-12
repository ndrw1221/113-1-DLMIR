import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI
from pathlib import Path
from tqdm import tqdm
import random
from tqdm import tqdm


def load_tokenizer():
    """Load the tokenizer."""
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
    # tokenizer.load(Path(".", "tokenizer.json"))
    return tokenizer


def load_model(checkpoint_path, tokenizer):
    """Load the GPT-2 model from a checkpoint."""
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    model = GPT2LMHeadModel(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def generate_midi(model, tokenizer, priming_seq, max_bars=32):
    """Generate MIDI continuation up to a specified number of bars."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    input_ids = priming_seq.copy()
    generated_ids = []
    num_bars = 0

    # Generate tokens until reaching the desired number of bars
    with torch.no_grad():
        progress_bar = tqdm(total=max_bars, desc="Generating MIDI", unit="bars")

        while num_bars < max_bars:
            # Generate the next token
            output = model.generate(
                input_ids=torch.tensor([input_ids], dtype=torch.long).to(device),
                max_new_tokens=1,
                do_sample=True,
                top_k=5,
                top_p=0.98,
                temperature=1.2,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Get the newly generated token
            new_token_id = output[0, -1].item()
            generated_ids.append(new_token_id)
            input_ids.append(new_token_id)

            # Truncate the input sequence if it exceeds the model's maximum context length
            if len(input_ids) > 1024:
                input_ids = input_ids[-1024:]

            # Check if the new token is a bar token
            if tokenizer[new_token_id] == "Bar_None":
                num_bars += 1
                progress_bar.update(1)

        progress_bar.close()

    midi = tokenizer([priming_seq + generated_ids])
    return midi


def save_midi(midi, output_path):
    """Save the generated MIDI to a file."""
    midi.dump_midi(output_path)


def main():
    # Load tokenizer and model
    tokenizer = load_tokenizer()
    checkpoint_path = Path("checkpoints", "epoch_100.pkl")
    model = load_model(checkpoint_path, tokenizer)

    # Get a random priming sequence
    prompt_songs = [
        "prompt_song/song_1.mid",
        "prompt_song/song_2.mid",
        "prompt_song/song_3.mid",
    ]
    for prompt_song in prompt_songs:
        print(f"Generating from {prompt_song}...")
        priming_seq = tokenizer(prompt_song)[0].ids

        # Generate MIDI with be exactly 32 bars (8 bars of the prompt + 24 bars of continuation)
        generated_midi = generate_midi(model, tokenizer, priming_seq, max_bars=24)

        # Save the generated MIDI to a file
        output_path = Path(
            "generated_midi",
            "from_prompt",
            f"{prompt_song.split('/')[-1][:-4]}",
            "epoch100_topk5_topp0.98_temperature1.2.mid",
        )
        output_path.parent.mkdir(exist_ok=True, parents=True)
        save_midi(generated_midi, output_path)
        print(f"MIDI saved to {output_path}")


if __name__ == "__main__":
    main()
