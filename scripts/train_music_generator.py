import os
import glob
import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import DATA_DIR, DEVICE, MODEL_DIR
from models.music_generator import MelodyGenerator


MIDI_DIR = os.path.join(DATA_DIR, "midi")
MELODY_MODEL_PATH = os.path.join(MODEL_DIR, "melody_generator.pt")

PITCH_MIN = 60  # MIDI note: middle C
PITCH_MAX = 72  # inclusive
VOCAB_SIZE = PITCH_MAX - PITCH_MIN + 1  # 13
SEQ_LEN = 32


def midi_to_note_sequence(midi_path: str) -> list[int]:
    """Extract a sequence of MIDI pitches (ints) from a MIDI file."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((n.start, n.pitch))
    if not notes:
        return []
    # Sort by start time
    notes.sort(key=lambda x: x[0])
    pitches = [p for _, p in notes]
    return pitches


def pitches_to_tokens(pitches: list[int]) -> list[int]:
    """
    Map pitches to token indices 0..12 for range [60..72].
    Notes outside this range are clipped; you can also choose to skip them.
    """
    tokens = []
    for p in pitches:
        p_clipped = min(max(p, PITCH_MIN), PITCH_MAX)
        tokens.append(p_clipped - PITCH_MIN)
    return tokens


class MelodyDataset(Dataset):
    """
    From each MIDI file we create many (input_seq, target_seq) pairs
    using a sliding window over the token sequence.
    """

    def __init__(self, midi_dir: str, seq_len: int = SEQ_LEN):
        self.samples = []
        midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) + glob.glob(
            os.path.join(midi_dir, "*.midi")
        )

        if not midi_files:
            print(f"[WARN] No MIDI files found in {midi_dir}.")
        else:
            print(f"Found {len(midi_files)} MIDI files in {midi_dir}.")

        for path in midi_files:
            pitches = midi_to_note_sequence(path)
            if len(pitches) < seq_len + 1:
                continue

            tokens = pitches_to_tokens(pitches)
            # Sliding window
            for i in range(len(tokens) - seq_len):
                inp = tokens[i : i + seq_len]
                tgt = tokens[i + 1 : i + seq_len + 1]
                self.samples.append((inp, tgt))

        print(f"Total melody training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return {
            "input_ids": torch.tensor(inp, dtype=torch.long),
            "labels": torch.tensor(tgt, dtype=torch.long),
        }


def train_melody_generator():
    dataset = MelodyDataset(MIDI_DIR, seq_len=SEQ_LEN)
    if len(dataset) == 0:
        raise RuntimeError(
            "No training samples were created for the melody generator. "
            "Check that you have enough MIDI files and that they are not empty."
        )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MelodyGenerator(vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_tokens = 0

        for batch in tqdm(loader, desc=f"Melody training epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)  # (B, T)
            labels = batch["labels"].to(DEVICE)        # (B, T)

            logits, _ = model(input_ids)               # (B, T, vocab)
            B, T, V = logits.shape

            loss = criterion(
                logits.view(B * T, V),
                labels.view(B * T),
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B * T
            total_tokens += B * T

        epoch_loss = running_loss / total_tokens
        print(f"[Melody] Epoch {epoch+1}: loss={epoch_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MELODY_MODEL_PATH)
    print(f"Saved melody generator model to {MELODY_MODEL_PATH}")


if __name__ == "__main__":
    train_melody_generator()