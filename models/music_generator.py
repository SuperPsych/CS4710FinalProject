import torch
import torch.nn as nn
import numpy as np
import pretty_midi

class MelodyGenerator(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        logits = self.fc(x)
        return logits, hidden

    def generate(self, start_seq, length=32, temperature=1.0, device="cpu"):
        self.eval()
        generated = list(start_seq)
        input_ids = torch.tensor(start_seq, dtype=torch.long, device=device).unsqueeze(0)
        hidden = None
        with torch.no_grad():
            for _ in range(length):
                logits, hidden = self.forward(input_ids, hidden)
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token.item())
                input_ids = next_token
        return generated

def notes_to_midi(note_sequence, output_path="generated_melody.mid", tempo=120):
    """
    note_sequence: list of integers representing MIDI pitches (e.g., 60 = middle C).
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    time = 0.0
    duration = 0.5
    for note in note_sequence:
        pitch = int(note)
        note_obj = pretty_midi.Note(
            velocity=80, pitch=pitch, start=time, end=time + duration
        )
        instrument.notes.append(note_obj)
        time += duration
    pm.instruments.append(instrument)
    pm.write(output_path)
    return output_path