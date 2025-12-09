import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.audio_utils import load_mel_spectrogram
from utils.lyrics_utils import load_lyrics
from models.emotion_lyrics_model import get_lyrics_tokenizer
from config import EMOTIONS


class MusicEmotionDataset(Dataset):
    """
    Flexible dataset for:
      - Classification: predicts discrete emotion_label
      - Regression: predicts continuous valence / arousal

    Modes (examples):
      - "audio"          -> audio + classification label
      - "lyrics"         -> lyrics + classification label
      - "audio+lyrics"   -> audio + lyrics + classification label
      - "audio_va"       -> audio + valence/arousal target (+ label if present)
      - "audio+lyrics_va"-> audio + lyrics + valence/arousal target (+ label if present)
    """
    def __init__(
        self,
        metadata_csv: str,
        mode: str = "audio+lyrics",
        bert_model_name: str = "distilbert-base-uncased",
    ):
        self.df = pd.read_csv(metadata_csv)
        self.mode = mode

        # Tokenizer is only needed if we might use lyrics
        self.tokenizer = get_lyrics_tokenizer(bert_model_name)

        # Check which annotations are available
        self.has_emotion_labels = "emotion_label" in self.df.columns
        self.has_va = "valence" in self.df.columns and "arousal" in self.df.columns

        # Classification mapping (only if labels are available)
        if self.has_emotion_labels:
            self.emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
        else:
            self.emotion_to_idx = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        lyrics_path = row.get("lyrics_path", "")

        sample = {}

        # -----------------------------
        # Audio features
        # -----------------------------
        if "audio" in self.mode:
            mel = load_mel_spectrogram(audio_path)  # (n_mels, spec_len)
            sample["audio"] = torch.tensor(mel).unsqueeze(0)  # (1, n_mels, spec_len)

        # -----------------------------
        # Lyrics features
        # -----------------------------
        if "lyrics" in self.mode:
            text = load_lyrics(lyrics_path)
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )
            sample["input_ids"] = encoded["input_ids"].squeeze(0)
            sample["attention_mask"] = encoded["attention_mask"].squeeze(0)

        # -----------------------------
        # Classification label
        # -----------------------------
        if self.has_emotion_labels:
            emotion = row["emotion_label"]
            label = self.emotion_to_idx[emotion]
            sample["label"] = label

        # -----------------------------
        # Valence/Arousal regression target
        # -----------------------------
        # If mode contains "va" and metadata has valence/arousal columns,
        # provide a continuous target vector [valence, arousal].
        if "va" in self.mode and self.has_va:
            v = float(row["valence"])
            a = float(row["arousal"])
            sample["target_va"] = torch.tensor([v, a], dtype=torch.float32)

        return sample
