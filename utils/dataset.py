import torch
from torch.utils.data import Dataset
import pandas as pd
from utils.audio_utils import load_mel_spectrogram
from utils.lyrics_utils import load_lyrics
from models.emotion_lyrics_model import get_lyrics_tokenizer
from config import EMOTIONS

class MusicEmotionDataset(Dataset):
    def __init__(self, metadata_csv: str, mode: str = "audio+lyrics", bert_model_name="distilbert-base-uncased"):
        self.df = pd.read_csv(metadata_csv)
        self.mode = mode
        self.tokenizer = get_lyrics_tokenizer(bert_model_name)
        self.emotion_to_idx = {e: i for i in EMOTIONS}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        lyrics_path = row["lyrics_path"]
        emotion = row["emotion_label"]

        label = self.emotion_to_idx[emotion]

        sample = {"label": label}

        if "audio" in self.mode:
            mel = load_mel_spectrogram(audio_path)  # (n_mels, spec_len)
            sample["audio"] = torch.tensor(mel).unsqueeze(0)  # (1, n_mels, spec_len)

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

        return sample