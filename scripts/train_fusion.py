import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import METADATA_CSV, EMOTIONS, DEVICE, MODEL_DIR
from utils.dataset import MusicEmotionDataset
from models.emotion_audio_model import AudioEmotionCNN
from models.emotion_lyrics_model import LyricsEmotionBERT
from models.fusion_model import FusionEmotionModel


AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_emotion_cnn.pt")
LYRICS_MODEL_PATH = os.path.join(MODEL_DIR, "lyrics_emotion_bert.pt")
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "fusion_emotion_model.pt")
BERT_MODEL_NAME = "distilbert-base-uncased"


class FusionLogitsDataset(Dataset):
    """
    Precomputes audio and lyrics logits for each sample using frozen models,
    then exposes them as a dataset for training the fusion network.
    """

    def __init__(self, metadata_csv: str, audio_model, lyrics_model, device: str):
        super().__init__()
        base_dataset = MusicEmotionDataset(
            metadata_csv,
            mode="audio+lyrics",
            bert_model_name=BERT_MODEL_NAME,
        )

        loader = DataLoader(base_dataset, batch_size=16, shuffle=False)

        audio_model.eval()
        lyrics_model.eval()

        audio_model.to(device)
        lyrics_model.to(device)

        audio_logits_list = []
        lyrics_logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Precomputing logits for fusion"):
                labels = batch["label"].to(device)

                # Audio logits
                audio = batch["audio"].to(device)
                a_logits = audio_model(audio)

                # Lyrics logits
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                l_logits = lyrics_model(input_ids, attention_mask)

                audio_logits_list.append(a_logits.cpu())
                lyrics_logits_list.append(l_logits.cpu())
                labels_list.append(labels.cpu())

        self.audio_logits = torch.cat(audio_logits_list, dim=0)
        self.lyrics_logits = torch.cat(lyrics_logits_list, dim=0)
        self.labels = torch.cat(labels_list, dim=0)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "audio_logits": self.audio_logits[idx],
            "lyrics_logits": self.lyrics_logits[idx],
            "label": self.labels[idx],
        }


def train_fusion():
    if not os.path.exists(AUDIO_MODEL_PATH):
        raise FileNotFoundError(
            f"Audio model not found at {AUDIO_MODEL_PATH}. "
            "Train it first with train_emotion_audio.py."
        )
    if not os.path.exists(LYRICS_MODEL_PATH):
        raise FileNotFoundError(
            f"Lyrics model not found at {LYRICS_MODEL_PATH}. "
            "Train it first with train_emotion_lyrics.py."
        )

    # Load frozen base models
    audio_model = AudioEmotionCNN(num_emotions=len(EMOTIONS))
    audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=DEVICE))

    lyrics_model = LyricsEmotionBERT(BERT_MODEL_NAME, num_emotions=len(EMOTIONS))
    lyrics_model.load_state_dict(torch.load(LYRICS_MODEL_PATH, map_location=DEVICE))

    # Build fusion dataset
    fusion_dataset = FusionLogitsDataset(
        METADATA_CSV,
        audio_model=audio_model,
        lyrics_model=lyrics_model,
        device=DEVICE,
    )

    # Fusion model
    fusion_model = FusionEmotionModel(num_emotions=len(EMOTIONS)).to(DEVICE)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    loader = DataLoader(fusion_dataset, batch_size=32, shuffle=True)
    num_epochs = 5

    for epoch in range(num_epochs):
        fusion_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(loader, desc=f"Fusion training epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            audio_logits = batch["audio_logits"].to(DEVICE)
            lyrics_logits = batch["lyrics_logits"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = fusion_model(audio_logits, lyrics_logits)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"[Fusion] Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(fusion_model.state_dict(), FUSION_MODEL_PATH)
    print(f"Saved fusion model to {FUSION_MODEL_PATH}")


if __name__ == "__main__":
    train_fusion()