import torch
from config import METADATA_CSV, EMOTIONS, DEVICE, MODEL_DIR
from utils.dataset import MusicEmotionDataset
from utils.training import train_classifier
from models.emotion_lyrics_model import LyricsEmotionBERT
import os

BERT_MODEL_NAME = "distilbert-base-uncased"

def main():
    dataset = MusicEmotionDataset(METADATA_CSV, mode="lyrics", bert_model_name=BERT_MODEL_NAME)
    model = LyricsEmotionBERT(BERT_MODEL_NAME, num_emotions=len(EMOTIONS))
    save_path = os.path.join(MODEL_DIR, "lyrics_emotion_bert.pt")
    train_classifier(
        model=model,
        dataset=dataset,
        device=DEVICE,
        save_path=save_path,
        batch_size=4,
        num_epochs=3,
        lr=2e-5,
        mode="lyrics",
    )

if __name__ == "__main__":
    main()