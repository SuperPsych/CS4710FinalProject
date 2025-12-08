import torch
from config import METADATA_CSV, EMOTIONS, DEVICE, MODEL_DIR
from utils.dataset import MusicEmotionDataset
from utils.training import train_classifier
from models.emotion_audio_model import AudioEmotionCNN
import os

def main():
    dataset = MusicEmotionDataset(METADATA_CSV, mode="audio")
    model = AudioEmotionCNN(num_emotions=len(EMOTIONS))
    save_path = os.path.join(MODEL_DIR, "audio_emotion_cnn.pt")
    train_classifier(
        model=model,
        dataset=dataset,
        device=DEVICE,
        save_path=save_path,
        batch_size=8,
        num_epochs=3,
        lr=1e-4,
        mode="audio",
    )

if __name__ == "__main__":
    main()