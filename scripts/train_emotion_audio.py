import torch
import os
import json
from torch.utils.data import Subset

from config import METADATA_CSV, EMOTIONS, DEVICE, MODEL_DIR
from utils.dataset import MusicEmotionDataset
from utils.training import train_classifier
from models.emotion_audio_model import AudioEmotionCNN


# Fast-training knobs
MAX_SAMPLES = 600      # only use first N samples to speed things up
NUM_EPOCHS = 5         # much lower than 15
BATCH_SIZE = 16        # bump up/down depending on your CPU


def main():
    print("=" * 70)
    print("FAST TRAINING: Audio-Based Emotion Classifier")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load dataset (and optionally subsample)
    # ------------------------------------------------------------------
    print(f"\nLoading dataset from {METADATA_CSV}...")
    full_dataset = MusicEmotionDataset(METADATA_CSV, mode="audio")
    full_len = len(full_dataset)
    print(f"Total samples in metadata: {full_len}")

    if full_len > MAX_SAMPLES:
        indices = list(range(MAX_SAMPLES))
        dataset = Subset(full_dataset, indices)
        print(f"Using SUBSET of dataset: {len(dataset)} samples (first {MAX_SAMPLES})")
    else:
        dataset = full_dataset
        print(f"Using full dataset: {len(dataset)} samples")

    # ------------------------------------------------------------------
    # 2. Initialize a lightweight model
    # ------------------------------------------------------------------
    # If you updated AudioEmotionCNN to accept base_channels, you can pass a smaller value:
    # model = AudioEmotionCNN(num_emotions=len(EMOTIONS), base_channels=8)
    model = AudioEmotionCNN(num_emotions=len(EMOTIONS))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Training on DEVICE = {DEVICE}")

    save_path = os.path.join(MODEL_DIR, "audio_emotion_cnn.pt")

    # ------------------------------------------------------------------
    # 3. Fast training configuration
    # ------------------------------------------------------------------
    # Turn off most of the heavy stuff to speed up:
    history = train_classifier(
        model=model,
        dataset=dataset,
        device=DEVICE,
        save_path=save_path,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=1e-4,
        mode="audio",

        # Speed hacks: disable the expensive tricks
        use_class_weights=False,
        use_balanced_sampling=False,
        val_split=0.1,             # small validation split
        early_stopping_patience=2,  # stop early if it stalls
        lr_scheduler=False,        # no scheduler for now
        gradient_clip=None,        # skip grad clipping
    )

    # ------------------------------------------------------------------
    # 4. Save training history
    # ------------------------------------------------------------------
    history_path = os.path.join(MODEL_DIR, "audio_emotion_history_fast.json")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Fast training history saved to {history_path}")


if __name__ == "__main__":
    main()
