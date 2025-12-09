import torch
import os
import json
from torch.utils.data import Subset
import pandas as pd
import random
from config import METADATA_CSV, EMOTIONS, DEVICE, MODEL_DIR
from utils.dataset import MusicEmotionDataset
from utils.training import train_classifier
from models.emotion_audio_model import AudioEmotionCNN


# Fast-training knobs
MAX_SAMPLES = 2000      # only use first N samples to speed things up
NUM_EPOCHS = 5         # much lower than 15
BATCH_SIZE = 16        # bump up/down depending on your CPU
N_PER_CLASS = 500


def main():
    print("=" * 70)
    print("FAST TRAINING: Audio-Based Emotion Classifier")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Build a BALANCED subset from metadata
    # ------------------------------------------------------------------
    print(f"\nLoading metadata from {METADATA_CSV}...")
    meta = pd.read_csv(METADATA_CSV)
    print("Label counts in full metadata:")
    print(meta["emotion_label"].value_counts())

    indices = []
    for emotion in EMOTIONS:
        class_indices = meta.index[meta["emotion_label"] == emotion].tolist()
        if not class_indices:
            print(f"⚠️ No samples for class '{emotion}' in metadata – skipping.")
            continue

        # Shuffle and cap at N_PER_CLASS
        random.shuffle(class_indices)
        keep = class_indices[:N_PER_CLASS]
        indices.extend(keep)
        print(f"Using {len(keep)} samples for class '{emotion}'")

    # Optional overall cap
    if len(indices) > MAX_SAMPLES:
        random.shuffle(indices)
        indices = indices[:MAX_SAMPLES]

    print(f"\nTotal selected indices: {len(indices)}")

    full_dataset = MusicEmotionDataset(METADATA_CSV, mode="audio")
    dataset = Subset(full_dataset, indices)
    print(f"Final dataset size: {len(dataset)}")

    # ------------------------------------------------------------------
    # 2. Initialize a lightweight model
    # ------------------------------------------------------------------
    # If you updated AudioEmotionCNN to accept base_channels, you can pass a smaller value:
    # model = AudioEmotionCNN(num_emotions=len(EMOTIONS), base_channels=8)
    model = AudioEmotionCNN(num_emotions=len(EMOTIONS))

    # Path to checkpoint
    save_path = os.path.join(MODEL_DIR, "audio_emotion_cnn.pt")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 🔄 2.5 Resume Training If Checkpoint Exists
    # ------------------------------------------------------------------
    if os.path.exists(save_path):
        print(f"\n🔄 Found existing checkpoint at {save_path} — resuming training from it.")
        state_dict = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(f"\n🆕 No checkpoint found at {save_path} — starting from scratch.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Training on DEVICE = {DEVICE}")

    # ------------------------------------------------------------------
    # 3. Fast training configuration
    # ------------------------------------------------------------------
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
        val_split=0.1,
        early_stopping_patience=2,
        lr_scheduler=False,
        gradient_clip=None,
    )

    # ------------------------------------------------------------------
    # 4. Save training history
    # ------------------------------------------------------------------
    history_path = os.path.join(MODEL_DIR, "audio_emotion_history_fast.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Fast training history saved to {history_path}")


if __name__ == "__main__":
    main()
