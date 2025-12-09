import torch
import os
import json
from config import METADATA_CSV, EMOTIONS, DEVICE, MODEL_DIR
from utils.dataset import MusicEmotionDataset
from utils.training import train_classifier
from models.emotion_audio_model import AudioEmotionCNN


def main():
    print("=" * 70)
    print("Training Audio-Based Emotion Classifier")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset from {METADATA_CSV}...")
    dataset = MusicEmotionDataset(METADATA_CSV, mode="audio")
    print(f"Total samples: {len(dataset)}")

    # Initialize model
    model = AudioEmotionCNN(num_emotions=len(EMOTIONS))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")

    save_path = os.path.join(MODEL_DIR, "audio_emotion_cnn.pt")

    # Train with enhanced settings
    history = train_classifier(
        model=model,
        dataset=dataset,
        device=DEVICE,
        save_path=save_path,
        batch_size=16,  # Adjust based on your GPU memory
        num_epochs=15,  # Increased from 3
        lr=1e-4,
        mode="audio",
        use_class_weights=True,
        use_balanced_sampling=True,
        val_split=0.15,
        early_stopping_patience=5,
        lr_scheduler=True,
        gradient_clip=1.0,
    )

    # Save training history
    history_path = os.path.join(MODEL_DIR, "audio_emotion_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training history saved to {history_path}")


if __name__ == "__main__":
    main()