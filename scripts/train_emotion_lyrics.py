import torch
import os
import json
from config import METADATA_CSV, EMOTIONS, DEVICE, MODEL_DIR
from utils.dataset import MusicEmotionDataset
from utils.training import train_classifier
from models.emotion_lyrics_model import LyricsEmotionBERT

BERT_MODEL_NAME = "distilbert-base-uncased"


def main():
    print("=" * 70)
    print("Training Lyrics-Based Emotion Classifier")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset from {METADATA_CSV}...")
    dataset = MusicEmotionDataset(METADATA_CSV, mode="lyrics", bert_model_name=BERT_MODEL_NAME)
    print(f"Total samples: {len(dataset)}")

    # Initialize model
    model = LyricsEmotionBERT(BERT_MODEL_NAME, num_emotions=len(EMOTIONS))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Freeze BERT layers for first few epochs (optional - uncomment to use)
    # print("\n⚠️  Freezing BERT backbone for initial training...")
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Trainable parameters (with frozen BERT): {trainable_params:,}")

    save_path = os.path.join(MODEL_DIR, "lyrics_emotion_bert.pt")

    # Train with enhanced settings
    # Note: Lower batch size and learning rate for BERT fine-tuning
    history = train_classifier(
        model=model,
        dataset=dataset,
        device=DEVICE,
        save_path=save_path,
        batch_size=8,  # Smaller batch for BERT
        num_epochs=10,  # Increased from 3
        lr=2e-5,  # Lower LR for fine-tuning
        mode="lyrics",
        use_class_weights=True,
        use_balanced_sampling=True,
        val_split=0.15,
        early_stopping_patience=4,
        lr_scheduler=True,
        gradient_clip=1.0,
    )

    # Save training history
    history_path = os.path.join(MODEL_DIR, "lyrics_emotion_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training history saved to {history_path}")


if __name__ == "__main__":
    main()