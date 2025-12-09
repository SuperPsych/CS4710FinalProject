# scripts/train_emotion_audio_va.py

import os
import json
import random
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import METADATA_CSV, DEVICE, MODEL_DIR
from utils.dataset import MusicEmotionDataset
from models.emotion_audio_model import AudioEmotionCNN


# ------------------------
# Training hyperparameters
# ------------------------
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.15
PATIENCE = 5  # early stopping


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def make_dataloaders(dataset, batch_size: int, val_split: float = 0.15):
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    return train_loader, val_loader, len(train_ds), len(val_ds)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    criterion: nn.Module,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        audio = batch["audio"].to(device)            # [B, 1, n_mels, spec_len]
        target_va = batch["target_va"].to(device)    # [B, 2]

        optimizer.zero_grad()
        preds = model(audio)                         # [B, 2]
        loss = criterion(preds, target_va)
        loss.backward()
        optimizer.step()

        bsz = target_va.size(0)
        running_loss += loss.item() * bsz
        total_samples += bsz

    return running_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            target_va = batch["target_va"].to(device)

            preds = model(audio)

            loss = criterion(preds, target_va)

            bsz = target_va.size(0)
            running_loss += loss.item() * bsz
            total_samples += bsz

            all_preds.append(preds.cpu())
            all_targets.append(target_va.cpu())

    avg_loss = running_loss / max(total_samples, 1)

    if all_preds:
        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        # RMSE per dimension
        mse = (preds_cat - targets_cat).pow(2).mean(dim=0)
        rmse = mse.sqrt()
        valence_rmse = rmse[0].item()
        arousal_rmse = rmse[1].item()
    else:
        valence_rmse = float("nan")
        arousal_rmse = float("nan")

    return {
        "loss": avg_loss,
        "valence_rmse": valence_rmse,
        "arousal_rmse": arousal_rmse,
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    history: Any,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    history = ckpt.get("history", {})
    return epoch, best_val_loss, history


def main():
    print("=" * 70)
    print("TRAINING: Audio Valence/Arousal Regressor")
    print("=" * 70)

    set_seed(42)

    # -------------------------
    # 1. Dataset & Dataloaders
    # -------------------------
    print(f"\nLoading dataset (VA mode) from {METADATA_CSV}...")
    dataset = MusicEmotionDataset(METADATA_CSV, mode="audio_va")
    print(f"Total samples: {len(dataset)}")

    train_loader, val_loader, n_train, n_val = make_dataloaders(
        dataset, batch_size=BATCH_SIZE, val_split=VAL_SPLIT
    )
    print(f"Train samples: {n_train} | Val samples: {n_val}")

    # -------------------------
    # 2. Model, optimizer, loss
    # -------------------------
    model = AudioEmotionCNN(output_dim=2)  # [valence, arousal]
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, "audio_emotion_cnn_va_checkpoint.pt")
    final_model_path = os.path.join(MODEL_DIR, "audio_emotion_cnn.pt")
    history_path = os.path.join(MODEL_DIR, "audio_emotion_va_history.json")

    # -------------------------
    # 3. Resume if checkpoint exists
    # -------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "valence_rmse": [],
        "arousal_rmse": [],
    }

    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Found checkpoint at {checkpoint_path} — resuming training.")
        start_epoch, best_val_loss, history = load_checkpoint(
            checkpoint_path, model, optimizer, DEVICE
        )
        print(f"Resuming from epoch {start_epoch + 1}, best_val_loss={best_val_loss:.6f}")
    else:
        print("\n🆕 No checkpoint found — starting from scratch.")

    # -------------------------
    # 4. Training loop with early stopping
    # -------------------------
    epochs_no_improve = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        print("\n" + "-" * 70)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, criterion)
        metrics = evaluate(model, val_loader, DEVICE, criterion)

        val_loss = metrics["loss"]
        valence_rmse = metrics["valence_rmse"]
        arousal_rmse = metrics["arousal_rmse"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["valence_rmse"].append(valence_rmse)
        history["arousal_rmse"].append(arousal_rmse)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}")
        print(f"  Valence RMSE: {valence_rmse:.4f}")
        print(f"  Arousal RMSE: {arousal_rmse:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            print(f"  ✅ New best val loss: {val_loss:.6f} (prev: {best_val_loss:.6f})")
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save best model and checkpoint
            torch.save(model.state_dict(), final_model_path)
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, best_val_loss, history
            )
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print(f"\n⏹ Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    # -------------------------
    # 5. Save training history
    # -------------------------
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete.")
    print(f"   Best val loss: {best_val_loss:.6f}")
    print(f"   Best model saved to: {final_model_path}")
    print(f"   History saved to: {history_path}")


if __name__ == "__main__":
    main()
