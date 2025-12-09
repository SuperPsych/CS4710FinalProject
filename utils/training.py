import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def compute_class_weights(dataset, device):
    """
    Compute class weights based on inverse frequency.
    Helps balance training when classes are imbalanced.
    """
    labels = [dataset[i]["label"] for i in range(len(dataset))]
    label_counts = Counter(labels)

    # Total samples
    total = len(labels)
    num_classes = len(label_counts)

    # Compute weights: total / (num_classes * count_for_class)
    weights = torch.zeros(num_classes, device=device)
    for label, count in label_counts.items():
        weights[label] = total / (num_classes * count)

    # Normalize weights so they average to 1.0
    weights = weights / weights.mean()

    print(f"\n{'=' * 70}")
    print(f"Class Distribution in Training Data:")
    print(f"{'=' * 70}")
    for label, count in sorted(label_counts.items()):
        pct = count / total * 100
        print(f"  Class {label}: {count:4d} samples ({pct:5.1f}%) - weight: {weights[label]:.3f}")
    print(f"{'=' * 70}")

    return weights


def create_balanced_sampler(dataset):
    """
    Create a WeightedRandomSampler that balances classes during training.
    This ensures each batch has roughly equal representation of all classes.
    """
    labels = [dataset[i]["label"] for i in range(len(dataset))]
    label_counts = Counter(labels)

    # Weight for each sample (inverse of class frequency)
    sample_weights = [1.0 / label_counts[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"✅ Using balanced sampling strategy")
    return sampler


def split_dataset(dataset, val_split=0.15, random_state=42):
    """
    Split dataset into train and validation sets with stratification.
    """
    indices = list(range(len(dataset)))
    labels = [dataset[i]["label"] for i in indices]

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=random_state,
        stratify=labels
    )

    return train_indices, val_indices


class SubsetDataset(torch.utils.data.Dataset):
    """Wrapper to create a subset of a dataset given indices."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    # Preserve emotion_to_idx for class weight computation
    @property
    def emotion_to_idx(self):
        return self.dataset.emotion_to_idx


def evaluate_model(model, loader, criterion, device, mode, num_classes):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    # Per-class tracking
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)

            if mode == "audio":
                audio = batch["audio"].to(device)
                logits = model(audio)
            elif mode == "lyrics":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
            elif mode == "fusion":
                audio_logits = batch["audio_logits"].to(device)
                lyrics_logits = batch["lyrics_logits"].to(device)
                logits = model(audio_logits, lyrics_logits)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update per-class accuracy
            for label_idx in range(num_classes):
                mask = labels == label_idx
                if mask.sum() > 0:
                    class_correct[label_idx] += (preds[mask] == labels[mask]).sum()
                    class_total[label_idx] += mask.sum()

    avg_loss = running_loss / total
    accuracy = correct / total

    # Calculate per-class accuracies
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = (class_correct[i] / class_total[i]).item()
        else:
            class_accuracies[i] = 0.0

    return avg_loss, accuracy, all_preds, all_labels, class_accuracies


def train_classifier(
        model,
        dataset,
        device,
        save_path,
        batch_size=16,
        num_epochs=15,
        lr=1e-4,
        mode="audio",
        use_class_weights=True,
        use_balanced_sampling=True,
        val_split=0.15,
        early_stopping_patience=5,
        lr_scheduler=True,
        gradient_clip=1.0,
        weight_decay=0.01,
        warmup_epochs=0,
):
    """
    Enhanced training with validation, early stopping, and learning rate scheduling.

    Args:
        model: PyTorch model to train
        dataset: Training dataset
        device: Device to train on ('cuda' or 'cpu')
        save_path: Path to save the trained model
        batch_size: Batch size for training
        num_epochs: Maximum number of training epochs
        lr: Initial learning rate
        mode: Training mode ('audio', 'lyrics', or 'fusion')
        use_class_weights: Whether to use class weights in loss function
        use_balanced_sampling: Whether to use balanced sampling during training
        val_split: Fraction of data to use for validation
        early_stopping_patience: Stop if validation doesn't improve for N epochs
        lr_scheduler: Whether to use learning rate scheduling
        gradient_clip: Max norm for gradient clipping (None to disable)
        weight_decay: L2 regularization strength
        warmup_epochs: Number of epochs for learning rate warmup
    """
    from config import EMOTIONS

    model.to(device)
    num_classes = len(EMOTIONS)

    print(f"\n{'=' * 70}")
    print(f"Training Configuration:")
    print(f"{'=' * 70}")
    print(f"  Mode: {mode}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Gradient clipping: {gradient_clip}")
    print(f"  Weight decay: {weight_decay}")
    print(f"{'=' * 70}")

    # Split into train and validation
    print(f"\nSplitting dataset: {100 * (1 - val_split):.0f}% train, {100 * val_split:.0f}% val")
    train_indices, val_indices = split_dataset(dataset, val_split=val_split)
    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create data loaders
    if use_balanced_sampling:
        train_sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Compute class weights if enabled
    if use_class_weights:
        class_weights = compute_class_weights(train_dataset, device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"\n✅ Using weighted loss with label smoothing (0.1)")
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        print(f"\n⚠️  Training without class weights")

    # Learning rate scheduler
    scheduler = None
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7
        )
        print(f"✅ Using ReduceLROnPlateau scheduler")

    # Training tracking
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"\n{'=' * 70}")
    print(f"Starting Training")
    print(f"{'=' * 70}\n")

    for epoch in range(num_epochs):
        # ===== LEARNING RATE WARMUP =====
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # ===== TRAINING PHASE =====
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Track per-class accuracy
        class_correct = torch.zeros(num_classes, device=device)
        class_total = torch.zeros(num_classes, device=device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            labels = batch["label"].to(device)

            if mode == "audio":
                audio = batch["audio"].to(device)
                logits = model(audio)
            elif mode == "lyrics":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
            elif mode == "fusion":
                audio_logits = batch["audio_logits"].to(device)
                lyrics_logits = batch["lyrics_logits"].to(device)
                logits = model(audio_logits, lyrics_logits)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            if gradient_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update per-class accuracy
            for label_idx in range(num_classes):
                mask = labels == label_idx
                if mask.sum() > 0:
                    class_correct[label_idx] += (preds[mask] == labels[mask]).sum()
                    class_total[label_idx] += mask.sum()

            # Update progress bar
            current_acc = correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ===== VALIDATION PHASE =====
        val_loss, val_acc, val_preds, val_labels, val_class_accs = evaluate_model(
            model, val_loader, criterion, device, mode, num_classes
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print epoch summary
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
        print(f"{'=' * 70}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Print per-class metrics
        print(f"\n  Per-class Validation Accuracy:")
        print(f"  {'-' * 50}")
        for label_idx, emotion in enumerate(EMOTIONS):
            train_class_acc = (class_correct[label_idx] / class_total[label_idx]).item() if class_total[
                                                                                                label_idx] > 0 else 0.0
            val_class_acc = val_class_accs.get(label_idx, 0.0)
            print(f"    {emotion.capitalize():8s}: Train={train_class_acc:.4f}, Val={val_class_acc:.4f}")

        # Learning rate info
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  Learning rate: {current_lr:.2e}")

        # Learning rate scheduling
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step(val_acc)

        # Save best model based on validation accuracy
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            improved = True
            print(f"\n  ✅ NEW BEST MODEL! Val Acc: {best_val_acc:.4f}, Val Loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"\n  No improvement for {epochs_without_improvement} epoch(s)")
            print(f"  Best val acc so far: {best_val_acc:.4f}")

        print(f"{'=' * 70}\n")

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n{'=' * 70}")
            print(f"🛑 EARLY STOPPING TRIGGERED")
            print(f"{'=' * 70}")
            print(f"  No improvement for {early_stopping_patience} consecutive epochs")
            print(f"  Best validation accuracy: {best_val_acc:.4f}")
            print(f"  Stopping at epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 70}\n")
            break

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"🎉 TRAINING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"  Total epochs trained: {epoch + 1}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  Model saved to: {save_path}")
    print(f"{'=' * 70}\n")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
    }