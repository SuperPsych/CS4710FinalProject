import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train_classifier(
    model,
    dataset,
    device,
    save_path,
    batch_size=16,
    num_epochs=5,
    lr=1e-4,
    mode="audio",
):
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")