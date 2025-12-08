import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

def evaluate_classifier(model, dataset, device, mode="audio"):
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_labels = []
    all_preds = []

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

            preds = logits.argmax(dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    print(classification_report(all_labels, all_preds))