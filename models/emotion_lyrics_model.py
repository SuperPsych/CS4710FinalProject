import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LyricsEmotionBERT(nn.Module):
    """
    Wrapper on top of a pretrained transformer (e.g. distilbert-base-uncased).
    """
    def __init__(self, model_name: str, num_emotions: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_emotions)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits

def get_lyrics_tokenizer(model_name: str = "distilbert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)
