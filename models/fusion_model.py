import torch
import torch.nn as nn

class FusionEmotionModel(nn.Module):
    """
    Fuse audio and lyrics logits.
    Can be a simple weighted average or a small MLP.
    """
    def __init__(self, num_emotions: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_emotions * 2, num_emotions),
            nn.ReLU(),
            nn.Linear(num_emotions, num_emotions),
        )

    def forward(self, audio_logits, lyrics_logits):
        x = torch.cat([audio_logits, lyrics_logits], dim=-1)
        return self.fc(x)