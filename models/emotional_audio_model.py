import torch
import torch.nn as nn

class AudioEmotionCNN(nn.Module):
    """
    Simple CNN on mel-spectrograms: input [B, 1, N_MELS, SPEC_LEN]
    """
    def __init__(self, num_emotions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x