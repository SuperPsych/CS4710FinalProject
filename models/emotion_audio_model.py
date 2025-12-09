import torch
import torch.nn as nn

class AudioEmotionCNN(nn.Module):
    """
    Lightweight CNN on mel-spectrograms: input [B, 1, N_MELS, SPEC_LEN]
    Much faster on CPU than the original version.
    """
    def __init__(self, num_emotions: int, base_channels: int = 8):
        super().__init__()

        c1 = base_channels          # 8 by default
        c2 = base_channels * 2      # 16
        c3 = base_channels * 4      # 32

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        # downsample by 2

            # Block 2
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),

            # Global average pool → [B, c3, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                  # [B, c3]
            nn.Dropout(0.3),
            nn.Linear(c3, num_emotions),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
