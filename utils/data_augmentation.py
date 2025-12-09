import numpy as np
import librosa
import torch
from typing import Tuple


def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    Stretch or compress audio in time without changing pitch.
    rate > 1.0: faster (shorter)
    rate < 1.0: slower (longer)
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: int = 0) -> np.ndarray:
    """
    Shift pitch up or down by n_steps semitones.
    Positive n_steps: higher pitch
    Negative n_steps: lower pitch
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add Gaussian noise to audio.
    """
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def time_mask(mel_spec: np.ndarray, max_mask_size: int = 10) -> np.ndarray:
    """
    Mask random time steps in mel spectrogram (SpecAugment).
    """
    spec = mel_spec.copy()
    n_mels, time_steps = spec.shape

    # Random mask size and position
    mask_size = np.random.randint(1, max_mask_size)
    mask_start = np.random.randint(0, max(1, time_steps - mask_size))

    # Apply mask (set to mean value)
    spec[:, mask_start:mask_start + mask_size] = spec.mean()
    return spec


def frequency_mask(mel_spec: np.ndarray, max_mask_size: int = 10) -> np.ndarray:
    """
    Mask random frequency bins in mel spectrogram (SpecAugment).
    """
    spec = mel_spec.copy()
    n_mels, time_steps = spec.shape

    # Random mask size and position
    mask_size = np.random.randint(1, min(max_mask_size, n_mels))
    mask_start = np.random.randint(0, max(1, n_mels - mask_size))

    # Apply mask (set to mean value)
    spec[mask_start:mask_start + mask_size, :] = spec.mean()
    return spec


def augment_mel_spectrogram(
        mel_spec: np.ndarray,
        time_mask_prob: float = 0.5,
        freq_mask_prob: float = 0.5,
) -> np.ndarray:
    """
    Apply augmentation to mel spectrogram.
    This is fast and efficient for training.
    """
    spec = mel_spec.copy()

    if np.random.random() < time_mask_prob:
        spec = time_mask(spec)

    if np.random.random() < freq_mask_prob:
        spec = frequency_mask(spec)

    return spec


class AugmentedMusicEmotionDataset(torch.utils.data.Dataset):
    """
    Wrapper around MusicEmotionDataset that applies data augmentation.
    Use this for training to increase diversity of underrepresented classes.
    """

    def __init__(
            self,
            base_dataset,
            augment_prob: float = 0.5,
            oversample_minority: bool = True,
    ):
        """
        Args:
            base_dataset: MusicEmotionDataset instance
            augment_prob: Probability of applying augmentation
            oversample_minority: If True, oversample minority classes to balance
        """
        self.base_dataset = base_dataset
        self.augment_prob = augment_prob

        # Calculate class distribution
        from collections import Counter
        labels = [base_dataset[i]["label"] for i in range(len(base_dataset))]
        self.label_counts = Counter(labels)

        # Create augmented indices for oversampling
        if oversample_minority:
            self.indices = self._create_balanced_indices(labels)
        else:
            self.indices = list(range(len(base_dataset)))

        print(f"\n📊 Dataset statistics:")
        print(f"  Original size: {len(base_dataset)}")
        print(f"  Augmented size: {len(self.indices)}")
        print(f"  Augmentation probability: {augment_prob}")

    def _create_balanced_indices(self, labels):
        """
        Create indices that oversample minority classes.
        """
        from collections import defaultdict

        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        # Find max class size
        max_count = max(len(indices) for indices in class_indices.values())

        # Oversample each class to match max_count
        balanced_indices = []
        for label, indices in class_indices.items():
            # Repeat indices to reach max_count
            repeats = max_count // len(indices)
            remainder = max_count % len(indices)

            balanced_indices.extend(indices * repeats)
            balanced_indices.extend(np.random.choice(indices, remainder, replace=False).tolist())

        # Shuffle
        np.random.shuffle(balanced_indices)
        return balanced_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get base sample
        real_idx = self.indices[idx]
        sample = self.base_dataset[real_idx]

        # Apply augmentation with probability
        if np.random.random() < self.augment_prob and "audio" in sample:
            mel_spec = sample["audio"].squeeze(0).numpy()  # Remove channel dim
            mel_spec = augment_mel_spectrogram(mel_spec)
            sample["audio"] = torch.tensor(mel_spec).unsqueeze(0)  # Add channel back

        return sample

    @property
    def emotion_to_idx(self):
        return self.base_dataset.emotion_to_idx
