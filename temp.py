"""
Analyze the valence/arousal distribution to find optimal thresholds.
This will help fix the emotion mapping.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load the DEAM metadata
STATIC_1 = "data/raw/deam_meta/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv"
STATIC_2 = "data/raw/deam_meta/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_2000_2058.csv"

VALENCE_COL = " valence_mean"
AROUSAL_COL = " arousal_mean"


def analyze_distribution():
    """Analyze the actual valence/arousal distribution."""

    df1 = pd.read_csv(STATIC_1)
    df2 = pd.read_csv(STATIC_2)
    df = pd.concat([df1, df2], ignore_index=True)

    valence = df[VALENCE_COL].values
    arousal = df[AROUSAL_COL].values

    print("="*70)
    print("VALENCE/AROUSAL DISTRIBUTION ANALYSIS")
    print("="*70)

    print(f"\nValence statistics:")
    print(f"  Min:    {valence.min():.2f}")
    print(f"  25%:    {np.percentile(valence, 25):.2f}")
    print(f"  Median: {np.percentile(valence, 50):.2f}")
    print(f"  75%:    {np.percentile(valence, 75):.2f}")
    print(f"  Max:    {valence.max():.2f}")
    print(f"  Mean:   {valence.mean():.2f}")
    print(f"  Std:    {valence.std():.2f}")

    print(f"\nArousal statistics:")
    print(f"  Min:    {arousal.min():.2f}")
    print(f"  25%:    {np.percentile(arousal, 25):.2f}")
    print(f"  Median: {np.percentile(arousal, 50):.2f}")
    print(f"  75%:    {np.percentile(arousal, 75):.2f}")
    print(f"  Max:    {arousal.max():.2f}")
    print(f"  Mean:   {arousal.mean():.2f}")
    print(f"  Std:    {arousal.std():.2f}")

    # Test different mapping strategies
    print("\n" + "="*70)
    print("TESTING DIFFERENT EMOTION MAPPING STRATEGIES")
    print("="*70)

    strategies = {
        "Current (threshold=0.15)": lambda v, a: map_emotion_v1(v, a, 0.15),
        "Median-based": lambda v, a: map_emotion_median(v, a, valence, arousal),
        "Percentile 33/66": lambda v, a: map_emotion_percentile(v, a, valence, arousal, 33, 66),
        "Percentile 25/75": lambda v, a: map_emotion_percentile(v, a, valence, arousal, 25, 75),
        "Higher threshold (0.25)": lambda v, a: map_emotion_v1(v, a, 0.25),
        "Lower threshold (0.10)": lambda v, a: map_emotion_v1(v, a, 0.10),
    }

    for strategy_name, mapping_func in strategies.items():
        emotions = [mapping_func(v, a) for v, a in zip(valence, arousal)]
        counts = Counter(emotions)

        print(f"\n{strategy_name}:")
        total = len(emotions)
        for emotion in ["happy", "sad", "angry", "calm"]:
            count = counts.get(emotion, 0)
            pct = count / total * 100
            print(f"  {emotion:8s}: {count:4d} ({pct:5.1f}%)")

        # Calculate balance metric
        max_count = max(counts.values())
        min_count = min(counts.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"  Balance ratio: {ratio:.2f}:1", end="")
        if ratio < 1.5:
            print(" ✅ EXCELLENT")
        elif ratio < 2.0:
            print(" ✅ GOOD")
        elif ratio < 3.0:
            print(" ⚠️  ACCEPTABLE")
        else:
            print(" ❌ POOR")

    # Visualize the best strategy
    print("\n" + "="*70)
    print("RECOMMENDED STRATEGY")
    print("="*70)

    # Find best strategy (lowest ratio)
    best_strategy = None
    best_ratio = float('inf')

    for strategy_name, mapping_func in strategies.items():
        emotions = [mapping_func(v, a) for v, a in zip(valence, arousal)]
        counts = Counter(emotions)
        max_count = max(counts.values())
        min_count = min(counts.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')

        if ratio < best_ratio:
            best_ratio = ratio
            best_strategy = strategy_name

    print(f"\n✅ Best strategy: {best_strategy}")
    print(f"   Balance ratio: {best_ratio:.2f}:1")

    return df, best_strategy


def map_emotion_v1(v, a, threshold):
    """Current mapping with adjustable threshold."""
    # Normalize to [-1, 1]
    v_norm = (v - 5.0) / 4.0
    a_norm = (a - 5.0) / 4.0

    if a_norm > threshold:
        if v_norm > threshold:
            return "happy"
        elif v_norm < -threshold:
            return "angry"
        else:
            return "happy" if v_norm >= 0 else "angry"
    elif a_norm < -threshold:
        if v_norm > threshold:
            return "calm"
        elif v_norm < -threshold:
            return "sad"
        else:
            return "calm"
    else:
        if v_norm > threshold * 1.5:
            return "happy"
        elif v_norm < -threshold * 1.5:
            return "sad"
        else:
            return "calm" if v_norm > 0 else "sad"


def map_emotion_median(v, a, all_valence, all_arousal):
    """Use median splits."""
    v_median = np.median(all_valence)
    a_median = np.median(all_arousal)

    v_high = v > v_median
    a_high = a > a_median

    if v_high and a_high:
        return "happy"
    elif v_high and not a_high:
        return "calm"
    elif not v_high and a_high:
        return "angry"
    else:
        return "sad"


def map_emotion_percentile(v, a, all_valence, all_arousal, low_pct, high_pct):
    """Use percentile-based splits."""
    v_low = np.percentile(all_valence, low_pct)
    v_high = np.percentile(all_valence, high_pct)
    a_low = np.percentile(all_arousal, low_pct)
    a_high = np.percentile(all_arousal, high_pct)

    # Three zones for each dimension
    if v > v_high:
        v_zone = "high"
    elif v < v_low:
        v_zone = "low"
    else:
        v_zone = "mid"

    if a > a_high:
        a_zone = "high"
    elif a < a_low:
        a_zone = "low"
    else:
        a_zone = "mid"

    # Map to emotions
    if a_zone == "high":
        if v_zone == "high" or v_zone == "mid":
            return "happy"
        else:
            return "angry"
    elif a_zone == "low":
        if v_zone == "high" or v_zone == "mid":
            return "calm"
        else:
            return "sad"
    else:  # mid arousal
        if v_zone == "high":
            return "happy"
        elif v_zone == "low":
            return "sad"
        else:
            return "calm"


def visualize_quadrants(df):
    """Create a scatter plot showing emotion quadrants."""
    try:
        import matplotlib.pyplot as plt

        valence = df[VALENCE_COL].values
        arousal = df[AROUSAL_COL].values

        # Normalize
        v_norm = (valence - 5.0) / 4.0
        a_norm = (arousal - 5.0) / 4.0

        # Map to emotions using best strategy
        emotions = [map_emotion_percentile(v, a, valence, arousal, 25, 75)
                   for v, a in zip(valence, arousal)]

        # Color map
        color_map = {
            "happy": "yellow",
            "sad": "blue",
            "angry": "red",
            "calm": "green"
        }
        colors = [color_map[e] for e in emotions]

        plt.figure(figsize=(10, 8))
        plt.scatter(v_norm, a_norm, c=colors, alpha=0.5, s=20)

        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plt.xlabel('Valence (normalized)', fontsize=12)
        plt.ylabel('Arousal (normalized)', fontsize=12)
        plt.title('Emotion Distribution in Valence-Arousal Space', fontsize=14)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[e], label=e.capitalize())
                          for e in ["happy", "sad", "angry", "calm"]]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig('emotion_distribution.png', dpi=150)
        print("\n✅ Visualization saved as 'emotion_distribution.png'")

    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")


if __name__ == "__main__":
    df, best_strategy = analyze_distribution()
    visualize_quadrants(df)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"\n1. Update build_deam_metadata.py to use: {best_strategy}")
    print(f"2. Run: python scripts/build_deam_metadata.py")
    print(f"3. Retrain: python scripts/train_emotion_audio.py")
    print("="*70 + "\n")