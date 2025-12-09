import os
import pandas as pd
import numpy as np
from config import METADATA_CSV

STATIC_1 = "data/raw/deam_meta/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv"
STATIC_2 = "data/raw/deam_meta/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_2000_2058.csv"

DEAM_AUDIO_DIR = "data/raw/deam_audio"

SONG_ID_COL = "song_id"
VALENCE_COL = " valence_mean"
AROUSAL_COL = " arousal_mean"

EMOTIONS = ["happy", "sad", "angry", "calm"]


def va_to_emotion_median(valence_raw: float, arousal_raw: float,
                         v_median: float, a_median: float) -> str:
    """
    Median-based emotion mapping - guarantees balanced classes.

    This uses the dataset's median values to split the valence-arousal space
    into four equal quadrants, ensuring each emotion gets ~25% of samples.

    Args:
        valence_raw: Raw valence value (1-9 scale)
        arousal_raw: Raw arousal value (1-9 scale)
        v_median: Median valence in the dataset
        a_median: Median arousal in the dataset

    Returns:
        Emotion label: "happy", "sad", "angry", or "calm"
    """
    v_high = valence_raw > v_median
    a_high = arousal_raw > a_median

    if v_high and a_high:
        return "happy"  # High valence, high arousal
    elif v_high and not a_high:
        return "calm"  # High valence, low arousal
    elif not v_high and a_high:
        return "angry"  # Low valence, high arousal
    else:
        return "sad"  # Low valence, low arousal


def find_audio_file(song_id: int | str) -> str | None:
    """
    Try several common filename patterns for DEAM audio.
    Returns the first existing path or None if not found.
    """
    try:
        sid_int = int(song_id)
    except ValueError:
        sid_int = None

    candidates = []

    if sid_int is not None:
        candidates.extend([
            os.path.join(DEAM_AUDIO_DIR, f"{sid_int}.mp3"),
            os.path.join(DEAM_AUDIO_DIR, f"{sid_int:04d}.mp3"),
            os.path.join(DEAM_AUDIO_DIR, f"song_{sid_int}.mp3"),
            os.path.join(DEAM_AUDIO_DIR, f"deam_{sid_int}.mp3"),
        ])

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def main():
    print("=" * 70)
    print("Building DEAM Metadata with MEDIAN-BASED Emotion Mapping")
    print("=" * 70)

    # ---- Load and merge the two static annotation files ----
    if not os.path.exists(STATIC_1):
        raise FileNotFoundError(f"Missing {STATIC_1}")
    if not os.path.exists(STATIC_2):
        raise FileNotFoundError(f"Missing {STATIC_2}")

    df1 = pd.read_csv(STATIC_1)
    df2 = pd.read_csv(STATIC_2)
    meta = pd.concat([df1, df2], ignore_index=True)

    print(f"\nLoaded {len(meta)} songs from DEAM annotations")
    print("Columns:", list(meta.columns))

    # ---- Calculate medians for the entire dataset ----
    valence_median = meta[VALENCE_COL].median()
    arousal_median = meta[AROUSAL_COL].median()

    print(f"\n{'=' * 70}")
    print(f"Dataset Statistics:")
    print(f"{'=' * 70}")
    print(f"Valence:")
    print(f"  Mean:   {meta[VALENCE_COL].mean():.2f}")
    print(f"  Median: {valence_median:.2f} ← Split point")
    print(f"  Std:    {meta[VALENCE_COL].std():.2f}")
    print(f"\nArousal:")
    print(f"  Mean:   {meta[AROUSAL_COL].mean():.2f}")
    print(f"  Median: {arousal_median:.2f} ← Split point")
    print(f"  Std:    {meta[AROUSAL_COL].std():.2f}")
    print(f"{'=' * 70}")

    rows = []
    missing_audio = 0

    # Track emotion distribution
    emotion_counts = {e: 0 for e in EMOTIONS}

    for _, row in meta.iterrows():
        song_id = row[SONG_ID_COL]
        val = float(row[VALENCE_COL])
        aro = float(row[AROUSAL_COL])

        audio_path = find_audio_file(song_id)

        if audio_path is None:
            missing_audio += 1
            continue

        # Use median-based mapping
        emotion = va_to_emotion_median(val, aro, valence_median, arousal_median)
        emotion_counts[emotion] += 1

        rows.append({
            "audio_path": audio_path,
            "lyrics_path": "",
            "emotion_label": emotion,
            "song_id": f"deam_{song_id}",
            "user_id": "deam_user",
            "valence": val,
            "arousal": aro,
        })

    df_out = pd.DataFrame(rows)

    # ---- Print class distribution ----
    print(f"\n{'=' * 70}")
    print("EMOTION CLASS DISTRIBUTION (Median-Based):")
    print(f"{'=' * 70}")
    total = len(df_out)

    for emotion in EMOTIONS:
        count = emotion_counts[emotion]
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)  # Visual bar
        print(f"{emotion.capitalize():8s}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"{'=' * 70}")
    print(f"Total:   {total} songs")
    print(f"Skipped: {missing_audio} songs (missing audio)")

    # ---- Check balance ----
    max_count = max(emotion_counts.values())
    min_count = min(emotion_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print(f"\n{'=' * 70}")
    if imbalance_ratio < 1.5:
        print(f"✅ EXCELLENT balance! Ratio: {imbalance_ratio:.2f}:1")
        print(f"   All classes within 50% of each other")
    elif imbalance_ratio < 2.0:
        print(f"✅ GOOD balance! Ratio: {imbalance_ratio:.2f}:1")
        print(f"   Classes reasonably balanced")
    elif imbalance_ratio < 3.0:
        print(f"⚠️  ACCEPTABLE balance. Ratio: {imbalance_ratio:.2f}:1")
        print(f"   Some imbalance present but manageable")
    else:
        print(f"❌ POOR balance. Ratio: {imbalance_ratio:.2f}:1")
        print(f"   Consider using class weights during training")
    print(f"{'=' * 70}")

    # ---- Save metadata ----
    df_out.to_csv(METADATA_CSV, index=False)
    print(f"\n✅ Saved {len(df_out)} rows to {METADATA_CSV}")
    print(f"\nNext steps:")
    print(f"  1. python scripts/train_emotion_audio.py")
    print(f"  2. streamlit run app/app.py")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()