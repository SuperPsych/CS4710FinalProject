# scripts/build_deam_metadata_dynamic.py

import os
import pandas as pd
from config import METADATA_CSV

# Paths to the DEAM dynamic annotation files (based on your folder layout)
AROUSAL_CSV = "data/raw/deam_meta/annotations_averaged_per_song/dynamic/arousal.csv"
VALENCE_CSV = "data/raw/deam_meta/annotations_averaged_per_song/dynamic/valence.csv"

DEAM_AUDIO_DIR = "data/raw/deam_audio"

EMOTIONS = ["happy", "sad", "angry", "calm"]


def va_to_emotion(valence: float, arousal: float) -> str:
    """
    Map DEAM dynamic valence/arousal (already roughly in [-1, 1])
    into 4 discrete emotions using quadrants.

      v >= 0, a >= 0 -> happy
      v >= 0, a <  0 -> calm
      v <  0, a >= 0 -> angry
      v <  0, a <  0 -> sad

    You can tweak thresholds if you want a neutral band, but this
    matches the 4-class setup (happy/calm/angry/sad).
    """

    # optional: small dead-zone around 0; comment out if not wanted
    # TH = 0.05
    # if abs(valence) < TH:
    #     valence = 0.0
    # if abs(arousal) < TH:
    #     arousal = 0.0

    if valence >= 0 and arousal >= 0:
        return "happy"
    if valence >= 0 and arousal < 0:
        return "calm"
    if valence < 0 and arousal >= 0:
        return "angry"
    return "sad"


def find_audio_file(song_id) -> str | None:
    """
    Try several common filename patterns for DEAM audio.
    This reuses the idea we had before. If you know the exact
    pattern of your .wav files, you can tweak the candidates list.
    """
    try:
        sid_int = int(song_id)
    except (ValueError, TypeError):
        sid_int = None

    candidates = []

    if sid_int is not None:
        candidates.extend([
            os.path.join(DEAM_AUDIO_DIR, f"{sid_int}.wav"),
            os.path.join(DEAM_AUDIO_DIR, f"{sid_int:04d}.wav"),
            os.path.join(DEAM_AUDIO_DIR, f"song_{sid_int}.wav"),
            os.path.join(DEAM_AUDIO_DIR, f"deam_{sid_int}.wav"),
        ])

    song_str = str(song_id)
    candidates.extend([
        os.path.join(DEAM_AUDIO_DIR, f"{song_str}.wav"),
        os.path.join(DEAM_AUDIO_DIR, f"{song_str}.mp3"),
    ])

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    # ---- 1. Load dynamic arousal and valence ----
    if not os.path.exists(AROUSAL_CSV):
        raise FileNotFoundError(f"Missing {AROUSAL_CSV}")
    if not os.path.exists(VALENCE_CSV):
        raise FileNotFoundError(f"Missing {VALENCE_CSV}")

    arousal = pd.read_csv(AROUSAL_CSV)
    valence = pd.read_csv(VALENCE_CSV)

    # Just in case, strip whitespace in column names
    arousal.columns = [c.strip() for c in arousal.columns]
    valence.columns = [c.strip() for c in valence.columns]

    # ---- 2. Merge on song_id ----
    # both have 'song_id' + many sample_* columns
    merged = pd.merge(arousal, valence, on="song_id", suffixes=("_aro", "_val"))

    # identify dynamic columns
    aro_cols = [c for c in merged.columns if c.startswith("sample_") and c.endswith("_aro")]
    val_cols = [c for c in merged.columns if c.startswith("sample_") and c.endswith("_val")]

    if not aro_cols or not val_cols:
        raise RuntimeError("Could not find dynamic sample_* columns in arousal/valence CSVs.")

    # ---- 3. Compute per-song mean valence/arousal across time ----
    # We use mean over time, ignoring NaNs (because songs have different lengths).
    merged["valence_mean_dynamic"] = merged[val_cols].mean(axis=1, skipna=True)
    merged["arousal_mean_dynamic"] = merged[aro_cols].mean(axis=1, skipna=True)

    print("Dynamic valence summary:")
    print(merged["valence_mean_dynamic"].describe())
    print("\nDynamic arousal summary:")
    print(merged["arousal_mean_dynamic"].describe())

    # ---- 4. Map to discrete emotion labels ----
    merged["emotion_label"] = merged.apply(
        lambda r: va_to_emotion(r["valence_mean_dynamic"], r["arousal_mean_dynamic"]),
        axis=1,
    )

    # ---- 5. Build metadata rows with audio_path + labels ----
    rows = []
    missing_audio = 0

    for _, row in merged.iterrows():
        song_id = row["song_id"]
        audio_path = find_audio_file(song_id)

        if audio_path is None:
            missing_audio += 1
            continue

        rows.append({
            "audio_path": audio_path,
            "lyrics_path": "",  # DEAM doesn’t provide lyrics
            "emotion_label": row["emotion_label"],
            "song_id": f"deam_{song_id}",
            "user_id": "deam_user",  # dummy user
        })

    df_out = pd.DataFrame(rows)

    # ---- 6. Save to metadata.csv (overwrites previous) ----
    df_out.to_csv(METADATA_CSV, index=False)

    print(f"\nSaved {len(df_out)} rows to {METADATA_CSV}")
    print(f"Skipped {missing_audio} rows due to missing audio files.")
    print("Done building dynamic-based metadata.")


if __name__ == "__main__":
    main()
