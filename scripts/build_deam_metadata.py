import os
import pandas as pd
from config import METADATA_CSV

STATIC_1 = "data/raw/deam_meta/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_1_2000.csv"
STATIC_2 = "data/raw/deam_meta/annotations_averaged_per_song/song_level/static_annotations_averaged_songs_2000_2058.csv"

DEAM_AUDIO_DIR = "data/raw/deam_audio"

# ---- 1. CONFIG: adjust these if your column names differ ----
# Open one of the CSVs once to confirm these names.
SONG_ID_COL    = "song_id"       # or "songID" etc.
VALENCE_COL    = "valence_mean"  # e.g. "valence_mean"
AROUSAL_COL    = "arousal_mean"  # e.g. "arousal_mean"

# We’ll map valence/arousal → these 4 labels
EMOTIONS = ["happy", "sad", "angry", "calm"]


def va_to_emotion(valence: float, arousal: float) -> str:
    """
    Simple quadrant-based mapping.
    Assumes valence/arousal roughly in [-1, 1].
    Tweak thresholds if needed (e.g., 0.0 → 0.1).
    """
    if valence >= 0 and arousal >= 0:
        return "happy"   # high valence, high arousal
    if valence >= 0 and arousal < 0:
        return "calm"    # high valence, low arousal
    if valence < 0 and arousal >= 0:
        return "angry"   # low valence, high arousal
    return "sad"         # low valence, low arousal


def find_audio_file(song_id: int | str) -> str | None:
    """
    Try several common filename patterns for DEAM audio.
    This makes the script robust even if the exact naming is a bit different.
    Returns the first existing path or None if not found.
    """

    # normalize to int if possible (because DEAM song ids are 1..2058)
    try:
        sid_int = int(song_id)
    except ValueError:
        sid_int = None

    candidates = []

    # Generic guesses (you can add/remove patterns once you see your filenames)
    if sid_int is not None:
        candidates.extend([
            os.path.join(DEAM_AUDIO_DIR, f"{sid_int}.wav"),
            os.path.join(DEAM_AUDIO_DIR, f"{sid_int:04d}.wav"),
            os.path.join(DEAM_AUDIO_DIR, f"song_{sid_int}.wav"),
            os.path.join(DEAM_AUDIO_DIR, f"deam_{sid_int}.wav"),
        ])

    # Also try string-based directly
    song_str = str(song_id)
    candidates.extend([
        os.path.join(DEAM_AUDIO_DIR, f"{song_str}.wav"),
        os.path.join(DEAM_AUDIO_DIR, f"{song_str}.mp3"),
    ])

    for path in candidates:
        if os.path.exists(path):
            return path

    # If nothing matched, return None
    return None


def main():
    # ---- 2. Load and merge the two static annotation files ----
    if not os.path.exists(STATIC_1):
        raise FileNotFoundError(f"Missing {STATIC_1}")
    if not os.path.exists(STATIC_2):
        raise FileNotFoundError(f"Missing {STATIC_2}")

    df1 = pd.read_csv(STATIC_1)
    df2 = pd.read_csv(STATIC_2)
    meta = pd.concat([df1, df2], ignore_index=True)

    # Quick sanity check on columns
    print("Columns in static annotation file:", list(meta.columns))

    rows = []
    missing_audio = 0

    for _, row in meta.iterrows():
        song_id = row[SONG_ID_COL]
        val = float(row[VALENCE_COL])
        aro = float(row[AROUSAL_COL])

        audio_path = find_audio_file(song_id)

        if audio_path is None:
            missing_audio += 1
            continue

        emotion = va_to_emotion(val, aro)

        rows.append({
            "audio_path": audio_path,
            "lyrics_path": "",   # DEAM has no lyrics
            "emotion_label": emotion,
            "song_id": f"deam_{song_id}",
            "user_id": "deam_user",  # dummy user for now
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(METADATA_CSV, index=False)

    print(f"Saved {len(df_out)} rows to {METADATA_CSV}")
    print(f"Skipped {missing_audio} rows due to missing audio files.")
    print("Done.")


if __name__ == "__main__":
    main()