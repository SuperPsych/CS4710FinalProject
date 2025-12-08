import pandas as pd
from config import METADATA_CSV

def main():
    # TODO: implement real preprocessing for your specific datasets.
    # Example metadata format:
    df = pd.DataFrame([
        {
            "audio_path": "data/raw/example1.wav",
            "lyrics_path": "data/raw/example1.txt",
            "emotion_label": "happy",
            "song_id": "song1",
            "user_id": "user1",
        },
    ])
    df.to_csv(METADATA_CSV, index=False)
    print(f"Saved example metadata to {METADATA_CSV}")

if __name__ == "__main__":
    main()