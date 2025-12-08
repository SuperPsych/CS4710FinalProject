import pandas as pd
from config import METADATA_CSV, PROCESSED_DIR
import os

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    meta = pd.read_csv(METADATA_CSV)

    # Placeholder: in a real version, you'd load feature vectors from disk.
    # For now we use random features.
    import numpy as np
    feature_dim = 16
    features = np.random.randn(len(meta), feature_dim)

    song_features = pd.DataFrame(
        features,
        index=meta["song_id"],
        columns=[f"f{i}" for i in range(feature_dim)],
    )
    song_features["emotion_label"] = meta["emotion_label"].values

    out_path = os.path.join(PROCESSED_DIR, "song_features.csv")
    song_features.to_csv(out_path)
    print(f"Saved song_features to {out_path}")

if __name__ == "__main__":
    main()