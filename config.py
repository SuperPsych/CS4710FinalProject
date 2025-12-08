import os

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")

EMOTIONS = ["happy", "sad", "angry", "calm"]

AUDIO_SR = 22050
N_MELS = 128
SPEC_LEN = 128  # time frames

DEVICE = "cuda"
# DEVICE = "cpu"  # uncomment if no GPU

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)