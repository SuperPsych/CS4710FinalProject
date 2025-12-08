import librosa
import numpy as np
from config import AUDIO_SR, N_MELS, SPEC_LEN

def load_mel_spectrogram(path: str):
    y, sr = librosa.load(path, sr=AUDIO_SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)

    if S_dB.shape[1] < SPEC_LEN:
        pad_width = SPEC_LEN - S_dB.shape[1]
        S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode="constant")
    elif S_dB.shape[1] > SPEC_LEN:
        S_dB = S_dB[:, :SPEC_LEN]
    return S_dB.astype(np.float32)