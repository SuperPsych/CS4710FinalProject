import streamlit as st
import torch
import pandas as pd
import os

from config import (
    MODEL_DIR,
    PROCESSED_DIR,
    EMOTIONS,
    DEVICE,
)
from models.emotion_audio_model import AudioEmotionCNN
from models.emotion_lyrics_model import LyricsEmotionBERT, get_lyrics_tokenizer
from models.music_generator import MelodyGenerator, notes_to_midi
from models.recommender import HybridRecommender
from utils.audio_utils import load_mel_spectrogram
from utils.lyrics_utils import load_lyrics

def load_models():
    audio_model_path = os.path.join(MODEL_DIR, "audio_emotion_cnn.pt")
    lyrics_model_path = os.path.join(MODEL_DIR, "lyrics_emotion_bert.pt")

    audio_model = AudioEmotionCNN(num_emotions=len(EMOTIONS))
    if os.path.exists(audio_model_path):
        audio_model.load_state_dict(torch.load(audio_model_path, map_location=DEVICE))
    audio_model.to(DEVICE)
    audio_model.eval()

    bert_model_name = "distilbert-base-uncased"
    lyrics_model = LyricsEmotionBERT(bert_model_name, num_emotions=len(EMOTIONS))
    if os.path.exists(lyrics_model_path):
        lyrics_model.load_state_dict(torch.load(lyrics_model_path, map_location=DEVICE))
    lyrics_model.to(DEVICE)
    lyrics_model.eval()

    tokenizer = get_lyrics_tokenizer(bert_model_name)

    # Simple melody generator with a small pitch vocabulary (60–72)
    vocab_size = 13
    melody_model = MelodyGenerator(vocab_size=vocab_size)

    melody_model_path = os.path.join(MODEL_DIR, "melody_generator.pt")
    if os.path.exists(melody_model_path):
        melody_model.load_state_dict(torch.load(melody_model_path, map_location=DEVICE))
        print(f"Loaded trained melody generator from {melody_model_path}")
    else:
        print("No trained melody generator found; using randomly initialized weights.")

    melody_model.to(DEVICE)
    melody_model.eval()

    # Recommender
    sf_path = os.path.join(PROCESSED_DIR, "song_features.csv")
    if os.path.exists(sf_path):
        song_features = pd.read_csv(sf_path, index_col=0)
        recommender = HybridRecommender(song_features)
    else:
        recommender = None

    return audio_model, lyrics_model, tokenizer, melody_model, recommender

def predict_emotion_from_audio(audio_file, audio_model):
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    mel = load_mel_spectrogram(tmp_path)
    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = audio_model(x)
        pred = logits.argmax(dim=-1).item()
    return EMOTIONS[pred]

def predict_emotion_from_lyrics(text, tokenizer, lyrics_model):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = lyrics_model(input_ids, attention_mask)
        pred = logits.argmax(dim=-1).item()
    return EMOTIONS[pred]

def main():
    st.title("🎵 AI-Based Music Generation & Emotion-Aware Recommendation")

    st.sidebar.header("Input Options")
    mode = st.sidebar.radio("Choose mode:", ["Upload audio", "Paste lyrics", "Select emotion"])

    audio_model, lyrics_model, tokenizer, melody_model, recommender = load_models()

    selected_emotion = None

    if mode == "Upload audio":
        audio_file = st.file_uploader("Upload an audio file (wav)", type=["wav"])
        if audio_file is not None and st.button("Detect Emotion from Audio"):
            emotion = predict_emotion_from_audio(audio_file, audio_model)
            st.success(f"Detected emotion: **{emotion}**")
            selected_emotion = emotion

    elif mode == "Paste lyrics":
        text = st.text_area("Paste song lyrics:")
        if text.strip() and st.button("Detect Emotion from Lyrics"):
            emotion = predict_emotion_from_lyrics(text, tokenizer, lyrics_model)
            st.success(f"Detected emotion: **{emotion}**")
            selected_emotion = emotion

    else:
        selected_emotion = st.selectbox("Select an emotion:", EMOTIONS)

    if selected_emotion:
        st.subheader(f"🎯 Using emotion: {selected_emotion}")

        # Generate a simple melody
        st.markdown("### 🎹 Generated Melody (Emotion-Conditioned placeholder)")

        start_seq = [6]  # center pitch index => some middle note
        tokens = melody_model.generate(start_seq, length=16, device=DEVICE)
        # Map token indices [0..12] to MIDI pitches [60..72]
        pitches = [60 + t for t in tokens]

        midi_path = notes_to_midi(pitches, output_path="generated_melody.mid")
        st.write("A short melody has been generated as `generated_melody.mid` in the project folder.")

        # Recommend songs
        st.markdown("### 🎧 Recommended Songs")
        if recommender is not None:
            recs = recommender.recommend_for_emotion(selected_emotion, top_k=10)
            if recs:
                for s in recs:
                    st.write(f"- {s}")
            else:
                st.info("No songs found for this emotion (check song_features.csv).")
        else:
            st.info("Recommender index not found. Run build_recommender_index.py first.")

if __name__ == "__main__":
    main()