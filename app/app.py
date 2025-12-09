import streamlit as st
import torch
import pandas as pd
import os
import tempfile
import librosa
import soundfile as sf

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


def convert_audio_to_wav(audio_file, target_sr=22050):
    """
    Convert uploaded audio file to WAV format compatible with librosa.
    Supports: mp3, wav, flac, ogg, m4a, aac, wma, and more.

    Args:
        audio_file: Streamlit UploadedFile object
        target_sr: Target sample rate for conversion

    Returns:
        Path to temporary WAV file
    """
    # Create temporary file with original extension
    suffix = os.path.splitext(audio_file.name)[1].lower()
    if not suffix:
        suffix = '.audio'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_input:
        tmp_input.write(audio_file.read())
        tmp_input_path = tmp_input.name

    try:
        # Load audio using librosa (handles many formats via audioread/soundfile)
        y, sr = librosa.load(tmp_input_path, sr=target_sr, mono=True)

        # Create output WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_output:
            tmp_output_path = tmp_output.name

        # Write as WAV
        sf.write(tmp_output_path, y, sr)

        return tmp_output_path

    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{audio_file.name}': {str(e)}")
    finally:
        # Clean up input temp file
        if os.path.exists(tmp_input_path):
            try:
                os.unlink(tmp_input_path)
            except:
                pass


def load_models():
    """Load all pre-trained models."""
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
    """
    Predict emotion from uploaded audio file.

    Args:
        audio_file: Streamlit UploadedFile object
        audio_model: Pre-trained AudioEmotionCNN model

    Returns:
        Predicted emotion string
    """
    try:
        # Convert to WAV format
        wav_path = convert_audio_to_wav(audio_file)

        # Load mel spectrogram
        mel = load_mel_spectrogram(wav_path)
        x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            logits = audio_model(x)
            pred = logits.argmax(dim=-1).item()

        return EMOTIONS[pred]

    finally:
        # Clean up temporary WAV file
        if 'wav_path' in locals() and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except:
                pass


def predict_emotion_from_lyrics(text, tokenizer, lyrics_model):
    """Predict emotion from lyrics text."""
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

    st.markdown("""
    This system detects emotions from audio or lyrics, generates melodies, 
    and recommends songs tailored to your emotional state.
    """)

    st.sidebar.header("Input Options")
    mode = st.sidebar.radio(
        "Choose mode:",
        ["Upload audio", "Paste lyrics", "Select emotion"]
    )

    # Display supported formats in sidebar
    if mode == "Upload audio":
        st.sidebar.markdown("""
        **Supported formats:**
        - WAV, MP3, FLAC
        - OGG, M4A, AAC
        - WMA, and more
        """)

    audio_model, lyrics_model, tokenizer, melody_model, recommender = load_models()

    selected_emotion = None

    if mode == "Upload audio":
        audio_file = st.file_uploader(
            "Upload an audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a", "aac", "wma", "mp4", "webm"],
            help="Upload any audio file - it will be automatically converted"
        )

        if audio_file is not None:
            # Display file info
            file_details = {
                "Filename": audio_file.name,
                "File size": f"{audio_file.size / 1024:.2f} KB",
                "File type": audio_file.type or "Unknown"
            }
            st.write("**Uploaded file:**")
            for key, val in file_details.items():
                st.text(f"{key}: {val}")

            if st.button("🎧 Detect Emotion from Audio"):
                with st.spinner("Processing audio file..."):
                    try:
                        emotion = predict_emotion_from_audio(audio_file, audio_model)
                        st.success(f"✅ Detected emotion: **{emotion}**")
                        selected_emotion = emotion
                    except Exception as e:
                        st.error(f"❌ Error processing audio: {str(e)}")
                        st.info("Please try a different file or format.")

    elif mode == "Paste lyrics":
        text = st.text_area(
            "Paste song lyrics:",
            height=200,
            placeholder="Enter the lyrics of a song here..."
        )
        if text.strip() and st.button("📝 Detect Emotion from Lyrics"):
            with st.spinner("Analyzing lyrics..."):
                emotion = predict_emotion_from_lyrics(text, tokenizer, lyrics_model)
                st.success(f"✅ Detected emotion: **{emotion}**")
                selected_emotion = emotion

    else:
        selected_emotion = st.selectbox(
            "Select an emotion:",
            EMOTIONS,
            help="Choose the emotion you want to explore"
        )

    # Generate and recommend based on selected emotion
    if selected_emotion:
        st.divider()
        st.subheader(f"🎯 Working with emotion: **{selected_emotion.upper()}**")

        col1, col2 = st.columns(2)

        with col1:
            # Generate a simple melody
            st.markdown("### 🎹 Generated Melody")
            st.write("A short melody conditioned on the detected emotion:")

            start_seq = [6]  # center pitch index => some middle note
            tokens = melody_model.generate(start_seq, length=16, device=DEVICE)
            # Map token indices [0..12] to MIDI pitches [60..72]
            pitches = [60 + t for t in tokens]

            midi_path = notes_to_midi(pitches, output_path="generated_melody.mid")
            st.info(f"✅ Generated `generated_melody.mid` (in project folder)")

            # Display the note sequence
            with st.expander("View generated notes"):
                st.write(f"MIDI pitches: {pitches}")

        with col2:
            # Recommend songs
            st.markdown("### 🎧 Recommended Songs")
            if recommender is not None:
                recs = recommender.recommend_for_emotion(selected_emotion, top_k=10)
                if recs:
                    st.write(f"Top {len(recs)} songs for **{selected_emotion}** mood:")
                    for idx, song in enumerate(recs, 1):
                        st.write(f"{idx}. {song}")
                else:
                    st.warning("No songs found for this emotion.")
                    st.info("Check that `song_features.csv` has songs labeled with this emotion.")
            else:
                st.error("❌ Recommender not available")
                st.info("Run `scripts/build_recommender_index.py` first to build the song index.")


if __name__ == "__main__":
    main()