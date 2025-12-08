AI-Based Music Generation and Recommendation System
===================================================

This project implements:

1. Emotion detection from music (audio + lyrics)
2. Emotion-conditioned music generation (simple melody generator)
3. Hybrid recommendation system combining emotion and similarity
4. A simple Streamlit UI to interact with the system

Usage (example):

1. Prepare data/metadata.csv with columns:
   - audio_path
   - lyrics_path
   - emotion_label
   - user_id (optional, for recommendations)
   - song_id

2. Run preprocessing:
   python scripts/preprocess_data.py

3. Train models (example):
   python scripts/train_emotion_audio.py
   python scripts/train_emotion_lyrics.py
   python scripts/train_fusion.py
   python scripts/train_music_generator.py
   python scripts/build_recommender_index.py

4. Start app:
   streamlit run app/app.py