import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class HybridRecommender:
    def __init__(self, song_features_df: pd.DataFrame, user_song_matrix: pd.DataFrame | None = None):
        """
        song_features_df: DataFrame with index = song_id, columns = features + 'emotion_label'
        user_song_matrix: DataFrame with rows=user_id, cols=song_id, values=play counts/rating
        """
        self.song_features_df = song_features_df
        self.user_song_matrix = user_song_matrix

        self.feature_matrix = song_features_df.drop(columns=["emotion_label"], errors="ignore").values
        self.song_ids = song_features_df.index.to_list()

    def recommend_for_emotion(self, emotion: str, top_k: int = 10):
        mask = self.song_features_df["emotion_label"] == emotion
        subset = self.song_features_df[mask]
        return subset.index.to_list()[:top_k]

    def recommend_for_song(self, song_id: str, top_k: int = 10):
        if song_id not in self.song_ids:
            return []
        idx = self.song_ids.index(song_id)
        sims = cosine_similarity(self.feature_matrix[idx:idx+1], self.feature_matrix)[0]
        ranked_indices = np.argsort(sims)[::-1]
        recs = [self.song_ids[i] for i in ranked_indices if self.song_ids[i] != song_id]
        return recs[:top_k]

    def recommend_for_user(self, user_id: str, top_k: int = 10):
        if self.user_song_matrix is None or user_id not in self.user_song_matrix.index:
            return []
        user_vector = self.user_song_matrix.loc[user_id].values.reshape(1, -1)
        sims = cosine_similarity(user_vector, self.user_song_matrix.values)[0]
        ranked_users = np.argsort(sims)[::-1]

        scores = np.zeros(self.user_song_matrix.shape[1])
        for u_idx in ranked_users[:10]:
            scores += self.user_song_matrix.iloc[u_idx].values

        already_listened = self.user_song_matrix.loc[user_id] > 0
        scores[already_listened.values] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_k]
        song_ids = self.user_song_matrix.columns.to_list()
        return [song_ids[i] for i in top_indices if scores[i] > -np.inf]
