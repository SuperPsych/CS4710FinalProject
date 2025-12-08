import pandas as pd
df = pd.read_csv("data/metadata.csv")
print(df.head())
print(df["emotion_label"].value_counts())