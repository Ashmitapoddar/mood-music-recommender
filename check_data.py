# Helper script to clean raw Spotify dataset and generate songs.csv

import pandas as pd

df = pd.read_csv("high_popularity_spotify_data.csv")

print("Columns:", df.columns)
print("\nSample rows:")
print(df.head())

print("\nValence range:", df["valence"].min(), "-", df["valence"].max())
print("Energy range:", df["energy"].min(), "-", df["energy"].max())


# Load original data
df = pd.read_csv("high_popularity_spotify_data.csv")

# Select only the needed columns
df_clean = df[["track_name", "track_artist", "valence", "energy", "track_album_name","track_href","track_album_release_date","track_popularity"]]

# Drop rows with missing values
df_clean = df_clean.dropna()

# Drop duplicates (optional)
df_clean = df_clean.drop_duplicates()

# Save the cleaned version
df_clean.to_csv("songs.csv", index=False)

print("✅ Cleaned songs.csv file created!")
