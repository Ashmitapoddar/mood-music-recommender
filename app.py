import streamlit as st
import pandas as pd
from transformers import pipeline

import warnings
import asyncio



try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

warnings.filterwarnings("ignore", message=".*torch::class_.*")

warnings.filterwarnings("ignore", message=".*torch::class_.*", category=UserWarning)



# Page setup
st.set_page_config(page_title="Mood Music Recommender", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("songs.csv")

df = load_data()

# Mood Mapping Logic
def map_mood(valence, energy):
    if valence >= 0.6 and energy >= 0.6:
        return "happy"
    elif valence <= 0.4 and energy <= 0.4:
        return "sad"
    elif energy >= 0.7 and valence < 0.5:
        return "angry"
    elif energy <= 0.4 and valence >= 0.5:
        return "relaxed"
    elif 0.4 <= valence <= 0.7 and 0.4 <= energy <= 0.7:
        return "romantic"
    else:
        return "neutral"

if 'mood' not in df.columns:
    df["mood"] = df.apply(lambda row: map_mood(row["valence"], row["energy"]), axis=1)

# Load NLP classifier
@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

classifier = load_classifier()

# User input
st.title("🎵 AI-Based Mood Music Recommender")
st.markdown("Tell us how you're feeling and get song recommendations that match your mood! 💬")

user_input = st.text_input("How are you feeling today?", placeholder="e.g., I’m feeling tired and peaceful...")

if user_input:
    emotion_prediction = classifier(user_input)[0]
    detected_emotion = emotion_prediction[0]['label'].lower()

    
    # Map emotions to your predefined moods
    emotion_to_mood = {
        "joy": "happy",
        "anger": "angry",
        "sadness": "sad",
        "fear": "relaxed",
        "love": "romantic",
        "surprise": "neutral"
    }

    # Get the corresponding mood
    detected_mood = emotion_to_mood.get(detected_emotion, "neutral")

    st.success(f"Detected mood: **{detected_mood.capitalize()}** (based on emotion: *{detected_emotion}*)")

    # Filter songs
    filtered_songs = df[df["mood"] == detected_mood][["track_name", "track_artist","track_album_name","track_href","track_popularity","track_album_release_date"]].drop_duplicates().sample(frac=1, random_state=None).head(10)  # This shuffles the songs randomly



    if not filtered_songs.empty:
        st.subheader(f"🎧 Songs for mood: *{detected_mood.capitalize()}*")
        for _, row in filtered_songs.iterrows():
            st.markdown(f"""
                <div style="
                    background-color: green;
                    border-left: 6px solid #6c63ff;
                    padding: 15px 20px;
                    margin: 10px 0 20px 0;
                    border-radius: 12px;
                    color: white;
                ">
                    <h4 style="margin-bottom: 6px;">🎵 {row['track_name']}</h4>
                    <p style="margin: 0;">🎤 <strong>{row['track_artist']}</strong></p>
                    <p style="margin: 0;">💿 Album: {row['track_album_name']}</p>
                     <p style="margin: 0;">💿 Album: {row['track_href']}</p>
                    <p style="margin: 0;">⏱️ Duration: {row['track_album_release_date']}</p>
                    <p style="margin: 0;">📊 Popularity: {row['track_popularity']}</p>
                  
                </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("😕 No songs found for this mood. Try describing your mood differently.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ by Mizi using NLP, Streamlit & Spotify</p>", unsafe_allow_html=True)
