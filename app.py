import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # Import joblib

@st.cache_data(persist='disk')  # Cache the data and model
def load_model_and_data():
    df = pd.read_csv('songs.csv')
    df_encoded = pd.get_dummies(df[['artist', 'genre']])
    scaler = StandardScaler()
    num_features = scaler.fit_transform(df[['danceability', 'energy', 'tempo']])
    features = pd.concat([df_encoded, pd.DataFrame(num_features)], axis=1)
    sim_matrix = cosine_similarity(features)
    return sim_matrix, df

sim_matrix, df = load_model_and_data()  # Load data and model outside the main function

def recommend(song_name, df, similarity_matrix, top_n=3):
    if song_name not in df['song_name'].values:
        return []

    idx = df[df['song_name'] == song_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_songs = [df.iloc[i[0]]['song_name'] for i in sim_scores]
    return recommended_songs

def main():
    st.title("🎵 Music Recommender (Content-Based)")

    song_list = df['song_name'].tolist()
    selected_song = st.selectbox("Choose a song to get similar recommendations:", song_list)

    if st.button("Recommend"):
        recommendations = recommend(selected_song, df, sim_matrix)
        st.subheader("Recommended Songs:")
        for song in recommendations:
            st.write(f"- {song}")

if __name__ == '__main__':
    main()