import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import requests
import streamlit as st

# ================== Streamlit UI ==================
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations with posters — powered by TMDB!")

# ================== Load Dataset ==================
try:
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    st.success("✅ Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

if not {'movieId', 'title', 'genres'}.issubset(movies.columns):
    st.error("❌ Required columns ('movieId', 'title', 'genres') not found in your CSV.")
    st.stop()

movies = movies[['movieId', 'title', 'genres']].fillna('')

# Extract year from title if present
def extract_year(title):
    if "(" in title and ")" in title:
        return title.split("(")[-1].replace(")", "")
    return ""

movies['year'] = movies['title'].apply(extract_year)

# ================== TF-IDF Model ==================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)

# ================== TMDB API ==================
API_KEY = "ea6489e0f7fb8a885e72fdec213d85b6"  # your API key

def fetch_poster(movie_title):
    try:
        # Clean movie title (remove year)
        query = movie_title.split("(")[0].strip()

        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"

        response = requests.get(url)
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

        return "https://via.placeholder.com/300x450?text=No+Poster"
    except:
        return "https://via.placeholder.com/300x450?text=Error"


# ================== Recommendation Logic ==================
def recommend(movie_title, n=5):
    matches = movies[movies['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return []
    idx = matches.index[0]

    distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)
    recs = [(movies.iloc[i]['title'], movies.iloc[i]['genres']) for i in indices.flatten()[1:]]
    return recs

# ================== Streamlit UI ==================
movie_name = st.text_input("🎥 Enter a movie you like:")

if st.button("Recommend"):
    if not movie_name.strip():
        st.warning("⚠️ Please enter a movie title.")
    else:
        recommendations = recommend(movie_name)
        if recommendations:
            st.subheader("✨ Recommended Movies:")
            cols = st.columns(5)

            for idx, (title, genres) in enumerate(recommendations):
                poster = fetch_poster(title)

                with cols[idx % 5]:
                    st.image(poster, caption=title, width=150)
                    st.caption(f"🎭 {genres}")

        else:
            st.error("❌ Movie not found. Try another title.")

    
        

