import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import requests
import streamlit as st

# =============== Load Dataset ===============
st.title("🎬 Movie Recommendation System")
st.write("Get smart movie recommendations with posters — based on genres!")

try:
    movies = pd.read_csv("movies.csv", low_memory=False)
    st.success("✅ Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Show CSV columns and preview
st.write("**Columns in your CSV:**", movies.columns.tolist())
st.dataframe(movies.head())

# Keep only useful columns
movies = movies[['movieId', 'title', 'genres']]

# Fill missing genres if any
movies['genres'] = movies['genres'].fillna('')

# =============== TF-IDF Vectorization ===============
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Fit Nearest Neighbors Model
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)

# =============== Recommendation Function ===============
def recommend(movie_title, n=5):
    # Case-insensitive search
    matches = movies[movies['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return []

    idx = matches.index[0]
    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=n + 1)

    recommended_movies = []
    for i in indices.flatten()[1:]:  # exclude itself
        title = movies.iloc[i]['title']
        genres = movies.iloc[i]['genres']
        recommended_movies.append((title, genres))
    return recommended_movies


# =============== Fetch Poster from TMDB API ===============
API_KEY = "4cf08ed379ac1a8b6ca6432aac08db10"

def fetch_poster(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("results"):
            for movie in data["results"]:
                if movie.get("poster_path"):
                    return f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
        # fallback
        return "https://via.placeholder.com/200x300?text=No+Poster"
    except Exception as e:
        print("Poster fetch error:", e)
        return "https://via.placeholder.com/200x300?text=Error"


# =============== Streamlit UI ===============
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
                    st.image(poster, caption=f"{title}", width=150)
                    st.caption(f"🎭 Genres: {genres}")
        else:
            st.error("❌ Movie not found in database. Try another title.")


    
        








