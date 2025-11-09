import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import requests
import streamlit as st

# =============== Load Dataset ===============
movies = pd.read_csv("movies.csv", low_memory=False)

st.write("Columns in your CSV:", movies.columns.tolist())
# Keep only useful columns
movies = movies[['movieId', 'title', 'genres']]
movies['overview'] = movies['genres']  # use genres as text for similarity
movies['release_date'] = None

# Extract year from release_date
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['year'] = movies['release_date'].dt.year

# =============== TF-IDF Vectorization ===============
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Fit Nearest Neighbors Model
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)

# =============== Recommendation Function ===============
def recommend(movie_title, n=5):
    if movie_title not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie_title].index[0]
    distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)

    recommended_movies = []
    for i in indices.flatten()[1:]:  # exclude itself
        title = movies.iloc[i]['title']
        year = movies.iloc[i]['year']
        recommended_movies.append((title, year))
    return recommended_movies

# =============== Fetch Poster from TMDB API ===============
def fetch_poster(title):
    API_KEY = "4cf08ed379ac1a8b6ca6432aac08db10"
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            for movie in data["results"]:
                if movie.get("poster_path"):  # ✅ pick first valid poster
                    return f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"

        # fallback
        return "https://via.placeholder.com/200x300?text=No+Poster+Found"
    except Exception as e:
        print("Poster fetch error:", e)
        return "https://via.placeholder.com/200x300?text=Error"


# =============== Streamlit App ===============
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations with posters!")

movie_name = st.text_input("Enter a movie you like:")

if st.button("Recommend"):
    recommendations = recommend(movie_name)
    if recommendations:
        st.subheader("Recommended Movies:")
        cols = st.columns(5)
        for idx, (title, year) in enumerate(recommendations):
            poster = fetch_poster(title, year)
            with cols[idx % 5]:
                st.image(poster, caption=f"{title} ({year})", width=150)
    else:
        st.error("Movie not found in database. Try another title.")

    
        





