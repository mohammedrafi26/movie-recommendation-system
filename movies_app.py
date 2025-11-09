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
import requests
    API_KEY = "ea6489e0f7fb8a885e72fdec213d85b6"
   def fetch_poster(title, year=None):
    try:
        # Search for the movie by title (and optionally year)
        query = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"
        if year:
            query += f"&year={year}"
        
        response = requests.get(query)
        data = response.json()
        
        if data["results"]:
            poster_path = data["results"][0]["poster_path"]
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
        return "https://via.placeholder.com/500x750?text=No+Image"
    
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
        return "https://via.placeholder.com/500x750?text=No+Image"
       

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
            poster = fetch_poster(title)
            with cols[idx % 5]:
                st.image(poster, caption=f"{title} ({year})", width=150)
    else:
        st.error("Movie not found in database. Try another title.")


    
        









