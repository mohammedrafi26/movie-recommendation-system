import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Movies API
export const getPopularMovies = (page = 1) => axios.get(`${API}/movies/popular?page=${page}`);
export const getTrendingMovies = (timeWindow = 'week', page = 1) => axios.get(`${API}/movies/trending?time_window=${timeWindow}&page=${page}`);
export const getTopRatedMovies = (page = 1) => axios.get(`${API}/movies/top-rated?page=${page}`);
export const getUpcomingMovies = (page = 1) => axios.get(`${API}/movies/upcoming?page=${page}`);
export const getNowPlayingMovies = (page = 1) => axios.get(`${API}/movies/now-playing?page=${page}`);
export const searchMovies = (query, page = 1) => axios.get(`${API}/movies/search?query=${encodeURIComponent(query)}&page=${page}`);
export const getGenres = () => axios.get(`${API}/movies/genres`);
export const discoverMovies = (params) => axios.get(`${API}/movies/discover`, { params });
export const getMovieDetails = (movieId) => axios.get(`${API}/movies/${movieId}`);

// AI Recommendations
export const getAIRecommendations = (data) => axios.post(`${API}/ai/recommend`, data);

// Watchlist API
export const getWatchlist = () => axios.get(`${API}/watchlist`);
export const addToWatchlist = (movie) => axios.post(`${API}/watchlist`, movie);
export const removeFromWatchlist = (movieId) => axios.delete(`${API}/watchlist/${movieId}`);
export const checkWatchlist = (movieId) => axios.get(`${API}/watchlist/check/${movieId}`);

// Ratings API
export const getRatings = () => axios.get(`${API}/ratings`);
export const addRating = (rating) => axios.post(`${API}/ratings`, rating);
export const deleteRating = (movieId) => axios.delete(`${API}/ratings/${movieId}`);
export const getMovieRating = (movieId) => axios.get(`${API}/ratings/${movieId}`);

export default API;
