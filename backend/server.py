from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import httpx
import json
import time

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# TMDB Configuration
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"

# Cache for TMDB requests
cache: Dict[str, Dict] = {}
CACHE_TTL = 3600  # 1 hour

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Pydantic Models
class WatchlistItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    movie_id: int
    title: str
    poster_path: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WatchlistCreate(BaseModel):
    movie_id: int
    title: str
    poster_path: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None

class Rating(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    movie_id: int
    title: str
    poster_path: Optional[str] = None
    rating: float
    review: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RatingCreate(BaseModel):
    movie_id: int
    title: str
    poster_path: Optional[str] = None
    rating: float
    review: Optional[str] = None

class AIRecommendRequest(BaseModel):
    mood: Optional[str] = None
    genres: Optional[List[str]] = None
    description: Optional[str] = None

# TMDB Helper Functions
async def tmdb_request(endpoint: str, params: Optional[Dict] = None, ttl: int = CACHE_TTL) -> Optional[Dict]:
    """Make a cached request to TMDB API"""
    params = params or {}
    cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
    
    # Check cache
    cached = cache.get(cache_key)
    if cached and time.time() - cached["ts"] < ttl:
        return cached["data"]
    
    url = f"{TMDB_BASE_URL}{endpoint}"
    params = {"api_key": TMDB_API_KEY, **params}
    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 2))
                time.sleep(retry_after)
                response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            cache[cache_key] = {"data": data, "ts": time.time()}
            return data
    except Exception as e:
        logging.error(f"TMDB request failed: {e}")
        return None

def get_image_url(path: Optional[str], size: str = "w500") -> Optional[str]:
    """Get full image URL from TMDB path"""
    return f"{IMAGE_BASE_URL}{size}{path}" if path else None

def process_movie(movie: Dict) -> Dict:
    """Process movie data to include full image URLs"""
    return {
        **movie,
        "poster_url": get_image_url(movie.get("poster_path"), "w500"),
        "backdrop_url": get_image_url(movie.get("backdrop_path"), "w1280"),
        "poster_thumb": get_image_url(movie.get("poster_path"), "w185"),
    }

def process_movies(movies: List[Dict]) -> List[Dict]:
    """Process list of movies"""
    return [process_movie(m) for m in movies]

# TMDB Movie Endpoints
@api_router.get("/movies/popular")
async def get_popular_movies(page: int = Query(1, ge=1, le=500)):
    data = await tmdb_request("/movie/popular", {"page": page})
    if not data:
        raise HTTPException(status_code=500, detail="Failed to fetch popular movies")
    return {"results": process_movies(data.get("results", [])), "page": data.get("page"), "total_pages": data.get("total_pages")}

@api_router.get("/movies/trending")
async def get_trending_movies(time_window: str = "week", page: int = Query(1, ge=1, le=500)):
    data = await tmdb_request(f"/trending/movie/{time_window}", {"page": page})
    if not data:
        raise HTTPException(status_code=500, detail="Failed to fetch trending movies")
    return {"results": process_movies(data.get("results", [])), "page": data.get("page"), "total_pages": data.get("total_pages")}

@api_router.get("/movies/top-rated")
async def get_top_rated_movies(page: int = Query(1, ge=1, le=500)):
    data = await tmdb_request("/movie/top_rated", {"page": page})
    if not data:
        raise HTTPException(status_code=500, detail="Failed to fetch top rated movies")
    return {"results": process_movies(data.get("results", [])), "page": data.get("page"), "total_pages": data.get("total_pages")}

@api_router.get("/movies/upcoming")
async def get_upcoming_movies(page: int = Query(1, ge=1, le=500)):
    data = await tmdb_request("/movie/upcoming", {"page": page})
    if not data:
        raise HTTPException(status_code=500, detail="Failed to fetch upcoming movies")
    return {"results": process_movies(data.get("results", [])), "page": data.get("page"), "total_pages": data.get("total_pages")}

@api_router.get("/movies/now-playing")
async def get_now_playing_movies(page: int = Query(1, ge=1, le=500)):
    data = await tmdb_request("/movie/now_playing", {"page": page})
    if not data:
        raise HTTPException(status_code=500, detail="Failed to fetch now playing movies")
    return {"results": process_movies(data.get("results", [])), "page": data.get("page"), "total_pages": data.get("total_pages")}

@api_router.get("/movies/search")
async def search_movies(query: str, page: int = Query(1, ge=1, le=500)):
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    data = await tmdb_request("/search/movie", {"query": query, "page": page})
    if not data:
        raise HTTPException(status_code=500, detail="Failed to search movies")
    return {"results": process_movies(data.get("results", [])), "page": data.get("page"), "total_pages": data.get("total_pages")}

@api_router.get("/movies/genres")
async def get_genres():
    data = await tmdb_request("/genre/movie/list", ttl=86400)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to fetch genres")
    return {"genres": data.get("genres", [])}

@api_router.get("/movies/discover")
async def discover_movies(
    genre_id: Optional[int] = None,
    year: Optional[int] = None,
    sort_by: str = "popularity.desc",
    min_rating: Optional[float] = None,
    page: int = Query(1, ge=1, le=500)
):
    params = {
        "page": page,
        "sort_by": sort_by,
        "with_genres": genre_id,
        "year": year,
        "vote_average.gte": min_rating
    }
    data = await tmdb_request("/discover/movie", params)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to discover movies")
    return {"results": process_movies(data.get("results", [])), "page": data.get("page"), "total_pages": data.get("total_pages")}

@api_router.get("/movies/{movie_id}")
async def get_movie_details(movie_id: int):
    data = await tmdb_request(f"/movie/{movie_id}", {
        "append_to_response": "credits,videos,images,similar,recommendations"
    })
    if not data:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    # Process the movie data
    result = process_movie(data)
    
    # Get trailer
    videos = data.get("videos", {}).get("results", [])
    trailer = None
    for v in videos:
        if v.get("type") == "Trailer" and v.get("site") == "YouTube":
            trailer = f"https://www.youtube.com/embed/{v['key']}"
            break
    if not trailer:
        for v in videos:
            if v.get("type") == "Teaser" and v.get("site") == "YouTube":
                trailer = f"https://www.youtube.com/embed/{v['key']}"
                break
    
    result["trailer_url"] = trailer
    result["cast"] = data.get("credits", {}).get("cast", [])[:10]
    result["crew"] = data.get("credits", {}).get("crew", [])[:5]
    result["similar"] = process_movies(data.get("similar", {}).get("results", [])[:8])
    result["recommendations"] = process_movies(data.get("recommendations", {}).get("results", [])[:8])
    
    return result

# AI Recommendation Endpoint
@api_router.post("/ai/recommend")
async def get_ai_recommendations(request: AIRecommendRequest):
    from emergentintegrations.llm.chat import LlmChat, UserMessage
    
    # Build the prompt
    prompt_parts = ["Based on the user's preferences, suggest 6 specific movie recommendations. Return ONLY a JSON array of movie titles, nothing else."]
    
    if request.mood:
        prompt_parts.append(f"Mood: {request.mood}")
    if request.genres:
        prompt_parts.append(f"Preferred genres: {', '.join(request.genres)}")
    if request.description:
        prompt_parts.append(f"What they're looking for: {request.description}")
    
    prompt = "\n".join(prompt_parts)
    prompt += '\n\nRespond with ONLY a JSON array like: ["Movie 1", "Movie 2", ...]'
    
    try:
        chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=str(uuid.uuid4()),
            system_message="You are a movie recommendation expert. You suggest movies based on user preferences. Always respond with a JSON array of exactly 6 movie titles."
        ).with_model("openai", "gpt-4o-mini")
        
        response = await chat.send_message(UserMessage(text=prompt))
        
        # Parse the response to get movie titles
        try:
            # Clean the response - remove markdown code blocks if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()
            
            movie_titles = json.loads(clean_response)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract movie names
            movie_titles = [line.strip().strip('"').strip("'") for line in response.split('\n') if line.strip()][:6]
        
        # Search for each movie on TMDB
        movies = []
        for title in movie_titles[:6]:
            search_data = await tmdb_request("/search/movie", {"query": title})
            if search_data and search_data.get("results"):
                movies.append(process_movie(search_data["results"][0]))
        
        return {"recommendations": movies, "titles": movie_titles}
    except Exception as e:
        logging.error(f"AI recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI recommendation failed: {str(e)}")

# Watchlist Endpoints
@api_router.get("/watchlist", response_model=List[WatchlistItem])
async def get_watchlist():
    items = await db.watchlist.find({}, {"_id": 0}).sort("added_at", -1).to_list(100)
    for item in items:
        if isinstance(item.get('added_at'), str):
            item['added_at'] = datetime.fromisoformat(item['added_at'])
    return items

@api_router.post("/watchlist", response_model=WatchlistItem)
async def add_to_watchlist(item: WatchlistCreate):
    # Check if already in watchlist
    existing = await db.watchlist.find_one({"movie_id": item.movie_id}, {"_id": 0})
    if existing:
        raise HTTPException(status_code=400, detail="Movie already in watchlist")
    
    watchlist_item = WatchlistItem(**item.model_dump())
    doc = watchlist_item.model_dump()
    doc['added_at'] = doc['added_at'].isoformat()
    await db.watchlist.insert_one(doc)
    return watchlist_item

@api_router.delete("/watchlist/{movie_id}")
async def remove_from_watchlist(movie_id: int):
    result = await db.watchlist.delete_one({"movie_id": movie_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Movie not in watchlist")
    return {"message": "Removed from watchlist"}

@api_router.get("/watchlist/check/{movie_id}")
async def check_watchlist(movie_id: int):
    item = await db.watchlist.find_one({"movie_id": movie_id}, {"_id": 0})
    return {"in_watchlist": item is not None}

# Rating Endpoints
@api_router.get("/ratings", response_model=List[Rating])
async def get_ratings():
    items = await db.ratings.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    for item in items:
        if isinstance(item.get('created_at'), str):
            item['created_at'] = datetime.fromisoformat(item['created_at'])
    return items

@api_router.post("/ratings", response_model=Rating)
async def add_rating(item: RatingCreate):
    # Update if exists, otherwise create
    existing = await db.ratings.find_one({"movie_id": item.movie_id}, {"_id": 0})
    
    rating_obj = Rating(**item.model_dump())
    doc = rating_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    if existing:
        await db.ratings.update_one({"movie_id": item.movie_id}, {"$set": doc})
    else:
        await db.ratings.insert_one(doc)
    
    return rating_obj

@api_router.delete("/ratings/{movie_id}")
async def delete_rating(movie_id: int):
    result = await db.ratings.delete_one({"movie_id": movie_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Rating not found")
    return {"message": "Rating deleted"}

@api_router.get("/ratings/{movie_id}")
async def get_movie_rating(movie_id: int):
    item = await db.ratings.find_one({"movie_id": movie_id}, {"_id": 0})
    if not item:
        return {"rating": None}
    if isinstance(item.get('created_at'), str):
        item['created_at'] = datetime.fromisoformat(item['created_at'])
    return item

# Health check
@api_router.get("/")
async def root():
    return {"message": "Movie Recommendation API", "status": "running"}

# Include the router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
