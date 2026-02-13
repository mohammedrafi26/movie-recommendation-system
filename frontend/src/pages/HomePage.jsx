import { useState, useEffect } from 'react';
import HeroSection from '../components/HeroSection';
import MovieRow from '../components/MovieRow';
import { getTrendingMovies, getPopularMovies, getTopRatedMovies, getUpcomingMovies, getNowPlayingMovies, getMovieDetails } from '../api';
import { Skeleton } from '../components/ui/skeleton';

const HomePage = () => {
  const [heroMovie, setHeroMovie] = useState(null);
  const [trending, setTrending] = useState([]);
  const [popular, setPopular] = useState([]);
  const [topRated, setTopRated] = useState([]);
  const [upcoming, setUpcoming] = useState([]);
  const [nowPlaying, setNowPlaying] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [trendingRes, popularRes, topRatedRes, upcomingRes, nowPlayingRes] = await Promise.all([
          getTrendingMovies(),
          getPopularMovies(),
          getTopRatedMovies(),
          getUpcomingMovies(),
          getNowPlayingMovies()
        ]);

        setTrending(trendingRes.data.results || []);
        setPopular(popularRes.data.results || []);
        setTopRated(topRatedRes.data.results || []);
        setUpcoming(upcomingRes.data.results || []);
        setNowPlaying(nowPlayingRes.data.results || []);

        // Get detailed info for hero movie
        if (trendingRes.data.results?.[0]) {
          const heroDetails = await getMovieDetails(trendingRes.data.results[0].id);
          setHeroMovie(heroDetails.data);
        }
      } catch (error) {
        console.error('Failed to fetch movies:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div data-testid="home-loading" className="min-h-screen bg-obsidian">
        {/* Hero Skeleton */}
        <div className="h-[85vh] relative">
          <Skeleton className="absolute inset-0 bg-obsidian-light" />
          <div className="absolute bottom-24 left-6 space-y-4">
            <Skeleton className="h-6 w-32 bg-obsidian-lighter" />
            <Skeleton className="h-16 w-96 bg-obsidian-lighter" />
            <Skeleton className="h-4 w-64 bg-obsidian-lighter" />
            <Skeleton className="h-12 w-48 bg-obsidian-lighter rounded-full" />
          </div>
        </div>
        
        {/* Row Skeletons */}
        {[1, 2, 3].map((i) => (
          <div key={i} className="py-8 max-w-7xl mx-auto px-6">
            <Skeleton className="h-8 w-48 mb-6 bg-obsidian-lighter" />
            <div className="flex gap-4">
              {[1, 2, 3, 4, 5, 6].map((j) => (
                <Skeleton key={j} className="flex-shrink-0 w-[200px] aspect-[2/3] rounded-xl bg-obsidian-lighter" />
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div data-testid="home-page" className="min-h-screen bg-obsidian">
      {/* Hero Section */}
      <HeroSection movie={heroMovie} />

      {/* Movie Rows */}
      <div className="-mt-16 relative z-10">
        <MovieRow 
          title="Trending This Week" 
          movies={trending.slice(0, 10)} 
          viewAllLink="/browse?category=trending"
          showRank={true}
        />
        
        <MovieRow 
          title="Now Playing" 
          movies={nowPlaying.slice(0, 10)} 
          viewAllLink="/browse?category=now-playing"
        />
        
        <MovieRow 
          title="Popular Movies" 
          movies={popular.slice(0, 10)} 
          viewAllLink="/browse?category=popular"
        />
        
        <MovieRow 
          title="Top Rated" 
          movies={topRated.slice(0, 10)} 
          viewAllLink="/browse?category=top-rated"
        />
        
        <MovieRow 
          title="Coming Soon" 
          movies={upcoming.slice(0, 10)} 
          viewAllLink="/browse?category=upcoming"
        />
      </div>
    </div>
  );
};

export default HomePage;
