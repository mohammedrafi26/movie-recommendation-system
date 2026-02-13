import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Search, Filter, X, ChevronDown } from 'lucide-react';
import MovieCard from '../components/MovieCard';
import { Input } from '../components/ui/input';
import { Button } from '../components/ui/button';
import { Skeleton } from '../components/ui/skeleton';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../components/ui/dropdown-menu';
import {
  searchMovies,
  getPopularMovies,
  getTrendingMovies,
  getTopRatedMovies,
  getUpcomingMovies,
  getNowPlayingMovies,
  getGenres,
  discoverMovies
} from '../api';

const CATEGORIES = [
  { id: 'popular', label: 'Popular', fetch: getPopularMovies },
  { id: 'trending', label: 'Trending', fetch: () => getTrendingMovies() },
  { id: 'top-rated', label: 'Top Rated', fetch: getTopRatedMovies },
  { id: 'upcoming', label: 'Upcoming', fetch: getUpcomingMovies },
  { id: 'now-playing', label: 'Now Playing', fetch: getNowPlayingMovies },
];

const SORT_OPTIONS = [
  { id: 'popularity.desc', label: 'Most Popular' },
  { id: 'vote_average.desc', label: 'Highest Rated' },
  { id: 'release_date.desc', label: 'Newest First' },
  { id: 'release_date.asc', label: 'Oldest First' },
];

const BrowsePage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [searchQuery, setSearchQuery] = useState('');
  const [movies, setMovies] = useState([]);
  const [genres, setGenres] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loadingMore, setLoadingMore] = useState(false);

  const category = searchParams.get('category') || 'popular';
  const genreId = searchParams.get('genre');
  const sortBy = searchParams.get('sort') || 'popularity.desc';

  // Fetch genres on mount
  useEffect(() => {
    const fetchGenres = async () => {
      try {
        const res = await getGenres();
        setGenres(res.data.genres || []);
      } catch (error) {
        console.error('Failed to fetch genres:', error);
      }
    };
    fetchGenres();
  }, []);

  // Fetch movies based on filters
  const fetchMovies = useCallback(async (pageNum = 1, append = false) => {
    if (!append) setLoading(true);
    else setLoadingMore(true);

    try {
      let res;

      if (searchQuery) {
        res = await searchMovies(searchQuery, pageNum);
      } else if (genreId) {
        res = await discoverMovies({ genre_id: genreId, sort_by: sortBy, page: pageNum });
      } else {
        const categoryConfig = CATEGORIES.find(c => c.id === category);
        if (categoryConfig) {
          res = await categoryConfig.fetch(pageNum);
        } else {
          res = await getPopularMovies(pageNum);
        }
      }

      const newMovies = res.data.results || [];
      setMovies(prev => append ? [...prev, ...newMovies] : newMovies);
      setTotalPages(res.data.total_pages || 1);
      setPage(pageNum);
    } catch (error) {
      console.error('Failed to fetch movies:', error);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [category, genreId, sortBy, searchQuery]);

  // Re-fetch when filters change
  useEffect(() => {
    setPage(1);
    fetchMovies(1, false);
  }, [fetchMovies]);

  // Handle search
  const handleSearch = (e) => {
    e.preventDefault();
    setPage(1);
    fetchMovies(1, false);
  };

  // Handle category change
  const handleCategoryChange = (newCategory) => {
    setSearchQuery('');
    setSearchParams({ category: newCategory });
  };

  // Handle genre change
  const handleGenreChange = (newGenreId) => {
    setSearchQuery('');
    if (newGenreId) {
      setSearchParams({ genre: newGenreId, sort: sortBy });
    } else {
      setSearchParams({ category: 'popular' });
    }
  };

  // Handle sort change
  const handleSortChange = (newSort) => {
    if (genreId) {
      setSearchParams({ genre: genreId, sort: newSort });
    }
  };

  // Load more movies
  const loadMore = () => {
    if (page < totalPages && !loadingMore) {
      fetchMovies(page + 1, true);
    }
  };

  // Clear all filters
  const clearFilters = () => {
    setSearchQuery('');
    setSearchParams({ category: 'popular' });
  };

  const selectedGenre = genres.find(g => g.id === parseInt(genreId));
  const hasFilters = searchQuery || genreId;

  return (
    <div data-testid="browse-page" className="min-h-screen bg-obsidian pt-24 pb-16">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-heading text-4xl md:text-5xl font-bold tracking-tight mb-2">
            Browse Movies
          </h1>
          <p className="text-muted-foreground">
            Explore thousands of movies from every genre
          </p>
        </div>

        {/* Search & Filters */}
        <div className="flex flex-col md:flex-row gap-4 mb-8">
          {/* Search */}
          <form onSubmit={handleSearch} className="flex-1 flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <Input
                data-testid="search-input"
                type="text"
                placeholder="Search for movies..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-12 h-12 bg-obsidian-light border-white/10 focus:border-electric rounded-full"
              />
            </div>
            <Button 
              type="submit" 
              data-testid="search-btn"
              className="h-12 px-6 rounded-full bg-electric hover:bg-electric/80"
            >
              Search
            </Button>
          </form>

          {/* Genre Filter */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button 
                data-testid="genre-filter-btn"
                variant="outline" 
                className="h-12 px-6 rounded-full bg-transparent border-white/10 hover:bg-white/5"
              >
                <Filter className="w-4 h-4 mr-2" />
                {selectedGenre ? selectedGenre.name : 'Genre'}
                <ChevronDown className="w-4 h-4 ml-2" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent 
              align="end" 
              className="max-h-80 overflow-y-auto bg-obsidian-light border-white/10"
            >
              <DropdownMenuItem 
                onClick={() => handleGenreChange(null)}
                className="cursor-pointer hover:bg-white/5"
              >
                All Genres
              </DropdownMenuItem>
              {genres.map(genre => (
                <DropdownMenuItem 
                  key={genre.id}
                  data-testid={`genre-${genre.id}`}
                  onClick={() => handleGenreChange(genre.id.toString())}
                  className="cursor-pointer hover:bg-white/5"
                >
                  {genre.name}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Sort (only when genre is selected) */}
          {genreId && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button 
                  data-testid="sort-btn"
                  variant="outline" 
                  className="h-12 px-6 rounded-full bg-transparent border-white/10 hover:bg-white/5"
                >
                  {SORT_OPTIONS.find(s => s.id === sortBy)?.label || 'Sort'}
                  <ChevronDown className="w-4 h-4 ml-2" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="bg-obsidian-light border-white/10">
                {SORT_OPTIONS.map(option => (
                  <DropdownMenuItem 
                    key={option.id}
                    onClick={() => handleSortChange(option.id)}
                    className="cursor-pointer hover:bg-white/5"
                  >
                    {option.label}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>

        {/* Category Tabs (only when no search or genre) */}
        {!hasFilters && (
          <div className="flex gap-2 mb-8 overflow-x-auto scrollbar-hide pb-2">
            {CATEGORIES.map(cat => (
              <button
                key={cat.id}
                data-testid={`category-${cat.id}`}
                onClick={() => handleCategoryChange(cat.id)}
                className={`px-5 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-all ${
                  category === cat.id
                    ? 'bg-white text-black'
                    : 'bg-white/5 text-white/70 hover:bg-white/10 hover:text-white'
                }`}
              >
                {cat.label}
              </button>
            ))}
          </div>
        )}

        {/* Active Filters */}
        {hasFilters && (
          <div className="flex items-center gap-2 mb-6">
            <span className="text-sm text-muted-foreground">Filters:</span>
            {searchQuery && (
              <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm bg-white/10">
                Search: "{searchQuery}"
                <button onClick={() => setSearchQuery('')}>
                  <X className="w-3 h-3 ml-1" />
                </button>
              </span>
            )}
            {selectedGenre && (
              <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm bg-white/10">
                {selectedGenre.name}
                <button onClick={() => handleGenreChange(null)}>
                  <X className="w-3 h-3 ml-1" />
                </button>
              </span>
            )}
            <button 
              onClick={clearFilters}
              className="text-sm text-electric hover:underline ml-2"
            >
              Clear all
            </button>
          </div>
        )}

        {/* Results */}
        {loading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            {[...Array(18)].map((_, i) => (
              <Skeleton key={i} className="aspect-[2/3] rounded-xl bg-obsidian-lighter" />
            ))}
          </div>
        ) : movies.length > 0 ? (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
              {movies.map(movie => (
                <MovieCard key={movie.id} movie={movie} />
              ))}
            </div>

            {/* Load More */}
            {page < totalPages && (
              <div className="flex justify-center mt-12">
                <Button
                  data-testid="load-more-btn"
                  onClick={loadMore}
                  disabled={loadingMore}
                  className="h-12 px-8 rounded-full bg-white/10 hover:bg-white/20"
                >
                  {loadingMore ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                      Loading...
                    </>
                  ) : (
                    'Load More'
                  )}
                </Button>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <p className="text-xl text-muted-foreground mb-4">No movies found</p>
            <Button 
              onClick={clearFilters}
              className="rounded-full"
            >
              Clear Filters
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default BrowsePage;
