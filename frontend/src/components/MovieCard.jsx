import { Link } from 'react-router-dom';
import { Star, Plus, Check } from 'lucide-react';
import { useState } from 'react';
import { addToWatchlist, removeFromWatchlist } from '../api';
import { toast } from 'sonner';

const MovieCard = ({ movie, isInWatchlist = false, onWatchlistChange, showRank = false, rank = 0 }) => {
  const [inWatchlist, setInWatchlist] = useState(isInWatchlist);
  const [isLoading, setIsLoading] = useState(false);

  const handleWatchlistClick = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsLoading(true);

    try {
      if (inWatchlist) {
        await removeFromWatchlist(movie.id);
        setInWatchlist(false);
        toast.success('Removed from watchlist');
      } else {
        await addToWatchlist({
          movie_id: movie.id,
          title: movie.title,
          poster_path: movie.poster_path,
          release_date: movie.release_date,
          vote_average: movie.vote_average
        });
        setInWatchlist(true);
        toast.success('Added to watchlist');
      }
      onWatchlistChange?.();
    } catch (error) {
      if (error.response?.status === 400) {
        setInWatchlist(true);
        toast.info('Already in watchlist');
      } else {
        toast.error('Something went wrong');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const posterUrl = movie.poster_url || movie.poster_path 
    ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
    : null;

  return (
    <Link 
      to={`/movie/${movie.id}`}
      data-testid={`movie-card-${movie.id}`}
      className="movie-card relative overflow-hidden rounded-xl aspect-[2/3] group cursor-pointer block"
    >
      {/* Rank Badge */}
      {showRank && (
        <div className="absolute top-2 left-2 z-20">
          <span className="font-accent text-5xl text-white/90 drop-shadow-lg" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.8)' }}>
            {rank}
          </span>
        </div>
      )}

      {/* Poster Image */}
      {posterUrl ? (
        <img
          src={posterUrl}
          alt={movie.title}
          className="w-full h-full object-cover transition-transform duration-500"
          loading="lazy"
        />
      ) : (
        <div className="w-full h-full bg-obsidian-lighter flex items-center justify-center">
          <span className="text-muted-foreground text-sm">No Image</span>
        </div>
      )}

      {/* Overlay */}
      <div className="movie-overlay absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent opacity-0 transition-opacity duration-300 flex flex-col justify-end p-4">
        <h3 className="font-heading text-lg font-semibold line-clamp-2 mb-1">{movie.title}</h3>
        
        <div className="flex items-center gap-3 text-sm">
          {movie.vote_average > 0 && (
            <div className="flex items-center gap-1">
              <Star className="w-4 h-4 text-gold fill-gold" />
              <span className="text-gold font-medium">{movie.vote_average?.toFixed(1)}</span>
            </div>
          )}
          {movie.release_date && (
            <span className="text-white/60">{movie.release_date?.slice(0, 4)}</span>
          )}
        </div>
      </div>

      {/* Watchlist Button */}
      <button
        data-testid={`watchlist-btn-${movie.id}`}
        onClick={handleWatchlistClick}
        disabled={isLoading}
        className={`absolute top-3 right-3 z-20 w-9 h-9 rounded-full flex items-center justify-center transition-all duration-300 ${
          inWatchlist 
            ? 'bg-electric text-white' 
            : 'bg-black/60 backdrop-blur-sm text-white/80 hover:bg-electric hover:text-white'
        }`}
      >
        {isLoading ? (
          <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        ) : inWatchlist ? (
          <Check className="w-4 h-4" strokeWidth={2} />
        ) : (
          <Plus className="w-4 h-4" strokeWidth={2} />
        )}
      </button>

      {/* Border glow on hover */}
      <div className="absolute inset-0 rounded-xl border border-white/0 group-hover:border-white/20 transition-colors duration-300 pointer-events-none" />
    </Link>
  );
};

export default MovieCard;
