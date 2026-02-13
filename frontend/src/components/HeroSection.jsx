import { Play, Info } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from './ui/button';

const HeroSection = ({ movie }) => {
  if (!movie) return null;

  const backdropUrl = movie.backdrop_url || (movie.backdrop_path 
    ? `https://image.tmdb.org/t/p/w1280${movie.backdrop_path}`
    : null);

  return (
    <section 
      data-testid="hero-section"
      className="relative h-[85vh] w-full flex items-end"
    >
      {/* Background Image */}
      {backdropUrl && (
        <div className="absolute inset-0">
          <img
            src={backdropUrl}
            alt={movie.title}
            className="w-full h-full object-cover"
          />
          {/* Gradient overlays */}
          <div className="absolute inset-0 bg-gradient-to-t from-obsidian via-obsidian/60 to-transparent" />
          <div className="absolute inset-0 bg-gradient-to-r from-obsidian/80 via-transparent to-transparent" />
        </div>
      )}

      {/* Content */}
      <div className="relative z-10 max-w-7xl mx-auto w-full px-6 pb-24">
        <div className="max-w-2xl animate-fade-in-up">
          {/* Label */}
          <span className="inline-block px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest bg-electric/20 text-electric border border-electric/30 mb-6">
            Featured Film
          </span>

          {/* Title */}
          <h1 className="font-heading text-5xl md:text-7xl font-bold tracking-tighter leading-none mb-4">
            {movie.title}
          </h1>

          {/* Meta */}
          <div className="flex items-center gap-4 text-sm text-white/70 mb-6">
            {movie.vote_average > 0 && (
              <span className="flex items-center gap-1 text-gold">
                ★ {movie.vote_average?.toFixed(1)}
              </span>
            )}
            {movie.release_date && (
              <span>{movie.release_date?.slice(0, 4)}</span>
            )}
            {movie.runtime && (
              <span>{Math.floor(movie.runtime / 60)}h {movie.runtime % 60}m</span>
            )}
          </div>

          {/* Overview */}
          <p className="text-base md:text-lg text-white/70 leading-relaxed mb-8 line-clamp-3">
            {movie.overview}
          </p>

          {/* Actions */}
          <div className="flex items-center gap-4">
            <Link to={`/movie/${movie.id}`}>
              <Button 
                data-testid="hero-details-btn"
                className="h-12 px-8 rounded-full bg-white text-black hover:bg-gray-200 font-medium transition-all duration-300 active:scale-95"
              >
                <Info className="w-5 h-5 mr-2" />
                View Details
              </Button>
            </Link>
            {movie.trailer_url && (
              <a href={movie.trailer_url} target="_blank" rel="noopener noreferrer">
                <Button 
                  data-testid="hero-trailer-btn"
                  variant="outline"
                  className="h-12 px-8 rounded-full bg-transparent border-white/20 hover:bg-white/10 font-medium transition-all duration-300"
                >
                  <Play className="w-5 h-5 mr-2" />
                  Watch Trailer
                </Button>
              </a>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
