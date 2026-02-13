import MovieCard from './MovieCard';
import { ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const MovieRow = ({ title, movies, viewAllLink, showRank = false }) => {
  if (!movies || movies.length === 0) return null;

  return (
    <section data-testid={`movie-row-${title?.toLowerCase().replace(/\s+/g, '-')}`} className="py-8">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="font-heading text-2xl md:text-3xl font-semibold tracking-tight">{title}</h2>
          {viewAllLink && (
            <Link 
              to={viewAllLink}
              data-testid={`view-all-${title?.toLowerCase().replace(/\s+/g, '-')}`}
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-white transition-colors group"
            >
              View All
              <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
          )}
        </div>

        {/* Carousel */}
        <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-4 snap-x snap-mandatory">
          {movies.map((movie, index) => (
            <div 
              key={movie.id} 
              className="flex-shrink-0 w-[160px] md:w-[200px] snap-start"
            >
              <MovieCard 
                movie={movie} 
                showRank={showRank}
                rank={index + 1}
              />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default MovieRow;
