import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Play, Plus, Check, Star, Calendar, Clock, ExternalLink } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Skeleton } from '../components/ui/skeleton';
import { Textarea } from '../components/ui/textarea';
import MovieCard from '../components/MovieCard';
import StarRating from '../components/StarRating';
import { getMovieDetails, checkWatchlist, addToWatchlist, removeFromWatchlist, getMovieRating, addRating } from '../api';
import { toast } from 'sonner';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '../components/ui/dialog';

const MovieDetailPage = () => {
  const { id } = useParams();
  const [movie, setMovie] = useState(null);
  const [loading, setLoading] = useState(true);
  const [inWatchlist, setInWatchlist] = useState(false);
  const [watchlistLoading, setWatchlistLoading] = useState(false);
  const [userRating, setUserRating] = useState(null);
  const [ratingDialogOpen, setRatingDialogOpen] = useState(false);
  const [reviewText, setReviewText] = useState('');
  const [ratingValue, setRatingValue] = useState(0);
  const [trailerOpen, setTrailerOpen] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [movieRes, watchlistRes, ratingRes] = await Promise.all([
          getMovieDetails(id),
          checkWatchlist(id),
          getMovieRating(id)
        ]);

        setMovie(movieRes.data);
        setInWatchlist(watchlistRes.data.in_watchlist);
        
        if (ratingRes.data.rating) {
          setUserRating(ratingRes.data);
          setRatingValue(ratingRes.data.rating);
          setReviewText(ratingRes.data.review || '');
        }
      } catch (error) {
        console.error('Failed to fetch movie:', error);
        toast.error('Failed to load movie details');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    window.scrollTo(0, 0);
  }, [id]);

  const handleWatchlist = async () => {
    setWatchlistLoading(true);
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
    } catch (error) {
      toast.error('Something went wrong');
    } finally {
      setWatchlistLoading(false);
    }
  };

  const handleSaveRating = async () => {
    if (ratingValue === 0) {
      toast.error('Please select a rating');
      return;
    }

    try {
      await addRating({
        movie_id: movie.id,
        title: movie.title,
        poster_path: movie.poster_path,
        rating: ratingValue,
        review: reviewText || null
      });
      setUserRating({ rating: ratingValue, review: reviewText });
      setRatingDialogOpen(false);
      toast.success('Rating saved!');
    } catch (error) {
      toast.error('Failed to save rating');
    }
  };

  if (loading) {
    return (
      <div data-testid="movie-detail-loading" className="min-h-screen bg-obsidian">
        <div className="h-[70vh] relative">
          <Skeleton className="absolute inset-0 bg-obsidian-light" />
        </div>
        <div className="max-w-7xl mx-auto px-6 -mt-32 relative z-10">
          <div className="flex flex-col md:flex-row gap-8">
            <Skeleton className="w-64 aspect-[2/3] rounded-xl bg-obsidian-lighter flex-shrink-0" />
            <div className="flex-1 space-y-4">
              <Skeleton className="h-12 w-3/4 bg-obsidian-lighter" />
              <Skeleton className="h-6 w-1/2 bg-obsidian-lighter" />
              <Skeleton className="h-32 w-full bg-obsidian-lighter" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="min-h-screen bg-obsidian flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-heading mb-4">Movie not found</h1>
          <Link to="/">
            <Button className="rounded-full">Go Home</Button>
          </Link>
        </div>
      </div>
    );
  }

  const backdropUrl = movie.backdrop_url || (movie.backdrop_path 
    ? `https://image.tmdb.org/t/p/w1280${movie.backdrop_path}`
    : null);
  
  const posterUrl = movie.poster_url || (movie.poster_path 
    ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
    : null);

  return (
    <div data-testid="movie-detail-page" className="min-h-screen bg-obsidian">
      {/* Backdrop */}
      <div className="h-[70vh] relative">
        {backdropUrl && (
          <>
            <img
              src={backdropUrl}
              alt={movie.title}
              className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-obsidian via-obsidian/60 to-transparent" />
            <div className="absolute inset-0 bg-gradient-to-r from-obsidian/80 via-transparent to-transparent" />
          </>
        )}

        {/* Back Button */}
        <Link 
          to="/"
          data-testid="back-btn"
          className="absolute top-24 left-6 z-20 flex items-center gap-2 px-4 py-2 rounded-full bg-black/40 backdrop-blur-sm hover:bg-black/60 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="text-sm">Back</span>
        </Link>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 -mt-64 relative z-10 pb-16">
        <div className="flex flex-col md:flex-row gap-8">
          {/* Poster */}
          <div className="flex-shrink-0">
            {posterUrl ? (
              <img
                src={posterUrl}
                alt={movie.title}
                className="w-64 rounded-xl shadow-2xl"
              />
            ) : (
              <div className="w-64 aspect-[2/3] rounded-xl bg-obsidian-lighter flex items-center justify-center">
                <span className="text-muted-foreground">No Image</span>
              </div>
            )}
          </div>

          {/* Info */}
          <div className="flex-1">
            {/* Title */}
            <h1 className="font-heading text-4xl md:text-5xl font-bold tracking-tight mb-4">
              {movie.title}
            </h1>

            {/* Meta */}
            <div className="flex flex-wrap items-center gap-4 text-sm text-white/70 mb-6">
              {movie.vote_average > 0 && (
                <div className="flex items-center gap-1">
                  <Star className="w-5 h-5 text-gold fill-gold" />
                  <span className="text-gold font-semibold text-lg">{movie.vote_average?.toFixed(1)}</span>
                  <span className="text-white/50">/ 10</span>
                </div>
              )}
              {movie.release_date && (
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  <span>{new Date(movie.release_date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</span>
                </div>
              )}
              {movie.runtime && (
                <div className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  <span>{Math.floor(movie.runtime / 60)}h {movie.runtime % 60}m</span>
                </div>
              )}
            </div>

            {/* Genres */}
            {movie.genres && movie.genres.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-6">
                {movie.genres.map(genre => (
                  <Link
                    key={genre.id}
                    to={`/browse?genre=${genre.id}`}
                    className="px-3 py-1 rounded-full text-xs font-medium bg-white/10 hover:bg-white/20 transition-colors"
                  >
                    {genre.name}
                  </Link>
                ))}
              </div>
            )}

            {/* Overview */}
            <p className="text-base md:text-lg text-white/80 leading-relaxed mb-8">
              {movie.overview}
            </p>

            {/* User Rating */}
            {userRating && (
              <div className="mb-6 p-4 rounded-xl bg-white/5 border border-white/10">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Your Rating</p>
                    <div className="flex items-center gap-2">
                      <StarRating rating={userRating.rating} readonly size="sm" />
                      <span className="text-gold font-semibold">{userRating.rating}/5</span>
                    </div>
                    {userRating.review && (
                      <p className="text-sm text-white/70 mt-2">"{userRating.review}"</p>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setRatingDialogOpen(true)}
                    className="text-electric hover:text-electric/80"
                  >
                    Edit
                  </Button>
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex flex-wrap gap-4">
              {movie.trailer_url && (
                <Button
                  data-testid="play-trailer-btn"
                  onClick={() => setTrailerOpen(true)}
                  className="h-12 px-8 rounded-full bg-white text-black hover:bg-gray-200 font-medium"
                >
                  <Play className="w-5 h-5 mr-2 fill-black" />
                  Play Trailer
                </Button>
              )}
              <Button
                data-testid="watchlist-btn"
                onClick={handleWatchlist}
                disabled={watchlistLoading}
                variant="outline"
                className={`h-12 px-8 rounded-full font-medium ${
                  inWatchlist 
                    ? 'bg-electric border-electric hover:bg-electric/80' 
                    : 'bg-transparent border-white/20 hover:bg-white/10'
                }`}
              >
                {watchlistLoading ? (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                ) : inWatchlist ? (
                  <Check className="w-5 h-5 mr-2" />
                ) : (
                  <Plus className="w-5 h-5 mr-2" />
                )}
                {inWatchlist ? 'In Watchlist' : 'Add to Watchlist'}
              </Button>
              <Button
                data-testid="rate-btn"
                onClick={() => setRatingDialogOpen(true)}
                variant="outline"
                className="h-12 px-8 rounded-full bg-transparent border-white/20 hover:bg-white/10 font-medium"
              >
                <Star className="w-5 h-5 mr-2" />
                {userRating ? 'Update Rating' : 'Rate Movie'}
              </Button>
            </div>
          </div>
        </div>

        {/* Cast */}
        {movie.cast && movie.cast.length > 0 && (
          <section className="mt-16">
            <h2 className="font-heading text-2xl font-semibold mb-6">Cast</h2>
            <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-4">
              {movie.cast.map(person => (
                <div key={person.id} className="flex-shrink-0 w-32 text-center">
                  {person.profile_path ? (
                    <img
                      src={`https://image.tmdb.org/t/p/w185${person.profile_path}`}
                      alt={person.name}
                      className="w-24 h-24 rounded-full object-cover mx-auto mb-2"
                    />
                  ) : (
                    <div className="w-24 h-24 rounded-full bg-obsidian-lighter mx-auto mb-2 flex items-center justify-center">
                      <span className="text-2xl text-muted-foreground">{person.name[0]}</span>
                    </div>
                  )}
                  <p className="text-sm font-medium line-clamp-1">{person.name}</p>
                  <p className="text-xs text-muted-foreground line-clamp-1">{person.character}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Similar Movies */}
        {movie.similar && movie.similar.length > 0 && (
          <section className="mt-16">
            <h2 className="font-heading text-2xl font-semibold mb-6">Similar Movies</h2>
            <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-4">
              {movie.similar.map(m => (
                <div key={m.id} className="flex-shrink-0 w-[160px] md:w-[200px]">
                  <MovieCard movie={m} />
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Recommendations */}
        {movie.recommendations && movie.recommendations.length > 0 && (
          <section className="mt-16">
            <h2 className="font-heading text-2xl font-semibold mb-6">You Might Also Like</h2>
            <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-4">
              {movie.recommendations.map(m => (
                <div key={m.id} className="flex-shrink-0 w-[160px] md:w-[200px]">
                  <MovieCard movie={m} />
                </div>
              ))}
            </div>
          </section>
        )}

        {/* External Links */}
        {movie.homepage && (
          <div className="mt-16">
            <a
              href={movie.homepage}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-electric hover:underline"
            >
              <ExternalLink className="w-4 h-4" />
              Visit Official Website
            </a>
          </div>
        )}
      </div>

      {/* Trailer Dialog */}
      <Dialog open={trailerOpen} onOpenChange={setTrailerOpen}>
        <DialogContent className="max-w-4xl p-0 bg-black border-white/10 overflow-hidden">
          <div className="aspect-video">
            {movie.trailer_url && (
              <iframe
                src={`${movie.trailer_url}?autoplay=1`}
                title={`${movie.title} Trailer`}
                className="w-full h-full"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Rating Dialog */}
      <Dialog open={ratingDialogOpen} onOpenChange={setRatingDialogOpen}>
        <DialogContent className="bg-obsidian-light border-white/10">
          <DialogHeader>
            <DialogTitle className="font-heading text-xl">Rate {movie.title}</DialogTitle>
          </DialogHeader>
          <div className="py-4 space-y-6">
            <div className="flex flex-col items-center gap-4">
              <StarRating 
                rating={ratingValue} 
                onRate={setRatingValue} 
                size="lg" 
              />
              <span className="text-2xl font-semibold text-gold">
                {ratingValue > 0 ? `${ratingValue}/5` : 'Select rating'}
              </span>
            </div>
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">
                Review (optional)
              </label>
              <Textarea
                data-testid="review-input"
                value={reviewText}
                onChange={(e) => setReviewText(e.target.value)}
                placeholder="Share your thoughts about this movie..."
                className="bg-obsidian border-white/10 focus:border-electric resize-none"
                rows={4}
              />
            </div>
            <div className="flex justify-end gap-3">
              <Button
                variant="ghost"
                onClick={() => setRatingDialogOpen(false)}
                className="rounded-full"
              >
                Cancel
              </Button>
              <Button
                data-testid="save-rating-btn"
                onClick={handleSaveRating}
                className="rounded-full bg-electric hover:bg-electric/80"
              >
                Save Rating
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default MovieDetailPage;
