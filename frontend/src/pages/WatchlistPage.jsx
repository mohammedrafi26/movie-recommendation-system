import { useState, useEffect } from 'react';
import { Bookmark, Trash2, Film } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { Skeleton } from '../components/ui/skeleton';
import MovieCard from '../components/MovieCard';
import { getWatchlist, removeFromWatchlist } from '../api';
import { toast } from 'sonner';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '../components/ui/alert-dialog';

const WatchlistPage = () => {
  const [watchlist, setWatchlist] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deleteId, setDeleteId] = useState(null);

  const fetchWatchlist = async () => {
    try {
      const res = await getWatchlist();
      setWatchlist(res.data || []);
    } catch (error) {
      console.error('Failed to fetch watchlist:', error);
      toast.error('Failed to load watchlist');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWatchlist();
  }, []);

  const handleRemove = async () => {
    if (!deleteId) return;
    
    try {
      await removeFromWatchlist(deleteId);
      setWatchlist(prev => prev.filter(item => item.movie_id !== deleteId));
      toast.success('Removed from watchlist');
    } catch (error) {
      toast.error('Failed to remove from watchlist');
    } finally {
      setDeleteId(null);
    }
  };

  // Convert watchlist items to movie format for MovieCard
  const movies = watchlist.map(item => ({
    id: item.movie_id,
    title: item.title,
    poster_path: item.poster_path,
    release_date: item.release_date,
    vote_average: item.vote_average
  }));

  return (
    <div data-testid="watchlist-page" className="min-h-screen bg-obsidian pt-24 pb-16">
      <div className="max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <div className="w-12 h-12 rounded-xl bg-electric/20 flex items-center justify-center">
            <Bookmark className="w-6 h-6 text-electric" />
          </div>
          <div>
            <h1 className="font-heading text-3xl md:text-4xl font-bold tracking-tight">
              My Watchlist
            </h1>
            <p className="text-muted-foreground">
              {watchlist.length} {watchlist.length === 1 ? 'movie' : 'movies'} saved
            </p>
          </div>
        </div>

        {/* Content */}
        {loading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            {[...Array(6)].map((_, i) => (
              <Skeleton key={i} className="aspect-[2/3] rounded-xl bg-obsidian-lighter" />
            ))}
          </div>
        ) : movies.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            {movies.map((movie, index) => (
              <div 
                key={movie.id} 
                className="relative group"
              >
                <MovieCard 
                  movie={movie} 
                  isInWatchlist={true}
                  onWatchlistChange={fetchWatchlist}
                />
                {/* Remove Button */}
                <button
                  data-testid={`remove-watchlist-${movie.id}`}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setDeleteId(movie.id);
                  }}
                  className="absolute bottom-3 right-3 z-20 w-9 h-9 rounded-full bg-red-500/80 backdrop-blur-sm flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500"
                >
                  <Trash2 className="w-4 h-4 text-white" />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <div className="w-20 h-20 rounded-2xl bg-white/5 flex items-center justify-center mb-6">
              <Film className="w-10 h-10 text-muted-foreground" />
            </div>
            <h2 className="font-heading text-2xl font-semibold mb-2">Your watchlist is empty</h2>
            <p className="text-muted-foreground mb-6 max-w-md">
              Start adding movies you want to watch later. They'll all appear here.
            </p>
            <Link to="/browse">
              <Button className="rounded-full bg-electric hover:bg-electric/80">
                Browse Movies
              </Button>
            </Link>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteId} onOpenChange={() => setDeleteId(null)}>
        <AlertDialogContent className="bg-obsidian-light border-white/10">
          <AlertDialogHeader>
            <AlertDialogTitle className="font-heading">Remove from Watchlist?</AlertDialogTitle>
            <AlertDialogDescription>
              This movie will be removed from your watchlist. You can always add it back later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="rounded-full">Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleRemove}
              className="rounded-full bg-red-500 hover:bg-red-600"
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default WatchlistPage;
