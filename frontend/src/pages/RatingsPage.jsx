import { useState, useEffect } from 'react';
import { Star, Trash2, Film, Edit2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { Skeleton } from '../components/ui/skeleton';
import { Textarea } from '../components/ui/textarea';
import StarRating from '../components/StarRating';
import { getRatings, deleteRating, addRating } from '../api';
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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '../components/ui/dialog';

const RatingsPage = () => {
  const [ratings, setRatings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deleteId, setDeleteId] = useState(null);
  const [editRating, setEditRating] = useState(null);
  const [editValue, setEditValue] = useState(0);
  const [editReview, setEditReview] = useState('');

  const fetchRatings = async () => {
    try {
      const res = await getRatings();
      setRatings(res.data || []);
    } catch (error) {
      console.error('Failed to fetch ratings:', error);
      toast.error('Failed to load ratings');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRatings();
  }, []);

  const handleDelete = async () => {
    if (!deleteId) return;
    
    try {
      await deleteRating(deleteId);
      setRatings(prev => prev.filter(item => item.movie_id !== deleteId));
      toast.success('Rating deleted');
    } catch (error) {
      toast.error('Failed to delete rating');
    } finally {
      setDeleteId(null);
    }
  };

  const openEditDialog = (rating) => {
    setEditRating(rating);
    setEditValue(rating.rating);
    setEditReview(rating.review || '');
  };

  const handleSaveEdit = async () => {
    if (editValue === 0) {
      toast.error('Please select a rating');
      return;
    }

    try {
      await addRating({
        movie_id: editRating.movie_id,
        title: editRating.title,
        poster_path: editRating.poster_path,
        rating: editValue,
        review: editReview || null
      });
      
      setRatings(prev => prev.map(r => 
        r.movie_id === editRating.movie_id 
          ? { ...r, rating: editValue, review: editReview }
          : r
      ));
      
      setEditRating(null);
      toast.success('Rating updated!');
    } catch (error) {
      toast.error('Failed to update rating');
    }
  };

  const averageRating = ratings.length > 0 
    ? (ratings.reduce((acc, r) => acc + r.rating, 0) / ratings.length).toFixed(1)
    : 0;

  return (
    <div data-testid="ratings-page" className="min-h-screen bg-obsidian pt-24 pb-16">
      <div className="max-w-4xl mx-auto px-6">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <div className="w-12 h-12 rounded-xl bg-gold/20 flex items-center justify-center">
            <Star className="w-6 h-6 text-gold" />
          </div>
          <div>
            <h1 className="font-heading text-3xl md:text-4xl font-bold tracking-tight">
              My Ratings
            </h1>
            <p className="text-muted-foreground">
              {ratings.length} {ratings.length === 1 ? 'movie' : 'movies'} rated • Average: {averageRating}★
            </p>
          </div>
        </div>

        {/* Content */}
        {loading ? (
          <div className="space-y-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-32 rounded-xl bg-obsidian-lighter" />
            ))}
          </div>
        ) : ratings.length > 0 ? (
          <div className="space-y-4">
            {ratings.map((rating) => {
              const posterUrl = rating.poster_path 
                ? `https://image.tmdb.org/t/p/w185${rating.poster_path}`
                : null;
              
              return (
                <div 
                  key={rating.movie_id}
                  data-testid={`rating-item-${rating.movie_id}`}
                  className="flex gap-4 p-4 rounded-xl bg-obsidian-light border border-white/5 hover:border-white/10 transition-colors group"
                >
                  {/* Poster */}
                  <Link to={`/movie/${rating.movie_id}`} className="flex-shrink-0">
                    {posterUrl ? (
                      <img
                        src={posterUrl}
                        alt={rating.title}
                        className="w-20 h-28 rounded-lg object-cover"
                      />
                    ) : (
                      <div className="w-20 h-28 rounded-lg bg-obsidian-lighter flex items-center justify-center">
                        <Film className="w-8 h-8 text-muted-foreground" />
                      </div>
                    )}
                  </Link>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <Link 
                      to={`/movie/${rating.movie_id}`}
                      className="font-heading text-lg font-semibold hover:text-electric transition-colors line-clamp-1"
                    >
                      {rating.title}
                    </Link>
                    
                    <div className="flex items-center gap-2 mt-1 mb-2">
                      <StarRating rating={rating.rating} readonly size="sm" />
                      <span className="text-gold font-semibold">{rating.rating}/5</span>
                    </div>

                    {rating.review && (
                      <p className="text-sm text-white/70 line-clamp-2">
                        "{rating.review}"
                      </p>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex items-start gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      data-testid={`edit-rating-${rating.movie_id}`}
                      onClick={() => openEditDialog(rating)}
                      className="w-8 h-8 rounded-full bg-white/5 hover:bg-white/10 flex items-center justify-center transition-colors"
                    >
                      <Edit2 className="w-4 h-4" />
                    </button>
                    <button
                      data-testid={`delete-rating-${rating.movie_id}`}
                      onClick={() => setDeleteId(rating.movie_id)}
                      className="w-8 h-8 rounded-full bg-red-500/20 hover:bg-red-500/30 text-red-400 flex items-center justify-center transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <div className="w-20 h-20 rounded-2xl bg-white/5 flex items-center justify-center mb-6">
              <Star className="w-10 h-10 text-muted-foreground" />
            </div>
            <h2 className="font-heading text-2xl font-semibold mb-2">No ratings yet</h2>
            <p className="text-muted-foreground mb-6 max-w-md">
              Start rating movies you've watched. Your reviews help you remember what you loved!
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
            <AlertDialogTitle className="font-heading">Delete Rating?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete your rating and review for this movie.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="rounded-full">Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleDelete}
              className="rounded-full bg-red-500 hover:bg-red-600"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Edit Dialog */}
      <Dialog open={!!editRating} onOpenChange={() => setEditRating(null)}>
        <DialogContent className="bg-obsidian-light border-white/10">
          <DialogHeader>
            <DialogTitle className="font-heading text-xl">
              Edit Rating: {editRating?.title}
            </DialogTitle>
          </DialogHeader>
          <div className="py-4 space-y-6">
            <div className="flex flex-col items-center gap-4">
              <StarRating 
                rating={editValue} 
                onRate={setEditValue} 
                size="lg" 
              />
              <span className="text-2xl font-semibold text-gold">
                {editValue > 0 ? `${editValue}/5` : 'Select rating'}
              </span>
            </div>
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">
                Review (optional)
              </label>
              <Textarea
                data-testid="edit-review-input"
                value={editReview}
                onChange={(e) => setEditReview(e.target.value)}
                placeholder="Share your thoughts about this movie..."
                className="bg-obsidian border-white/10 focus:border-electric resize-none"
                rows={4}
              />
            </div>
            <div className="flex justify-end gap-3">
              <Button
                variant="ghost"
                onClick={() => setEditRating(null)}
                className="rounded-full"
              >
                Cancel
              </Button>
              <Button
                data-testid="save-edit-btn"
                onClick={handleSaveEdit}
                className="rounded-full bg-electric hover:bg-electric/80"
              >
                Save Changes
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default RatingsPage;
