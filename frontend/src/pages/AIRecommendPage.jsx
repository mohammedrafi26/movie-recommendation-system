import { useState } from 'react';
import { Sparkles, Wand2, Film, Heart, Zap, Moon, Smile, Ghost, Swords, Search } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import MovieCard from '../components/MovieCard';
import { Skeleton } from '../components/ui/skeleton';
import { getAIRecommendations } from '../api';
import { toast } from 'sonner';

const MOODS = [
  { id: 'happy', label: 'Happy', icon: Smile, color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' },
  { id: 'romantic', label: 'Romantic', icon: Heart, color: 'bg-pink-500/20 text-pink-400 border-pink-500/30' },
  { id: 'thrilling', label: 'Thrilling', icon: Zap, color: 'bg-orange-500/20 text-orange-400 border-orange-500/30' },
  { id: 'scary', label: 'Scary', icon: Ghost, color: 'bg-purple-500/20 text-purple-400 border-purple-500/30' },
  { id: 'relaxing', label: 'Relaxing', icon: Moon, color: 'bg-blue-500/20 text-blue-400 border-blue-500/30' },
  { id: 'adventurous', label: 'Adventurous', icon: Swords, color: 'bg-green-500/20 text-green-400 border-green-500/30' },
];

const GENRES = [
  'Action', 'Comedy', 'Drama', 'Horror', 'Romance', 
  'Sci-Fi', 'Thriller', 'Animation', 'Documentary', 'Fantasy',
  'Mystery', 'Adventure', 'Crime', 'Family', 'Musical'
];

const AIRecommendPage = () => {
  const [selectedMood, setSelectedMood] = useState(null);
  const [selectedGenres, setSelectedGenres] = useState([]);
  const [description, setDescription] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const toggleGenre = (genre) => {
    setSelectedGenres(prev => 
      prev.includes(genre)
        ? prev.filter(g => g !== genre)
        : prev.length < 5 ? [...prev, genre] : prev
    );
  };

  const handleGetRecommendations = async () => {
    if (!selectedMood && selectedGenres.length === 0 && !description) {
      toast.error('Please select a mood, genres, or describe what you want');
      return;
    }

    setLoading(true);
    setHasSearched(true);

    try {
      const res = await getAIRecommendations({
        mood: selectedMood,
        genres: selectedGenres.length > 0 ? selectedGenres : null,
        description: description || null
      });

      setRecommendations(res.data.recommendations || []);
      
      if (res.data.recommendations?.length === 0) {
        toast.info('No recommendations found. Try different preferences.');
      }
    } catch (error) {
      console.error('Failed to get recommendations:', error);
      toast.error('Failed to get recommendations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedMood(null);
    setSelectedGenres([]);
    setDescription('');
    setRecommendations([]);
    setHasSearched(false);
  };

  return (
    <div data-testid="ai-recommend-page" className="min-h-screen bg-obsidian pt-24 pb-16">
      <div className="max-w-4xl mx-auto px-6">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-electric/20 mb-6">
            <Sparkles className="w-8 h-8 text-electric" />
          </div>
          <h1 className="font-heading text-4xl md:text-5xl font-bold tracking-tight mb-4">
            AI Movie Picker
          </h1>
          <p className="text-lg text-muted-foreground max-w-xl mx-auto">
            Tell us how you're feeling and what you're in the mood for. Our AI will find the perfect movies for you.
          </p>
        </div>

        {/* Selection Form */}
        <div className="space-y-8 mb-12">
          {/* Mood Selection */}
          <div>
            <h2 className="font-heading text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="text-2xl">😊</span>
              How are you feeling?
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {MOODS.map(mood => {
                const Icon = mood.icon;
                const isSelected = selectedMood === mood.id;
                return (
                  <button
                    key={mood.id}
                    data-testid={`mood-${mood.id}`}
                    onClick={() => setSelectedMood(isSelected ? null : mood.id)}
                    className={`flex items-center gap-3 p-4 rounded-xl border transition-all duration-300 ${
                      isSelected 
                        ? `${mood.color} scale-[1.02]`
                        : 'bg-white/5 border-white/10 hover:bg-white/10'
                    }`}
                  >
                    <Icon className="w-5 h-5" strokeWidth={1.5} />
                    <span className="font-medium">{mood.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Genre Selection */}
          <div>
            <h2 className="font-heading text-xl font-semibold mb-4 flex items-center gap-2">
              <Film className="w-6 h-6 text-electric" />
              Pick your genres <span className="text-sm font-normal text-muted-foreground">(up to 5)</span>
            </h2>
            <div className="flex flex-wrap gap-2">
              {GENRES.map(genre => {
                const isSelected = selectedGenres.includes(genre);
                return (
                  <button
                    key={genre}
                    data-testid={`genre-btn-${genre.toLowerCase()}`}
                    onClick={() => toggleGenre(genre)}
                    className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
                      isSelected 
                        ? 'bg-electric text-white' 
                        : 'bg-white/5 text-white/70 hover:bg-white/10 hover:text-white'
                    }`}
                  >
                    {genre}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Description */}
          <div>
            <h2 className="font-heading text-xl font-semibold mb-4 flex items-center gap-2">
              <Wand2 className="w-6 h-6 text-electric" />
              Describe what you want
            </h2>
            <Textarea
              data-testid="ai-description-input"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="E.g., 'I want something like Inception but with more humor' or 'A heartwarming story about friendship'"
              className="bg-obsidian-light border-white/10 focus:border-electric resize-none h-24"
            />
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4">
            <Button
              data-testid="get-recommendations-btn"
              onClick={handleGetRecommendations}
              disabled={loading}
              className="flex-1 h-14 rounded-full bg-electric hover:bg-electric/80 text-lg font-semibold glow-primary"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-3" />
                  Finding Movies...
                </>
              ) : (
                <>
                  <Search className="w-5 h-5 mr-3" />
                  Get Recommendations
                </>
              )}
            </Button>
            {hasSearched && (
              <Button
                data-testid="reset-btn"
                onClick={resetForm}
                variant="outline"
                className="h-14 px-8 rounded-full bg-transparent border-white/20 hover:bg-white/10"
              >
                Start Over
              </Button>
            )}
          </div>
        </div>

        {/* Results */}
        {loading && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {[...Array(6)].map((_, i) => (
              <Skeleton key={i} className="aspect-[2/3] rounded-xl bg-obsidian-lighter" />
            ))}
          </div>
        )}

        {!loading && hasSearched && recommendations.length > 0 && (
          <div>
            <h2 className="font-heading text-2xl font-semibold mb-6 flex items-center gap-3">
              <Sparkles className="w-6 h-6 text-gold" />
              AI Picks for You
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {recommendations.map((movie, index) => (
                <div 
                  key={movie.id} 
                  className="animate-fade-in-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <MovieCard movie={movie} />
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && hasSearched && recommendations.length === 0 && (
          <div className="text-center py-16">
            <p className="text-xl text-muted-foreground mb-4">
              No movies found matching your preferences
            </p>
            <Button onClick={resetForm} className="rounded-full">
              Try Different Preferences
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIRecommendPage;
