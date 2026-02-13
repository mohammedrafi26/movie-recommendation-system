import { Star } from 'lucide-react';
import { useState } from 'react';

const StarRating = ({ rating = 0, onRate, readonly = false, size = 'md' }) => {
  const [hoverRating, setHoverRating] = useState(0);

  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  const stars = [1, 2, 3, 4, 5];

  return (
    <div 
      className="flex items-center gap-1"
      onMouseLeave={() => !readonly && setHoverRating(0)}
    >
      {stars.map((star) => {
        const isFilled = star <= (hoverRating || rating);
        return (
          <button
            key={star}
            data-testid={`star-${star}`}
            type="button"
            disabled={readonly}
            onClick={() => !readonly && onRate?.(star)}
            onMouseEnter={() => !readonly && setHoverRating(star)}
            className={`transition-all duration-200 ${!readonly && 'cursor-pointer hover:scale-110'}`}
          >
            <Star
              className={`${sizeClasses[size]} ${
                isFilled 
                  ? 'text-gold fill-gold' 
                  : 'text-zinc-600'
              }`}
              strokeWidth={1.5}
            />
          </button>
        );
      })}
    </div>
  );
};

export default StarRating;
