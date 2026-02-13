import { Link, useLocation } from 'react-router-dom';
import { Film, Search, Sparkles, Bookmark, Star, Menu, X } from 'lucide-react';
import { useState, useEffect } from 'react';

export const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { to: '/', label: 'Home', icon: Film },
    { to: '/browse', label: 'Browse', icon: Search },
    { to: '/recommend', label: 'AI Pick', icon: Sparkles },
    { to: '/watchlist', label: 'Watchlist', icon: Bookmark },
    { to: '/ratings', label: 'Ratings', icon: Star },
  ];

  return (
    <nav 
      data-testid="navbar"
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled ? 'bg-black/80 backdrop-blur-lg border-b border-white/5' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group" data-testid="logo-link">
            <div className="w-10 h-10 rounded-xl bg-electric flex items-center justify-center glow-primary">
              <Film className="w-5 h-5 text-white" strokeWidth={2} />
            </div>
            <span className="font-heading text-2xl font-bold tracking-tight">FilmWise</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {navLinks.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                data-testid={`nav-${label.toLowerCase().replace(' ', '-')}`}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
                  location.pathname === to
                    ? 'bg-white/10 text-white'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon className="w-4 h-4" strokeWidth={1.5} />
                <span>{label}</span>
              </Link>
            ))}
          </div>

          {/* Mobile Menu Button */}
          <button
            data-testid="mobile-menu-btn"
            className="md:hidden p-2 rounded-lg hover:bg-white/10 transition-colors"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          >
            {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="md:hidden mt-4 py-4 border-t border-white/10 animate-fade-in-up">
            {navLinks.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                data-testid={`mobile-nav-${label.toLowerCase().replace(' ', '-')}`}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg text-base font-medium transition-all ${
                  location.pathname === to
                    ? 'bg-white/10 text-white'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                <Icon className="w-5 h-5" strokeWidth={1.5} />
                <span>{label}</span>
              </Link>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
};

export const Footer = () => {
  return (
    <footer data-testid="footer" className="bg-obsidian-light border-t border-white/5 py-12 mt-24">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-electric/20 flex items-center justify-center">
              <Film className="w-4 h-4 text-electric" strokeWidth={2} />
            </div>
            <span className="font-heading text-lg font-semibold">FilmWise</span>
          </div>
          <p className="text-sm text-muted-foreground text-center">
            Powered by TMDB & AI • Discover your next favorite movie
          </p>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>© 2026 FilmWise</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen bg-obsidian">
      <Navbar />
      <main>{children}</main>
      <Footer />
    </div>
  );
};

export default Layout;
