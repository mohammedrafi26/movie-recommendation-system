import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "./components/ui/sonner";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import BrowsePage from "./pages/BrowsePage";
import MovieDetailPage from "./pages/MovieDetailPage";
import AIRecommendPage from "./pages/AIRecommendPage";
import WatchlistPage from "./pages/WatchlistPage";
import RatingsPage from "./pages/RatingsPage";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/browse" element={<BrowsePage />} />
            <Route path="/movie/:id" element={<MovieDetailPage />} />
            <Route path="/recommend" element={<AIRecommendPage />} />
            <Route path="/watchlist" element={<WatchlistPage />} />
            <Route path="/ratings" element={<RatingsPage />} />
          </Routes>
        </Layout>
      </BrowserRouter>
      <Toaster position="bottom-right" richColors />
    </div>
  );
}

export default App;
