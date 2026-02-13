#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime
import time

class MovieAPITester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_movie_id = None
        self.test_results = []

    def log_test(self, name, status, details=""):
        """Log test result"""
        self.tests_run += 1
        if status:
            self.tests_passed += 1
            print(f"✅ {name}: PASSED {details}")
        else:
            print(f"❌ {name}: FAILED {details}")
        
        self.test_results.append({
            "test": name,
            "status": "PASSED" if status else "FAILED",
            "details": details
        })

    def test_health_check(self):
        """Test basic API health"""
        try:
            response = requests.get(f"{self.api_base}/", timeout=10)
            status = response.status_code == 200
            self.log_test("Health Check", status, f"Status: {response.status_code}")
            return status
        except Exception as e:
            self.log_test("Health Check", False, f"Error: {str(e)}")
            return False

    def test_tmdb_movies_endpoints(self):
        """Test all TMDB movie category endpoints"""
        endpoints = [
            ("Popular Movies", "/movies/popular"),
            ("Trending Movies", "/movies/trending"),
            ("Top Rated Movies", "/movies/top-rated"),
            ("Upcoming Movies", "/movies/upcoming"),
            ("Now Playing Movies", "/movies/now-playing")
        ]
        
        all_passed = True
        for name, endpoint in endpoints:
            try:
                response = requests.get(f"{self.api_base}{endpoint}", timeout=15)
                has_results = False
                movie_data = None
                
                if response.status_code == 200:
                    data = response.json()
                    has_results = len(data.get('results', [])) > 0
                    if has_results and not self.test_movie_id:
                        # Store first movie ID for detail testing
                        self.test_movie_id = data['results'][0]['id']
                        movie_data = data['results'][0]
                
                status = response.status_code == 200 and has_results
                details = f"Status: {response.status_code}, Results: {len(data.get('results', []))}"
                if movie_data:
                    details += f", Sample: {movie_data.get('title', 'Unknown')}"
                
                self.log_test(name, status, details)
                if not status:
                    all_passed = False
                    
            except Exception as e:
                self.log_test(name, False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_movie_search(self):
        """Test movie search functionality"""
        try:
            # Test with popular movie title
            response = requests.get(f"{self.api_base}/movies/search?query=Inception", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                has_results = len(data.get('results', [])) > 0
                status = has_results
                details = f"Status: {response.status_code}, Results: {len(data.get('results', []))}"
            else:
                status = False
                details = f"Status: {response.status_code}"
                
            self.log_test("Movie Search", status, details)
            return status
            
        except Exception as e:
            self.log_test("Movie Search", False, f"Error: {str(e)}")
            return False

    def test_genres(self):
        """Test genres endpoint"""
        try:
            response = requests.get(f"{self.api_base}/movies/genres", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                has_genres = len(data.get('genres', [])) > 0
                status = has_genres
                details = f"Status: {response.status_code}, Genres: {len(data.get('genres', []))}"
            else:
                status = False
                details = f"Status: {response.status_code}"
                
            self.log_test("Get Genres", status, details)
            return status
            
        except Exception as e:
            self.log_test("Get Genres", False, f"Error: {str(e)}")
            return False

    def test_movie_details(self):
        """Test movie details endpoint"""
        if not self.test_movie_id:
            self.log_test("Movie Details", False, "No test movie ID available")
            return False
            
        try:
            response = requests.get(f"{self.api_base}/movies/{self.test_movie_id}", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                has_title = 'title' in data
                has_overview = 'overview' in data
                status = has_title and has_overview
                details = f"Status: {response.status_code}, Title: {data.get('title', 'Missing')}"
                if data.get('cast'):
                    details += f", Cast: {len(data.get('cast', []))}"
            else:
                status = False
                details = f"Status: {response.status_code}"
                
            self.log_test("Movie Details", status, details)
            return status
            
        except Exception as e:
            self.log_test("Movie Details", False, f"Error: {str(e)}")
            return False

    def test_discover_movies(self):
        """Test movie discovery with filters"""
        try:
            # Test with action genre (28)
            response = requests.get(f"{self.api_base}/movies/discover?genre_id=28", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                has_results = len(data.get('results', [])) > 0
                status = has_results
                details = f"Status: {response.status_code}, Results: {len(data.get('results', []))}"
            else:
                status = False
                details = f"Status: {response.status_code}"
                
            self.log_test("Discover Movies", status, details)
            return status
            
        except Exception as e:
            self.log_test("Discover Movies", False, f"Error: {str(e)}")
            return False

    def test_ai_recommendations(self):
        """Test AI recommendations endpoint"""
        try:
            payload = {
                "mood": "happy",
                "genres": ["Comedy", "Action"],
                "description": "Something funny and exciting"
            }
            
            response = requests.post(
                f"{self.api_base}/ai/recommend", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                has_recommendations = len(data.get('recommendations', [])) > 0
                status = has_recommendations
                details = f"Status: {response.status_code}, Recommendations: {len(data.get('recommendations', []))}"
                if data.get('titles'):
                    details += f", Titles: {data['titles'][:2]}"
            else:
                status = False
                details = f"Status: {response.status_code}"
                if response.text:
                    details += f", Error: {response.text[:100]}"
                
            self.log_test("AI Recommendations", status, details)
            return status
            
        except Exception as e:
            self.log_test("AI Recommendations", False, f"Error: {str(e)}")
            return False

    def test_watchlist_operations(self):
        """Test watchlist CRUD operations"""
        if not self.test_movie_id:
            self.log_test("Watchlist Operations", False, "No test movie ID available")
            return False
        
        try:
            # Test add to watchlist
            payload = {
                "movie_id": self.test_movie_id,
                "title": "Test Movie",
                "poster_path": "/test.jpg",
                "release_date": "2023-01-01",
                "vote_average": 8.0
            }
            
            add_response = requests.post(f"{self.api_base}/watchlist", json=payload, timeout=10)
            add_success = add_response.status_code in [200, 201]
            
            # Test get watchlist
            get_response = requests.get(f"{self.api_base}/watchlist", timeout=10)
            get_success = get_response.status_code == 200
            
            # Test check watchlist
            check_response = requests.get(f"{self.api_base}/watchlist/check/{self.test_movie_id}", timeout=10)
            check_success = check_response.status_code == 200
            
            # Test remove from watchlist
            remove_response = requests.delete(f"{self.api_base}/watchlist/{self.test_movie_id}", timeout=10)
            remove_success = remove_response.status_code in [200, 404]  # 404 is ok if not in list
            
            status = add_success and get_success and check_success
            details = f"Add: {add_response.status_code}, Get: {get_response.status_code}, Check: {check_response.status_code}, Remove: {remove_response.status_code}"
            
            self.log_test("Watchlist Operations", status, details)
            return status
            
        except Exception as e:
            self.log_test("Watchlist Operations", False, f"Error: {str(e)}")
            return False

    def test_rating_operations(self):
        """Test rating CRUD operations"""
        if not self.test_movie_id:
            self.log_test("Rating Operations", False, "No test movie ID available")
            return False
        
        try:
            # Test add rating
            payload = {
                "movie_id": self.test_movie_id,
                "title": "Test Movie",
                "poster_path": "/test.jpg",
                "rating": 4.5,
                "review": "Great movie for testing!"
            }
            
            add_response = requests.post(f"{self.api_base}/ratings", json=payload, timeout=10)
            add_success = add_response.status_code in [200, 201]
            
            # Test get ratings
            get_response = requests.get(f"{self.api_base}/ratings", timeout=10)
            get_success = get_response.status_code == 200
            
            # Test get specific rating
            rating_response = requests.get(f"{self.api_base}/ratings/{self.test_movie_id}", timeout=10)
            rating_success = rating_response.status_code == 200
            
            # Test delete rating
            delete_response = requests.delete(f"{self.api_base}/ratings/{self.test_movie_id}", timeout=10)
            delete_success = delete_response.status_code in [200, 404]  # 404 is ok if not found
            
            status = add_success and get_success and rating_success
            details = f"Add: {add_response.status_code}, Get: {get_response.status_code}, Rating: {rating_response.status_code}, Delete: {delete_response.status_code}"
            
            self.log_test("Rating Operations", status, details)
            return status
            
        except Exception as e:
            self.log_test("Rating Operations", False, f"Error: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🎬 Starting Movie Recommendation API Tests...")
        print(f"Base URL: {self.base_url}")
        print("=" * 60)
        
        # Health check first
        if not self.test_health_check():
            print("❌ Health check failed - skipping other tests")
            return False
        
        # Core movie endpoints
        self.test_tmdb_movies_endpoints()
        self.test_movie_search()
        self.test_genres()
        self.test_movie_details()
        self.test_discover_movies()
        
        # AI functionality
        self.test_ai_recommendations()
        
        # User features
        self.test_watchlist_operations()
        self.test_rating_operations()
        
        # Summary
        print("=" * 60)
        print(f"📊 Tests Summary: {self.tests_passed}/{self.tests_run} passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed. Check the details above.")
            return False

def main():
    # Get backend URL from frontend env
    backend_url = "https://filmwise-3.preview.emergentagent.com"
    
    tester = MovieAPITester(backend_url)
    success = tester.run_all_tests()
    
    # Save detailed results
    try:
        with open('/app/test_results.json', 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "base_url": backend_url,
                "summary": {
                    "tests_run": tester.tests_run,
                    "tests_passed": tester.tests_passed,
                    "success_rate": round((tester.tests_passed / tester.tests_run) * 100, 2) if tester.tests_run > 0 else 0
                },
                "detailed_results": tester.test_results
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save test results: {e}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())