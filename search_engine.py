from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from collections import Counter
from cf_model import get_cf_recommendations

def build_search_index(df_movies):
    """Enhanced search index with better text preprocessing"""
    
    # Enhanced text preprocessing
    def preprocess_title(title):
        # Remove year in parentheses
        title = re.sub(r'\(\d{4}\)', '', title)
        # Remove special characters but keep spaces
        title = re.sub(r'[^\w\s]', ' ', title)
        # Convert to lowercase and strip
        return title.lower().strip()
    
    # Combine title and genres for richer content representation
    df_movies['processed_title'] = df_movies['ML_Title'].apply(preprocess_title)
    
    # Create genre text representation
    genre_cols = ['Action','Adventure','Animation',"Children's",'Comedy','Crime',
                  'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
                  'Romance','Sci-Fi','Thriller','War','Western']
    
    def get_genres(row):
        return ' '.join([col.lower().replace("'", "") for col in genre_cols if row[col] == 1])
    
    df_movies['genre_text'] = df_movies.apply(get_genres, axis=1)
    
    # Combine title and genres
    df_movies['combined_text'] = df_movies['processed_title'] + ' ' + df_movies['genre_text']
    
    # Enhanced TF-IDF with better parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8
    )
    
    tfidf_matrix = vectorizer.fit_transform(df_movies['combined_text'])
    return vectorizer, tfidf_matrix

def search_movies(query, vectorizer, tfidf_matrix, df_movies, top_k=5, threshold=0.1):
    """
    Enhanced movie search with direct title match and fallback to TF-IDF.
    """
    query_processed = re.sub(r'[^\w\s]', ' ', query.lower().strip())
    
    # Step 1: Check for an exact, case-insensitive match on the preprocessed title.
    direct_match = df_movies[df_movies['processed_title'] == query_processed]
    if not direct_match.empty:
        # If a direct match is found, return it as the top result with a perfect score.
        return direct_match.iloc[[0]][['ML_Title', 'ML_ID']].copy().assign(similarity=1.0)

    try:
        # Step 2: Fallback to TF-IDF search if no direct match is found.
        query_vec = vectorizer.transform([query_processed])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get all movies with a similarity score above a minimal threshold (> 0)
        above_thresh_idx = [i for i, s in enumerate(sims) if s > 0]
        
        if not above_thresh_idx:
            return pd.DataFrame(columns=['ML_Title', 'ML_ID', 'similarity'])
        
        # Sort by similarity and get the top results
        sorted_idx = sorted(above_thresh_idx, key=lambda i: sims[i], reverse=True)[:top_k]
        results = df_movies.iloc[sorted_idx][['ML_Title', 'ML_ID']].copy()
        results['similarity'] = [sims[i] for i in sorted_idx]
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return pd.DataFrame(columns=['ML_Title', 'ML_ID', 'similarity'])

def recommend_from_search(query, vectorizer, tfidf_matrix, df_movies, cf_algo, df_ratings, top_search=5, top_rec=8):
    """Enhanced hybrid recommendation with better proxy user selection"""
    
    search_results = search_movies(query, vectorizer, tfidf_matrix, df_movies, top_k=top_search)
    if search_results.empty:
        return []

    all_recs = []
    recommendation_scores = Counter()
    
    for _, row in search_results.iterrows():
        movie_id = row['ML_ID']
        similarity_score = row['similarity']
        
        # Find users who rated this movie highly (>= 4 stars)
        high_raters = df_ratings[
            (df_ratings['item'] == movie_id) & 
            (df_ratings['rate'] >= 4)
        ]['user'].unique()
        
        if len(high_raters) > 0:
            # Use multiple proxy users for better coverage
            proxy_users = high_raters[:min(3, len(high_raters))]
            
            for proxy_user in proxy_users:
                try:
                    recs = get_cf_recommendations(
                        proxy_user, df_ratings, df_movies, cf_algo, N=top_rec
                    )
                    
                    # Weight recommendations by search similarity
                    for rec in recs:
                        recommendation_scores[rec] += similarity_score
                        
                except Exception as e:
                    print(f"Error getting recommendations for user {proxy_user}: {e}")
                    continue
    
    # Sort by weighted scores and return top recommendations
    if recommendation_scores:
        sorted_recs = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
        return [rec for rec, score in sorted_recs[:top_rec]]
    
    return []