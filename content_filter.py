import pandas as pd
from typing import List

def filter_by_genres(movies: pd.DataFrame, selected_genres: List[str], limit: int = 20, min_popularity: int = 3) -> List[str]:
    """
    Enhanced genre filtering to ensure movies match ALL selected genres.
    
    Args:
        movies (pd.DataFrame): The DataFrame of movies with genre columns.
        selected_genres (List[str]): A list of genres to filter by.
        limit (int): The maximum number of movie titles to return.
        
    Returns:
        List[str]: A list of movie titles that match all selected genres.
    """
    if not selected_genres:
        return []
    
    # Corrected logic: We now create a mask that checks if a movie belongs to ALL selected genres.
    # The sum of the genre columns must equal the number of selected genres.
    # This correctly implements a logical AND operation.
    try:
        mask = (movies[selected_genres].sum(axis=1) == len(selected_genres))
    except KeyError as e:
        print(f"Error in genre filtering: {e}. Please check your genre column names in data_loader.py.")
        return []

    filtered_movies = movies.loc[mask].copy()
    
    if filtered_movies.empty:
        return []
        
    # Shuffle and limit the results for diversity
    if len(filtered_movies) > limit:
        # Sample a random subset to prevent the same movies from being returned every time
        sampled = filtered_movies.sample(n=limit, random_state=42)
        return sampled['ML_Title'].tolist()
    else:
        return filtered_movies['ML_Title'].tolist()