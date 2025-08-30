from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import numpy as np

def train_cf(df_ratings):
    """Enhanced CF training with hyperparameter optimization"""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_ratings[['user','item','rate']], reader)
    
    # Hyperparameter tuning for better accuracy
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.1]
    }
    
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    
    # Get best algorithm
    best_algo = gs.best_estimator['rmse']
    
    # Train on full dataset with best parameters
    trainset = data.build_full_trainset()
    best_algo.fit(trainset)
    
    # Create test set for evaluation
    trainset_eval, testset = train_test_split(data, test_size=0.2, random_state=42)
    eval_algo = SVD(**gs.best_params['rmse'])
    eval_algo.fit(trainset_eval)
    
    return best_algo, eval_algo, testset

def get_cf_recommendations(user_id, df_ratings, df_movies, cf_algo, N=10, min_rating_threshold=4.0):
    """Improved CF recommendations with rating threshold and popularity bias handling"""
    
    # Get user's rated items
    user_ratings = df_ratings[df_ratings['user'] == str(user_id)]
    rated_items = set(user_ratings['item'])
    
    # Filter out very unpopular movies (less than 5 ratings) for better quality
    item_popularity = df_ratings['item'].value_counts()
    popular_items = item_popularity[item_popularity >= 5].index
    
    all_items = set(df_ratings['item'].unique())
    unrated_popular = list((all_items & set(popular_items)) - rated_items)
    
    if not unrated_popular:
        unrated_popular = list(all_items - rated_items)
    
    # Get predictions
    preds = []
    for item in unrated_popular:
        pred = cf_algo.predict(str(user_id), str(item))
        if pred.est >= min_rating_threshold:  # Only recommend highly predicted items
            preds.append(pred)
    
    # Sort by prediction confidence
    preds.sort(key=lambda x: x.est, reverse=True)
    
    # Get top items
    top_items = [p.iid for p in preds[:N]]
    recommended_movies = df_movies[df_movies['ML_ID'].isin(top_items)]['ML_Title'].tolist()
    
    return recommended_movies