import pandas as pd
from dash import dcc

def load_data():
    genre_cols = ['unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime',
                  'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery',
                  'Romance','Sci-Fi','Thriller','War','Western']
    
    df_ratings = pd.read_csv('data/ml-100k/u.data', sep='\t',
                             names=['user','item','rate','timestamp'], dtype={'user':str,'item':str})
    
    df_movies = pd.read_csv('data/ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                            usecols=[0,1]+list(range(5,24)))
    df_movies.columns = ['ML_ID','ML_Title'] + genre_cols
    df_movies['ML_ID'] = df_movies['ML_ID'].astype(str)
    
    return df_ratings, df_movies, genre_cols
