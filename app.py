import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import re

# Import your improved modules
from data_loader import load_data
from cf_model import train_cf, get_cf_recommendations
from search_engine import build_search_index, recommend_from_search
from evaluation import evaluate_model
from content_filter import filter_by_genres

# -----------------------------
# Load Data & Train Models
# -----------------------------
print("📊 Loading data...")
df_ratings, df_movies, genre_cols = load_data()

print("🤖 Training collaborative filtering model...")
cf_algo, eval_algo, testset = train_cf(df_ratings)

print("📈 Evaluating model...")
eval_metrics = evaluate_model(eval_algo, testset)

print("🔍 Building search index...")
vectorizer, tfidf_matrix = build_search_index(df_movies)

print("✅ Setup complete!")

# -----------------------------
# Enhanced Dash App
# -----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H1("🎬 Advanced Hybrid Movie Recommender", className="text-center my-4"),

    # Personalized Recommendations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("👤 Personalized Recommendations", className="card-title"),
                    html.Label("Select User ID:"),
                    dcc.Dropdown(
                        id='user-dropdown',
                        options=[{'label': f'User {u}', 'value': u} for u in sorted(df_ratings['user'].unique()[:50])],
                        value=df_ratings['user'].unique()[0],
                        placeholder="Choose a user..."
                    ),
                    html.Br(),
                    dbc.Spinner(html.Div(id='cf-recommendations'))
                ])
            ], className="mb-4")
        ])
    ]),

    # Genre-Based Recommendations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🎭 Genre-Based Discovery", className="card-title"),
                    dbc.Checklist(
                        options=[{'label': g.replace("'", ""), 'value': g} for g in genre_cols if g != 'unknown'],
                        value=[],
                        id='genre-checklist',
                        inline=True,
                        style={'flexWrap': 'wrap'}
                    ),
                    html.Br(),
                    dbc.Spinner(html.Div(id='genre-results'))
                ])
            ], className="mb-4")
        ])
    ]),

    # Smart Search
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🔍 Smart Movie Search", className="card-title"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='search-input',
                            type='text',
                            placeholder='Enter movie title, genre, or keywords...',
                            debounce=True
                        ),
                        dbc.Button("Search", id="search-button", color="primary", n_clicks=0)
                    ]),
                    html.Br(),
                    dbc.Spinner(html.Div(id='search-results'))
                ])
            ])
        ])
    ])
], fluid=True)

# Enhanced Callbacks
@app.callback(
    Output('cf-recommendations', 'children'),
    Input('user-dropdown', 'value')
)
def update_cf_recs(user_id):
    if not user_id:
        return "Please select a user."
    
    try:
        recs = get_cf_recommendations(
            user_id, df_ratings, df_movies, cf_algo, N=8, min_rating_threshold=3.5
        )
        
        if not recs:
            return dbc.Alert("No recommendations found for this user.", color="warning")
        
        return dbc.ListGroup([
            dbc.ListGroupItem(f"⭐ {rec}") for rec in recs
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error generating recommendations: {str(e)}", color="danger")

@app.callback(
    Output('genre-results', 'children'),
    Input('genre-checklist', 'value')
)
def update_genre_results(selected_genres):
    if not selected_genres:
        return html.P("Select genres to discover movies.", className="text-muted")
    
    try:
        filtered_titles = filter_by_genres(df_movies, selected_genres, limit=12)
        
        if not filtered_titles:
            return dbc.Alert("No movies found for selected genres.", color="info")
        
        return dbc.ListGroup([
            dbc.ListGroupItem(f"🎬 {title}") for title in filtered_titles
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error filtering by genres: {str(e)}", color="danger")

# REPLACE your existing search callback with this single improved version

def find_movie_match(df_movies, query):
    """
    Enhanced movie matching function that handles various query formats
    """
    def preprocess_title(title):
        if pd.isna(title):
            return ""
        # Remove year, special chars, normalize spaces, lowercase
        title = re.sub(r'\(\d{4}\)', '', str(title))
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        return title.lower().strip()
    
    processed_query = preprocess_title(query)
    
    # Create processed_title column if it doesn't exist
    if 'processed_title' not in df_movies.columns:
        df_movies['processed_title'] = df_movies['ML_Title'].apply(preprocess_title)
    
    # 1. Exact match
    exact_matches = df_movies[df_movies['processed_title'] == processed_query]
    if not exact_matches.empty:
        return exact_matches.iloc[0]
    
    # 2. Substring match (query is contained in title)
    substring_matches = df_movies[
        df_movies['processed_title'].str.contains(processed_query, case=False, na=False, regex=False)
    ]
    if not substring_matches.empty:
        return substring_matches.iloc[0]
    
    # 3. All words present (for multi-word queries)
    if len(processed_query.split()) > 1:
        query_words = processed_query.split()
        word_pattern = '(?=.*' + ')(?=.*'.join(query_words) + ')'
        word_matches = df_movies[
            df_movies['processed_title'].str.contains(word_pattern, case=False, na=False, regex=True)
        ]
        if not word_matches.empty:
            return word_matches.iloc[0]
    
    return None

@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks'), Input('search-input', 'n_submit')],
    State('search-input', 'value')
)
def update_search_results(n_clicks, n_submit, query):
    if not query or (not n_clicks and not n_submit):
        return html.P("Enter a movie title or keyword to get smart recommendations.", className="text-muted")
    
    try:
        print(f"🔍 Searching for: '{query}'")  # Debug
        
        # Get hybrid recommendations
        recs = recommend_from_search(
            query, vectorizer, tfidf_matrix, df_movies, cf_algo, df_ratings,
            top_search=5, top_rec=10
        )
        
        # Find direct match using robust function
        matched_movie = find_movie_match(df_movies, query)
        
        # Add matched movie to the beginning if found and not already in recs
        searched_title = None
        if matched_movie is not None:
            searched_title = matched_movie['ML_Title']
            print(f"✅ Found direct match: '{searched_title}'")  # Debug
            
            if searched_title not in recs:
                recs.insert(0, searched_title)
                print(f"➕ Added to top of recommendations")  # Debug
            else:
                print(f"⚠️ Already in recommendations")  # Debug
        else:
            print(f"❌ No direct match found")  # Debug
        
        if not recs:
            return dbc.Alert("❌ No recommendations found. Try different keywords.", color="warning")
        
        # Format output
        list_items = []
        for i, rec in enumerate(recs):
            is_searched_movie = (i == 0 and searched_title and rec == searched_title)
            
            if is_searched_movie:
                list_items.append(
                    dbc.ListGroupItem([
                        html.Div([
                            html.Strong(f"🎬 {rec}"),
                            html.Small(" (Your Search)", className="text-success ms-2")
                        ])
                    ])
                )
            else:
                list_items.append(
                    dbc.ListGroupItem([
                        html.Div([
                            html.Strong(f"🎯 {rec}"),
                            html.Small(" (Recommendation)", className="text-muted ms-2")
                        ])
                    ])
                )
        
        return dbc.ListGroup(list_items)
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")  # Debug
        return dbc.Alert(f"Search error: {str(e)}", color="danger")    
# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8050)