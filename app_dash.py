import dash
from dash import html, dcc, Input, Output, State, callback_context
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

# Layout
app.layout = dbc.Container([
    html.H1("🎬 Advanced Hybrid Movie Recommender", className="text-center my-4"),

    # Model Performance Display
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Model Performance", className="card-title"),
                    html.Div([
                        html.P(f"RMSE: {eval_metrics['rmse']:.4f}"),
                        html.P(f"MAE: {eval_metrics['mae']:.4f}"),
                        html.P(f"Precision@10: {eval_metrics['precision']:.4f}"),
                        html.P(f"Recall@10: {eval_metrics['recall']:.4f}"),
                        html.P(f"F1@10: {eval_metrics['f1']:.4f}"),
                        html.P(f"NDCG@10: {eval_metrics['ndcg']:.4f}"),
                    ])
                ])
            ], className="mb-4")
        ])
    ]),

    dbc.Row([
        # CF Recommendations Section
        dbc.Col([
            html.H3("🤖 Personalized Recommendations"),
            html.P("Select a user ID to get recommendations based on their past ratings.", className="text-muted"),
            dcc.Dropdown(
                id='user-dropdown',
                options=[{'label': u, 'value': u} for u in df_ratings['user'].unique()],
                value=df_ratings['user'].unique()[0],
                clearable=False
            ),
            html.Br(),
            dbc.ListGroup(id='cf-recommendations')
        ]),

        # Genre Filter Section
        dbc.Col([
            html.H3("🎭 Explore by Genre"),
            html.P("Select one or more genres to see movies in those categories.", className="text-muted"),
            dbc.Checklist(
                options=[{'label': g, 'value': g} for g in genre_cols if g != 'unknown'],
                value=[],
                id='genre-checklist',
                inline=True
            ),
            html.Br(),
            dbc.ListGroup(id='genre-results')
        ]),

        # Search Section
        dbc.Col([
            html.H3("🔍 Search & Hybrid Recommendations"),
            html.P("Get smart recommendations based on a movie title or keyword.", className="text-muted"),
            dbc.InputGroup(
                [
                    dbc.Input(id='search-input', type='text', placeholder='e.g., Star Wars, The Godfather...', debounce=True),
                    dbc.Button("Search", id="search-button", n_clicks=0, color="primary")
                ],
                className="mb-3"
            ),
            html.Div(id='search-results')
        ])
    ])
], fluid=True)

# -----------------------------
# Callbacks
# -----------------------------
# CF Recommendations
@app.callback(
    Output('cf-recommendations', 'children'),
    Input('user-dropdown', 'value')
)
def update_cf_recs(user_id):
    if not user_id:
        return ""
    recs = get_cf_recommendations(user_id, df_ratings, df_movies, cf_algo, N=10)
    if not recs:
        return dbc.Alert(f"No personalized recommendations found for User {user_id}.", color="info")
    return dbc.ListGroup([dbc.ListGroupItem(f"🎬 {title}") for title in recs])

# Genre Filter
@app.callback(
    Output('genre-results', 'children'),
    Input('genre-checklist', 'value')
)
def update_genre_results(selected_genres):
    try:
        if not selected_genres:
            return html.P("Select genres to see recommendations.", className="text-muted")

        filtered_titles = filter_by_genres(df_movies, selected_genres, limit=20)
        
        if not filtered_titles:
            return dbc.Alert("No movies found for selected genres.", color="info")
        
        return dbc.ListGroup([
            dbc.ListGroupItem(f"🎬 {title}") for title in filtered_titles
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error filtering by genres: {str(e)}", color="danger")

# Search
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks'), Input('search-input', 'n_submit')],
    State('search-input', 'value')
)
def update_search_results(n_clicks, n_submit, query):
    if not query or (not n_clicks and not n_submit):
        return html.P("Enter a movie title or keyword to get smart recommendations.", className="text-muted")
    
    try:
        # Get the hybrid recommendations
        recs = recommend_from_search(
            query, vectorizer, tfidf_matrix, df_movies, cf_algo, df_ratings,
            top_search=5, top_rec=10
        )
        
        # Check for the direct search result
        search_result_df = df_movies[df_movies['processed_title'] == re.sub(r'[^\\w\\s]', ' ', query.lower().strip())]
        
        # If the searched movie is found, add it to the front of the recommendations list
        if not search_result_df.empty:
            searched_title = search_result_df.iloc[0]['ML_Title']
            if searched_title not in recs:
                recs.insert(0, searched_title)
        
        if not recs:
            return dbc.Alert("❌ No recommendations found. Try different keywords.", color="warning")
        
        # Format the output to clearly label the searched movie
        list_items = []
        for i, rec in enumerate(recs):
            is_searched_movie = (not search_result_df.empty and rec == searched_title)
            
            if is_searched_movie:
                list_items.append(
                    dbc.ListGroupItem(html.Div([
                        html.Strong(f"🎬 {rec}"),
                        html.Small(" (Your Search)", className="text-muted ms-2")
                    ]))
                )
            else:
                list_items.append(
                    dbc.ListGroupItem(html.Div([
                        html.Strong(f"🎯 {rec}"),
                        html.Small(" (Hybrid recommendation)", className="text-muted ms-2")
                    ]))
                )
        
        return dbc.ListGroup(list_items)
        
    except Exception as e:
        return dbc.Alert(f"Search error: {str(e)}", color="danger")

if __name__ == '__main__':
    app.run(debug=True)