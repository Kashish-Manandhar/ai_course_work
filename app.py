import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('TMDB_movie_dataset_v11.csv')
    df = df.dropna()
    df['soup'] = (
        df['overview'] + ' ' +
        df['genres'] + ' ' +
        df['keywords'] + ' ' +
        df['tagline']
    ).str.strip()
    df['title_lower'] = df['title'].str.lower()
    return df

df = load_data()

# Create TF-IDF matrix and similarity matrix
@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title_lower']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = compute_similarity(df)


# Recommendation function
def recommend(title, df=df, cosine_sim=cosine_sim, indices=indices, k=5):
    title = title.lower()
    if title not in indices:
        return pd.DataFrame(columns=['title', 'vote_average', 'popularity', 'overview'])

    
    

    idx = indices[title]
    print(idx)
    sim_scores = list(enumerate(cosine_sim[idx]))
    print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]  # Exclude the input movie

    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'vote_average', 'popularity', 'overview']]

# ---------------- Streamlit UI ----------------

st.title("🎬 Movie Recommendation System")

st.write("Type in a movie title to get similar movie recommendations based on content.")

title_input = st.text_input("Enter a movie title:", "")

if title_input:
    results = recommend(title_input)
    if results.empty:
        st.warning(f"No results found for '{title_input}'. Please check the spelling or try another title.")
    else:
        st.success(f"Top {len(results)} recommendations for '{title_input}':")
        for _, row in results.iterrows():
            st.markdown(f"### {row['title']}")
            st.markdown(f"**Rating:** {row['vote_average']} | **Popularity:** {round(row['popularity'], 2)}")
            st.markdown(f"**Overview:** {row['overview']}")
            st.markdown("---")
