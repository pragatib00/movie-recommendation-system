import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
import os
import gdown

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="Movie Matcher", layout="wide")
warnings.filterwarnings('ignore')

st.markdown("""
    <style>
    .movie-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        border-left: 6px solid #e50914;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: scale(1.02); }
    .genre-text { color: #555; font-size: 0.9rem; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING (Cloud-Ready Version) ---
@st.cache_resource
def load_data():
    file_path = 'movie_model.pkl'
    # Extracted from your Google Drive link
    file_id = '1FtWuskctC8p4xOWQSGhYgRjb8CzYRsqO'
    url = f'https://drive.google.com/uc?id={file_id}'

    # Download from Google Drive if not present locally on the server
    if not os.path.exists(file_path):
        with st.spinner("Downloading model file from Google Drive... This may take a minute due to file size (415MB)."):
            try:
                gdown.download(url, file_path, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the model data
data = load_data()

if data:
    knn = data['knn']
    user_movie_sparse = data['user_movie_sparse']
    user_mapping = data['user_mapping']
    movie_mapping = data['movie_mapping']
    movies_df = data['movies_df']

    # Set up ID mappings
    u_to_idx = {v: k for k, v in user_mapping.items()}
    idx_to_m = movie_mapping
else:
    st.stop() # Stop execution if data failed to load

# --- 3. RECOMMENDATION ENGINE ---
def get_recommendations(user_id, top_n=10):
    if user_id not in u_to_idx:
        return None
    
    u_idx = u_to_idx[user_id]
    distances, indices = knn.kneighbors(user_movie_sparse[u_idx], n_neighbors=21)
    
    neighbor_indices = indices[0][1:]
    similarities = 1 - distances[0][1:]
    watched_indices = set(user_movie_sparse[u_idx].indices)
    
    movie_scores = {}
    total_sim = {} 

    for idx, sim_weight in zip(neighbor_indices, similarities):
        neighbor_row = user_movie_sparse[idx]
        for m_idx, rating in zip(neighbor_row.indices, neighbor_row.data):
            if m_idx not in watched_indices:
                movie_scores[m_idx] = movie_scores.get(m_idx, 0) + (sim_weight * rating)
                total_sim[m_idx] = total_sim.get(m_idx, 0) + sim_weight

    if not movie_scores:
        return pd.DataFrame()

    normalized_scores = []
    for m_idx in movie_scores:
        final_score = movie_scores[m_idx] / total_sim[m_idx]
        normalized_scores.append((m_idx, final_score))

    top_items = sorted(normalized_scores, key=lambda x: x[1], reverse=True)[:top_n]
    actual_ids = [idx_to_m[item[0]] for item in top_items]
    
    return movies_df[movies_df['movieId'].isin(actual_ids)]

# --- 4. STREAMLIT UI ---
st.title(" Personalized Movie Matcher")
st.markdown("Discover hidden gems based on users with similar tastes.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("User Selection")
    user_id_input = st.number_input("Enter your User ID:", min_value=1, step=1)
    num_recs = st.select_slider("How many movies?", options=[5, 10, 15, 20], value=10)
    trigger = st.button("Generate My List")

with col2:
    if trigger:
        with st.spinner("Finding people like you..."):
            recs = get_recommendations(user_id_input, num_recs)
            
            if recs is None:
                st.error(" User ID not found.")
            elif recs.empty:
                st.warning("No new recommendations found.")
            else:
                st.success(f"Top {len(recs)} picks for User {user_id_input}:")
                grid = st.columns(2)
                for i, (_, row) in enumerate(recs.iterrows()):
                    with grid[i % 2]:
                        st.markdown(f"""
                            <div class="movie-card">
                                <div style="font-size: 1.1rem; font-weight: bold;">{row['title']}</div>
                                <div class="genre-text">{row['genres']}</div>
                            </div>
                        """, unsafe_allow_html=True)

st.divider()
st.caption("Machine Learning Model: K-Nearest Neighbors | Developed by Pragati Basnet")