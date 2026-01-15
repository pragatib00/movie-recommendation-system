import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="Movie Matcher", layout="wide")
warnings.filterwarnings('ignore')

# Custom CSS for better looking cards
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

# --- 2. DATA LOADING ---
@st.cache_resource
def load_data():
    # Ensure movie_model.pkl is in the same folder as app.py
    with open('movie_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

# Unpack data
data = load_data()
knn = data['knn']
user_movie_sparse = data['user_movie_sparse']
user_mapping = data['user_mapping']
movie_mapping = data['movie_mapping']
movies_df = data['movies_df']

# Reverse mappings for logic
u_to_idx = {v: k for k, v in user_mapping.items()}
idx_to_m = movie_mapping

# --- 3. RECOMMENDATION ENGINE ---
def get_recommendations(user_id, top_n=10):
    if user_id not in u_to_idx:
        return None
    
    u_idx = u_to_idx[user_id]
    
    # 1. Get neighbors (K=20 for better variety)
    distances, indices = knn.kneighbors(user_movie_sparse[u_idx], n_neighbors=21)
    
    neighbor_indices = indices[0][1:]
    similarities = 1 - distances[0][1:]
    
    # 2. Track already watched movies to avoid recommending them
    watched_movie_indices = set(user_movie_sparse[u_idx].indices)
    
    movie_scores = {}
    total_similarity = {} # For normalization

    # 3. Aggregate neighbor ratings with normalization
    for idx, sim_weight in zip(neighbor_indices, similarities):
        neighbor_row = user_movie_sparse[idx]
        
        for m_idx, rating in zip(neighbor_row.indices, neighbor_row.data):
            if m_idx not in watched_movie_indices:
                # Weighted score
                movie_scores[m_idx] = movie_scores.get(m_idx, 0) + (sim_weight * rating)
                # Sum of weights for normalization (prevents popularity bias)
                total_similarity[m_idx] = total_similarity.get(m_idx, 0) + sim_weight

    if not movie_scores:
        return pd.DataFrame()

    # 4. Normalize scores by dividing by sum of similarities
    normalized_scores = []
    for m_idx in movie_scores:
        final_score = movie_scores[m_idx] / total_similarity[m_idx]
        normalized_scores.append((m_idx, final_score))

    # 5. Sort and fetch movie details
    top_items = sorted(normalized_scores, key=lambda x: x[1], reverse=True)[:top_n]
    actual_ids = [idx_to_m[item[0]] for item in top_items]
    
    return movies_df[movies_df['movieId'].isin(actual_ids)]

# --- 4. STREAMLIT UI ---
st.title("🍿 Personalized Movie Matcher")
st.markdown("Discover hidden gems based on users with similar tastes to yours.")

# Layout with two columns for input
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("User Selection")
    user_id_input = st.number_input("Enter your User ID:", min_value=1, step=1, help="The ID used in the MovieLens dataset.")
    num_recs = st.select_slider("How many movies?", options=[5, 10, 15, 20], value=10)
    trigger = st.button("Generate My List")

with col2:
    if trigger:
        with st.spinner("Finding people like you..."):
            recommendations = get_recommendations(user_id_input, num_recs)
            
            if recommendations is None:
                st.error("⚠️ User ID not found in the training data.")
            elif recommendations.empty:
                st.warning("No new recommendations found for this user.")
            else:
                st.success(f"Top {len(recommendations)} picks for User {user_id_input}:")
                
                # Display in a 2-column grid
                grid_cols = st.columns(2)
                for i, (_, row) in enumerate(recommendations.iterrows()):
                    with grid_cols[i % 2]:
                        st.markdown(f"""
                            <div class="movie-card">
                                <div style="font-size: 1.2rem; font-weight: bold; color: #111;">{row['title']}</div>
                                <div class="genre-text">{row['genres']}</div>
                            </div>
                        """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("Machine Learning Model: K-Nearest Neighbors (Collaborative Filtering)")
st.caption("By Pragati Basnet")