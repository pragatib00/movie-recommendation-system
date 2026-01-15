## Movie Recommendation System
A personalized movie recommendation system built using Collaborative Filtering on the MovieLens dataset and deployed as an interactive Streamlit web application.

## Overview
This project recommends movies to users based on the preferences of similar users. It uses a KNN-based user-to-user collaborative filtering approach and evaluates performance using Precision@K, Recall@K, and RMSE.

## How it Works
- Clean and preprocess user–movie ratings
- Build a sparse user-movie matrix
- Find similar users using cosine similarity (KNN)
- Predict unseen movies using weighted neighbor ratings
- Recommend top-N movies to users

## Live App
- https://movie-recommender-system0001.streamlit.app/

## Evaluation
The model was evaluated using leave-one-out validation:

- Precision@10
- Recall@10
- RMSE

These metrics measure recommendation quality and prediction accuracy.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, SciPy
- Streamlit
- Git & GitHub

## Run Locally
git clone https://github.com/pragatib00/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
streamlit run app.py

## Author
Pragati Basnet
BSc. CSIT | Data Science & Machine Learning
GitHub: https://github.com/pragatib00
LinkedIn: https://www.linkedin.com/in/pragati-basnet-1595492a7/