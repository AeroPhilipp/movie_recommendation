import pandas as pd
import pickle
from pathlib import Path
from movie_recom.params import *
from sklearn.metrics.pairwise import cosine_similarity
#from movie_recom.ml_logic.preprocessor import create_input_NN
from sklearn.neighbors import NearestNeighbors #new

# # von chris
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow as tf

####################################
###     COSINE SIMILIRATY        ###
####################################

from sklearn.metrics.pairwise import cosine_similarity

def vector_cosine(user_tf_idf_vector):
    filepath_matrix = Path(PARENT_FOLDER_PATH).joinpath('processed_data/vectorized_summaries.pkl')
    tf_idf_matrix = pd.read_pickle(filepath_matrix)
    filepath_title = Path(PARENT_FOLDER_PATH).joinpath("raw_data/movie_title.pkl")
    titles = pd.read_pickle(filepath_title)
    cos_similarities = cosine_similarity(user_tf_idf_vector, tf_idf_matrix).flatten()
    similar_movies = pd.DataFrame({'title': titles.values, 'similarity': cos_similarities})
    similar_movies = similar_movies.sort_values(by='similarity', ascending=False)
    return similar_movies

def predict_NN(prompt_embedded):
    # Load model
    #file_path = os.path.join(PARENT_FOLDER_PATH, "saved_models", "NN_model.pkl") #ross
    file_path = os.path.join(PARENT_FOLDER_PATH, "saved_models", "model_whole_plot.pkl") #philipp
    neural_network = pickle.load(open(file_path, 'rb'))

    # Create input
    X = create_input_NN(prompt_embedded)

    y_pred = neural_network.predict([X.iloc[:, 0:128], X.iloc[:, 128:]])

    return y_pred

def get_also_liked(fav_list: list):
    # get k neigbors for given movie
    filepath_matrix = Path(PARENT_FOLDER_PATH).joinpath('processed_data/movies_user_behaviour_NMF_100c_wtitle.pkl')
    movies_NMF = pd.read_pickle(filepath_matrix)
    # Step 2: Fit a KNN model
    k = 50  # Number of neighbors to consider
    knn_model = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_model.fit(movies_NMF)

    # Step 3: Find the 10 nearest neighbors for a given movie
    # Let's say you want to find the neighbors for movie at index 'movie_index'

    #fav_id_list = [get_id_from_title(title) for title in fav_list]
    fav_mov_NMF_clusters = movies_NMF.loc[fav_list]

    distances, rows = knn_model.kneighbors([fav_mov_NMF_clusters.agg('mean')])

    # The 'indices' variable contains the indices of the 10 nearest neighbors
    # You can use these indices to retrieve the corresponding movies from your dataset
    nearest_neighbors_rows = rows[0].tolist()
    # nearest_neighbors_ids = [movies_keep[l] for l in nearest_neighbors_rows]
    # nearest_neighbors_distances = distances[0]
    title_list = movies_NMF.iloc[nearest_neighbors_rows].index.tolist()
    return title_list
