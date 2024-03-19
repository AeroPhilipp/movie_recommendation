import pandas as pd
import pickle
from pathlib import Path
from movie_recom.params import *
from sklearn.metrics.pairwise import cosine_similarity
from movie_recom.ml_logic.preprocessor import create_input_NN

# von chris
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

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
