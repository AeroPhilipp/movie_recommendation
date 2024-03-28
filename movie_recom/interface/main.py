import numpy as np
import pandas as pd

from pathlib import Path

from movie_recom.params import *
from movie_recom.ml_logic.encoders import tf_vectorize # bert_encode
from movie_recom.ml_logic.model import vector_cosine, get_also_liked #, predict_NN
#from movie_recom.ml_logic.preprocessor import create_output_NN

# def embed_prompt(prompt: str) -> pd.DataFrame:
#     """
#     embed the prompt
#     """
#     #put it into a dataframe for NN
#     prompt_embedded = bert_encode(prompt)
#     return prompt_embedded

# def merge_prompt_with_favorites(prompt_bert: pd.DataFrame, favs: list, weight_fav: float) -> pd.DataFrame:
#     # get the embedded data
#     # Load titles
#     filepath_title = Path(PARENT_FOLDER_PATH).joinpath("raw_data/movie_title.pkl")
#     titles = pd.read_pickle(filepath_title)

#     # Load embedded plots
#     filepath_plot = Path(PARENT_FOLDER_PATH).joinpath("processed_data/embeddings_plot.npy")
#     plot_embedded = pd.DataFrame(np.load(filepath_plot), columns=[str(i) for i in range(0,128,1)], index = titles.values)

#     df_filtered = plot_embedded[plot_embedded.index.isin(favs)] # embedded dataframe with just the favorites
#     mean_df = df_filtered.mean(axis=0).to_frame().T # get the mean of the dataframe, keep it as dataframe
#     mean_df_weighted = mean_df * weight_fav # weight the fav dataframe

#     prompt_bert_weighted = prompt_bert * (1 - weight_fav) # weight the prompt dataframe
#     series = prompt_bert_weighted.iloc[0,:] # convert the prompt dataframe to a series
#     mean_df_weighted.loc['prompt'] = series.to_list() # add the prompt to the dataframe (concat didnt work well)
#     sum_df = df_filtered.sum(axis=0).to_frame().T # get the mean of the dataframe, keep it as dataframe
#     sum_df.index = ['prompt'] # set the index to 'prompt'

#     return sum_df

def find_recommendation_vector(text):
    # Vectorise user input
    vectorized_prompt = tf_vectorize(text)
    #return dataframe with movie recommendations and similarity score
    return vector_cosine(vectorized_prompt)


def predict_text_similar_movies(prompt: str = 'drug addict getting his life back on track', fav_list: list=[], weight_n: float=1.0, weight_fav: float=0.5) -> list:
    '''
    get the prompt and recommend movies based on it
    '''

    # recommend with cosine similarity
    recom_list =  find_recommendation_vector(prompt)

    combined = recom_list
    combined['sum'] = 6 * combined['similarity']

    recommendations = combined.sort_values(by='sum', ascending=False)[0:50]
    # print for testing
    print(combined.sort_values(by='sum', ascending=False)[0:10])
    return recommendations['title'].tolist()

def predict_also_liked_movies(prompt: str = 'drug addict getting his life back on track', fav_list: list=[], weight_n: float=1.0, weight_fav: float=0.5) -> list:
    '''
    get the prompt and recommend movies based on it
    '''
    # fav_list = ['Space Jam',
    # 'Bed of Roses',
    # '101 Dalmatians',
    # 'Muppet Treasure Island',
    # 'Sudden Death',
    # 'Father of the Bride Part II',
    # 'Eye for an Eye',
    # 'Sgt. Bilko',
    # 'Striptease',
    # 'Renaissance Man']
    # recommend with NMF
    recom_list =  get_also_liked(fav_list)

    print(recom_list[:10])
    return recom_list

if __name__ == '__main__':
    pass
