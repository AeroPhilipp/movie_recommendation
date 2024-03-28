import numpy as np
import pandas as pd
from pathlib import Path
from movie_recom.params import *

# def shorten_synopsis(max_len: int, df: pd.DataFrame) -> pd.DataFrame:
#     '''
#     removes movies with plot_synopsis that are longer than max_len words
#     To Do: shorten long synopses to max_len words to keep all movies
#     '''
#     df['plot_word_count'] = df['plot_synopsis'].apply(lambda x: len(x.split()))
#     df_output = df[df['plot_word_count'] < max_len].copy()
#     df_output = df_output.drop(columns = 'plot_word_count', axis = 1)
#     return df_output

# def create_input_NN(prompt_embedded):


#     # Load titles
#     filepath_title = Path(PARENT_FOLDER_PATH).joinpath("raw_data/movie_title.pkl")
#     titles = pd.read_pickle(filepath_title)

#     # Load embedded plots
#     filepath_plot = Path(PARENT_FOLDER_PATH).joinpath("processed_data/embeddings_plot.npy")
#     plot_embedded = pd.DataFrame(np.load(filepath_plot), columns=[str(i) for i in range(0,128,1)], index = titles.values)

#     new_prompts = pd.DataFrame(np.repeat(prompt_embedded.values, plot_embedded.shape[0], axis=0), columns=[str(i) + "_" for i in range(0,128,1)], index = titles.values)

#     new_data = pd.merge(left=new_prompts, right=plot_embedded, left_index=True, right_index=True, how='left')

#     return new_data

# def create_output_NN(y_pred):
#     # Load titles
#     filepath_title = Path(PARENT_FOLDER_PATH).joinpath("raw_data/movie_title.pkl")
#     titles = pd.read_pickle(filepath_title)

#     recommendations = pd.DataFrame(y_pred, columns=['rating'], index = titles.values)

#     return recommendations
