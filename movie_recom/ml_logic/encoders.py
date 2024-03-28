import pandas as pd
# import embedding with bert
#from transformers import TFAutoModel #,AutoTokenizer,
# import embedding with tfidf
from movie_recom.params import *
import os
import pickle

# def bert_encode(prompt: str) -> pd.DataFrame:
#     '''
#     For NN:
#     convert the prompt to a vector using bert-tiny
#     input: prompt: str
#     output: df_encoded: pd.DataFrame (format compatible for NN)
#     '''

#     # instatiate the model
#     model_name = "prajjwal1/bert-tiny"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = TFAutoModel.from_pretrained(model_name, from_pt = True)

#     # Tokenize the text data
#     token_tensor = tokenizer(prompt, max_length=500, truncation=True, return_tensors="tf")
#     # Create input tensors
#     input_tensor = token_tensor['input_ids']
#     # Generate embeddings
#     prediction = model.predict(input_tensor)
#     # Process the embeddings as np
#     embedded_data = prediction.last_hidden_state[:, 0, :]
#     embedded_data = pd.DataFrame(embedded_data, columns=[str(i) + "_" for i in range(0,128,1)])

#     return embedded_data

def tf_vectorize(text):
    file_path = os.path.join(PARENT_FOLDER_PATH, "saved_models", 'tf_idf_vectorizer.pkl')
    tf_idf_vectorizer = pickle.load(open(file_path, 'rb'))
    user_tf_idf_vector = tf_idf_vectorizer.transform([text])
    return user_tf_idf_vector
