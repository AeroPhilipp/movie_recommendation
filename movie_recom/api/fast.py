import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from movie_recom.interface.main import embed_prompt, merge_prompt_with_favorites, predict_text_similar_movies, predict_also_liked_movies
from pathlib import Path
from movie_recom.params import *

app = FastAPI()
#load pickle model

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
        prompt: str,
        fav_list,
        weight_n: float,
        weight_fav: float
    ):
    """
    gives a list of n_recom recommendations based on the prompt
    """
    # generate output list

    movie_list = predict_text_similar_movies(prompt, fav_list.split("_"), float(weight_n), float(weight_fav))
    # load list of titles

    return {"Our recommendation is": movie_list}

@app.get("/alsoliked")
def alsoliked(
        prompt: str,
        fav_list,
        weight_n: float,
        weight_fav: float
    ):
    """
    gives a list of n_recom recommendations based on favourits
    """
    # generate output list

    movie_list = predict_also_liked_movies(prompt, fav_list.split("_"), float(weight_n), float(weight_fav))
    # load list of titles

    return {"Our recommendation is": movie_list}


@app.get("/")
def root():
    return {"message": "Welcome to the MovieRecommendation API"}

# # Define a root `/` endpoint
# @app.get('/')
# def index():
#     return {'ok': True}
