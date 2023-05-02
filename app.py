from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
import json

app = Flask(__name__)

# load the datasets
main_df = pd.read_csv('main.csv')
similarity_matrix = np.loadtxt('similarity_matrix.csv', delimiter=',')

# TMDb API configuration
api_key = '5bda0a39e6f7abce03804df5779fc584'

def get_poster(movie_id, api_key):
    # construct the API request URL
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'

    # send the GET request
    response = requests.get(url)

    # extract the poster path from the JSON response
    poster_path = json.loads(response.content)['poster_path']

    # construct the full poster URL
    poster_url = f'https://image.tmdb.org/t/p/w500{poster_path}'

    return poster_url

# create a recommendation function
def recommend_movies(movie_name, similarity_matrix=similarity_matrix, main_df=main_df):
    # get index of the movie
    idx = main_df[main_df['title'] == movie_name].index[0]

    # get the pairwise similarity scores for all movies with the movie
    similarity_scores = list(enumerate(similarity_matrix[idx]))

    # sort the movies based on the similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # get the top 15 most similar movies
    top_15 = similarity_scores[1:16]

    # get the movie indices
    movie_indices = [i[0] for i in top_15]

    # return the top 15 similar movies from the main_df
    return main_df.iloc[movie_indices]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommendation', methods=['POST'])
def recommendation():
    movie_name = request.form['movie_name']
    movie_info = main_df[main_df['title'] == movie_name].iloc[0]
    recommended_movies = recommend_movies(movie_name)
    recommended_movies.reset_index(inplace=True, drop=True)
    recommended_movies['poster'] = recommended_movies.apply(lambda x: get_poster(x['tmdbId'], api_key), axis=1)
    return render_template('recommendation.html', movie_info=movie_info, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
