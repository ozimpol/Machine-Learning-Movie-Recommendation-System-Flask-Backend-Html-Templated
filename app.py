from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from itertools import zip_longest 
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

movies_df = pd.read_csv('movies.csv', usecols=['movieId', 'title', 'genres'], dtype={'movieId': 'int32', 'title': 'str', 'genres': 'str'})
ratings_df = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating', 'timestamp'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
movies_merged_df = movies_df.merge(ratings_df, on='movieId')
movies_merged_df = movies_merged_df.dropna(axis=0, subset=['title'])

popularity_threshold = 50
movies_rating_count = movies_merged_df.groupby('title')['rating'].count().sort_values(ascending=False)
popular_movies = movies_rating_count[movies_rating_count >= popularity_threshold].index
popular_movies_df = movies_merged_df[movies_merged_df['title'].isin(popular_movies)]
movie_features_df = popular_movies_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

model_nmf = NMF(n_components=10, max_iter=1000, init='random', random_state=42)
movie_features_nmf = model_nmf.fit_transform(movie_features_df)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/movies')
def movies():
    page = request.args.get('page', 1, type=int)
    per_page = 14
    total_movies = len(movies_df)
    total_pages = (total_movies + per_page - 1) // per_page  

    start = (page - 1) * per_page
    end = start + per_page
    movies_page = movies_df.iloc[start:end]

    images = []
    for title in movies_page['title']:
        search_url = f'https://search.yahoo.com/search?p={title.replace(" ", "+")}+movie+poster'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            image_tag = soup.find('img')
            if image_tag:
                images.append(image_tag.get('src'))
            else:
                images.append(None)
        else:
            images.append(None)

    movies_with_images = zip_longest(movies_page.values.tolist(), images)
    return render_template('movies.html', movies_with_images=movies_with_images, page=page, total_pages=total_pages)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    term = request.args.get('term')
    suggestions = [title for title in movie_features_df.index if term.lower() in title.lower()]
    return jsonify(suggestions)

@app.route('/random-recommendation')
def random_recommendation():
    random_movies = movies_df.sample(n=6)['title'].tolist()
    genres = [movies_df[movies_df['title'] == title]['genres'].values[0] for title in random_movies]

    images = []
    for title in random_movies:
        search_url = f'https://search.yahoo.com/search?p={title.replace(" ", "+")}+movie+poster'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            image_tag = soup.find('img')
            if image_tag:
                images.append(image_tag.get('src'))
            else:
                images.append(None)
        else:
            images.append(None)

    zipped_data = zip_longest(random_movies, genres, images)
    return render_template('recommendation.html', movie_name='you', zipped_data=zipped_data)


@app.route('/recommendation', methods=['POST'])
def recommendation():
    movie_name = request.form['movie_name']
    if movie_name not in movie_features_df.index:
        return f"Couldn't find a movie named '{movie_name}'. Please try again."

    query_index = movie_features_df.index.get_loc(movie_name)
    query_nmf = model_nmf.transform(movie_features_df.iloc[query_index, :].values.reshape(1, -1))
    similarities = cosine_similarity(query_nmf, movie_features_nmf).flatten()
    similar_indices = similarities.argsort()[:-8:-1]
    recommendations = [movie_features_df.index[i] for i in similar_indices[1:]]
    genres = [movies_df[movies_df['title'] == title]['genres'].values[0] for title in recommendations]

    images = []
    for title in recommendations:
        search_url = f'https://search.yahoo.com/search?p={title.replace(" ", "+")}+movie+poster'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            image_tag = soup.find('img')
            if image_tag:
                images.append(image_tag.get('src'))
            else:
                images.append(None)
        else:
            images.append(None)

    zipped_data = zip_longest(recommendations, genres, images)
    return render_template('recommendation.html', movie_name=movie_name, zipped_data=zipped_data)

if __name__ == '__main__':
    app.run(debug=True)
