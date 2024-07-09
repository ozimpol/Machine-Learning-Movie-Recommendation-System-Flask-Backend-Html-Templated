import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

movies_df = pd.read_csv('movies.csv', usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
ratings_df = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating', 'timestamp'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

movies_merged_df = movies_df.merge(ratings_df, on='movieId')

movies_merged_df = movies_merged_df.dropna(axis=0, subset=['title'])

popularity_threshold = 50
movies_rating_count = movies_merged_df.groupby('title')['rating'].count().sort_values(ascending=False)
popular_movies = movies_rating_count[movies_rating_count >= popularity_threshold].index

popular_movies_df = movies_merged_df[movies_merged_df['title'].isin(popular_movies)]

movie_features_df = popular_movies_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
movie_features_csr = csr_matrix(movie_features_df.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_features_csr)

train_data, test_data = train_test_split(popular_movies_df, test_size=0.2, random_state=42)

svd_model = TruncatedSVD(n_components=50, random_state=42)
svd_train_matrix = svd_model.fit_transform(movie_features_df)
svd_test_matrix = svd_model.transform(movie_features_df)

nmf_model = NMF(n_components=50, init='random', max_iter=5000, random_state=42)
nmf_train_matrix = nmf_model.fit_transform(movie_features_df)
nmf_test_matrix = nmf_model.transform(movie_features_df)

knn_rmse_list = []
for movie_name in movie_features_df.index:
    query_index = movie_features_df.index.get_loc(movie_name)
    distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)
    knn_rmse_list.append(mean_squared_error(distances.flatten(), indices.flatten()))

svd_rmse = mean_squared_error(svd_train_matrix.flatten(), svd_test_matrix.flatten())

nmf_rmse = mean_squared_error(nmf_train_matrix.flatten(), nmf_test_matrix.flatten())

print(f"KNN RMSE: {sum(knn_rmse_list) / len(knn_rmse_list)}")
print(f"SVD RMSE: {svd_rmse}")
print(f"NMF RMSE: {nmf_rmse}")

knn_param_grid = {'n_neighbors': [5, 10, 15],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_search_knn = GridSearchCV(KNeighborsRegressor(metric='cosine'), knn_param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search_knn.fit(movie_features_csr, movie_features_df.index)
best_knn_params = grid_search_knn.best_params_

svd_param_grid = {'n_components': [10, 20, 30, 40, 50]}
grid_search_svd = GridSearchCV(TruncatedSVD(random_state=42), svd_param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search_svd.fit(movie_features_df)
best_svd_params = grid_search_svd.best_params_

nmf_param_grid = {'n_components': [10, 20, 30, 40, 50],
                  'max_iter': [1000, 2000, 3000, 4000, 5000]}
grid_search_nmf = GridSearchCV(NMF(init='random', random_state=42), nmf_param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search_nmf.fit(movie_features_df)
best_nmf_params = grid_search_nmf.best_params_

print(f"Best KNN Parameters: {best_knn_params}")
print(f"Best SVD Parameters: {best_svd_params}")
print(f"Best NMF Parameters: {best_nmf_params}")
