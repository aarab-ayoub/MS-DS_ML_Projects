import pandas as pd
import numpy as np


file_path = "ml-latest-small/ratings.csv"
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    users_list = df['userId'].unique() 
    movies_list = df['movieId'].unique()
    
    # print(df.head())
    # print(users_list)
    # print(len(users_list))
    # print(len(movies_list))

    users_obj = {user : i for i , user in enumerate(users_list)}
    movies_obj = {movie: i for i, movie in enumerate(movies_list)}
    
    utility_matrix = np.zeros((len(users_list), len(movies_list)))
    for row in df.itertuples():
        utility_matrix[users_obj[row.userId], movies_obj[row.movieId]] = row.rating
    return utility_matrix


def get_similarities(utility):
    pass

if __name__ == "__main__":
    ratings_matrix = load_data(file_path)
