# coding=utf-8
import pandas as pd
import numpy as np
import tensorflow as tf

# 数据预处理
def load_data():
    users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
    users = pd.read_table('../data/users.dat', sep='::', header=None, names=users_title, engine='python')
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('../data/movies.dat', sep='::', header=None, names=movies_title, engine='python')
    ratings_title = ['UserID','MovieID', 'Rating', 'timestamps']
    ratings = pd.read_table('../data/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')

    print(ratings.head())