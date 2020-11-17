import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def load_data():
    """加载数据"""
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"

    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    user = pd.read_csv(base_path + 'ml-1m/users.dat', sep='::', header=None, names=unames)

    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(base_path + 'ml-1m/ratings.dat', sep='::', header=None, names=rnames)

    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(base_path + 'ml-1m/movies.dat', sep='::', header=None, names=mnames)

    data = pd.merge(pd.merge(ratings, movies), user)  # .iloc[:10000]

    return data

def gen_data_set_sdm(data, seq_short_len=5, seq_prefer_len=50):

    data.sort_values("timestamp", inplace=True)
    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        genres_list = hist['genres'].tolist()
        rating_list = hist['rating'].tolist()
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            if i <= seq_short_len and i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], [0]*seq_prefer_len, pos_list[i], 1, len(hist[::-1]), 0,
                                  rating_list[i], genres_hist[::-1], [0]*seq_prefer_len))
            elif i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1][:seq_short_len], hist[::-1][seq_short_len:], pos_list[i], 1, seq_short_len,
                len(hist[::-1])-seq_short_len, rating_list[i], genres_hist[::-1][:seq_short_len], genres_hist[::-1][seq_short_len:]))
            elif i <= seq_short_len and i == len(pos_list) - 1:
                test_set.append((reviewerID, hist[::-1], [0] * seq_prefer_len, pos_list[i], 1, len(hist[::-1]), 0,
                                  rating_list[i], genres_hist[::-1], [0]*seq_prefer_len))
            else:
                test_set.append((reviewerID, hist[::-1][:seq_short_len], hist[::-1][seq_short_len:], pos_list[i], 1, seq_short_len,
                len(hist[::-1])-seq_short_len, rating_list[i], genres_hist[::-1][:seq_short_len], genres_hist[::-1][seq_short_len:]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set

def gen_model_input_sdm(train_set, user_profile, seq_short_len, seq_prefer_len):

    train_uid = np.array([line[0] for line in train_set])
    short_train_seq = [line[1] for line in train_set]
    prefer_train_seq = [line[2] for line in train_set]
    train_iid = np.array([line[3] for line in train_set])
    train_label = np.array([line[4] for line in train_set])
    train_short_len = np.array([line[5] for line in train_set])
    train_prefer_len = np.array([line[6] for line in train_set])
    short_train_seq_genres = np.array([line[8] for line in train_set])
    prefer_train_seq_genres = np.array([line[9] for line in train_set])

    train_short_item_pad = pad_sequences(short_train_seq, maxlen=seq_short_len, padding='post', truncating='post',
                                        value=0)
    train_prefer_item_pad = pad_sequences(prefer_train_seq, maxlen=seq_prefer_len, padding='post', truncating='post',
                                         value=0)
    train_short_genres_pad = pad_sequences(short_train_seq_genres, maxlen=seq_short_len, padding='post', truncating='post',
                                        value=0)
    train_prefer_genres_pad = pad_sequences(prefer_train_seq_genres, maxlen=seq_prefer_len, padding='post', truncating='post',
                                        value=0)

    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "short_movie_id": train_short_item_pad,
        "prefer_movie_id": train_prefer_item_pad, "prefer_sess_length": train_prefer_len, "short_sess_length":
        train_short_len, 'short_genres': train_short_genres_pad, 'prefer_genres': train_prefer_genres_pad}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label