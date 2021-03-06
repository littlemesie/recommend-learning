import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepctr.feature_column import SparseFeat, VarLenSparseFeat



def load_data():
    """加载数据"""
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
    rating_df = pd.read_csv(base_path + 'ml-100k/u.data', sep='\t',
                            names=['user_id', 'movie_id', 'rating', 'timestamp'])
    user_df = pd.read_csv(base_path + 'ml-100k/u.user', sep='|',
                            names=['user_id', 'age', 'gender', 'occupation', 'zip'])

    item_df = pd.read_csv(base_path + 'ml-100k/u.item', sep='|', encoding="ISO-8859-1",
                            names=['movie_id', 'title', 'release_date', 'video_release_date', 'url', 'unknown',
                                   'Action', 'Adventure', 'Animation','Children', 'Comedy', 'Crime', 'Documentary',
                                   'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                                   'Sci-Fi', 'Thriller', 'War', 'Western'])

    movie_types = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                   'Thriller', 'War', 'Western']

    movie_types_map = dict(zip(movie_types, range(len(movie_types))))

    item_df = item_df.drop(columns=['title', 'release_date', 'video_release_date', 'url'])

    item_data = []
    for i, row in item_df.iterrows():
        arr = []
        arr.append(row['movie_id'])
        ar = []
        for key in movie_types:

            if row[key]:
                ar.append(movie_types_map[key])
            else:
                ar.append(0)

        arr.append(ar)
        item_data.append(arr)

    item_data_ = pd.DataFrame(item_data, columns=['movie_id', 'movie_type'])

    data_df = pd.merge(rating_df, user_df, how="left")

    data_df = pd.merge(data_df, item_data_, how="left")

    return data_df



def gen_data_set(data, negsample=0):

    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        neg_list = []
        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list)*negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0, len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set

def gen_model_input(train_set, user_profile, item_profile, seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)

    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}


    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    movie_type_len = np.ones(len(train_uid)) * 20
    mt = item_profile.loc[train_model_input['movie_id']]['movie_type'].values
    movie_type_seq_pad = pad_sequences(mt, maxlen=20, padding='post', truncating='post', value=0)
    train_model_input['movie_type'] = movie_type_seq_pad
    train_model_input['movie_type_len'] = movie_type_len
    return train_model_input, train_label

def test_gen_model_input(test_set, user_profile, seq_max_len):

    train_uid = np.array([line[0] for line in test_set])
    train_seq = [line[1] for line in test_set]
    train_iid = np.array([line[2] for line in test_set])
    train_label = np.array([line[3] for line in test_set])
    train_hist_len = np.array([line[4] for line in test_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)

    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}


    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values
    return train_model_input, train_label
# a = load_data()
# print(a)