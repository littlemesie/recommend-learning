import os
import random
import numpy as np
import pandas as pd
import faiss
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss, recall_N
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from utils import metric

def load_data():
    """加载数据"""
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
    rating_df = pd.read_csv(base_path + 'ml-100k/u.data', sep='\t',
                            names=['user_id', 'movie_id', 'rating', 'timestamp'])
    user_df = pd.read_csv(base_path + 'ml-100k/u.user', sep='|',
                            names=['user_id', 'age', 'gender', 'occupation', 'zip'])

    # item_df = pd.read_csv(base_path + 'ml-100k/u.item', sep='|',
    #                         names=['movie_id', 'title', 'release_date', 'video_release_date', 'url', 'unknown',
    #                                'Action', 'Adventure', 'Animation','Children', 'Comedy', 'Crime', 'Documentary',
    #                                'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    #                                'Sci-Fi', 'Thriller', 'War', 'Western'])
    data_df = pd.merge(rating_df, user_df, how="left")

    return data_df



def gen_data_set(data, negsample=0):

    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

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

def gen_model_input(train_set,user_profile,seq_max_len):

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

    return train_model_input, train_label

def train():
    data = load_data()
    item_set = set(data['movie_id'].unique())
    SEQ_LEN = 50

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set = gen_data_set(data, 0)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)

    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    embedding_dim = 16

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
                            SparseFeat("age", feature_max_idx['age'], embedding_dim),
                            SparseFeat("occupation", feature_max_idx['occupation'], embedding_dim),
                            SparseFeat("zip", feature_max_idx['zip'], embedding_dim),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    # 3.Define Model and train

    K.set_learning_phase(True)
    import tensorflow as tf
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5,
                       user_dnn_hidden_units=(64, embedding_dim))

    model.compile(optimizer="adam", loss=sampledsoftmaxloss)  # "binary_crossentropy")

    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=256, epochs=50, verbose=1, validation_split=0.0, )

    # 4. Generate user features for testing and full item features for retrieval
    test_user_model_input = test_model_input
    all_item_model_input = {"movie_id": item_profile['movie_id'].values}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    # user_embs = user_embs[:, i, :]  # i in [0,k_max) if MIND
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    # print(user_embs)
    # print(item_embs)

    # 5. [Optional] ANN search by faiss  and evaluate the result

    test_true_label = {line[0]: [line[2]] for line in test_set}


    index = faiss.IndexFlatIP(embedding_dim)
    # faiss.normalize_L2(item_embs)
    index.add(item_embs)
    # faiss.normalize_L2(user_embs)
    D, I = index.search(np.ascontiguousarray(user_embs), 10)

    recommed_dict = {}
    for i, uid in enumerate(test_user_model_input['user_id']):
        recommed_dict.setdefault(uid, [])
        try:
            pred = [item_profile['movie_id'].values[x] for x in I[i]]
            recommed_dict[uid] = pred
        except:
            print(i)

    test_user_items = dict()
    for ts in test_set:
        if ts[0] not in test_user_items:
            test_user_items[ts[0]] = set(ts[1])
    item_popularity = dict()
    for ts in train_set:
        for item in ts[1]:
            if item in item_popularity:
                item_popularity[item] += 1
            else:
                item_popularity.setdefault(item, 1)

    precision = metric.precision(recommed_dict, test_user_items)
    recall = metric.recall(recommed_dict, test_user_items)
    coverage = metric.coverage(recommed_dict, item_set)
    popularity = metric.popularity(item_popularity, recommed_dict)

    print("precision:{:.4f}, recall:{:.4f}, coverage:{:.4f}, popularity:{:.4f}".format(precision, recall, coverage,
                                                                                       popularity))

train()

# precision:0.3242, recall:0.0309, coverage:0.1177, popularity:9.3637