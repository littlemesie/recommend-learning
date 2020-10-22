import pandas as pd
import numpy as np
import random
import tensorflow as tf

user_info_path = '../../data/ml-100k/u.user'
item_info_path = '../../data/ml-100k/u.item'
base_path = '../../data/ml-100k/ua.base'
test_path = '../../data/ml-100k/ua.test'


def user_item_data():
    user_header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_info = pd.read_csv(user_info_path, sep='|', names=user_header)
    user_info['age'] = pd.cut(user_info['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                              labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
    user_id = user_info[['user_id']]
    user_info = user_info.drop(columns=['zip_code'])
    user_info = pd.get_dummies(user_info, columns=['age', 'gender', 'occupation'])
    # user_info = pd.concat([user_id, user_info], axis=1)
    user_dict = {}
    for index, row in user_info.iterrows():
        user_dict.setdefault(row['user_id'], list(user_info.iloc[index, 1:].values))
    user_feature_size = user_info.shape[1] - 1
    # print(user_id)
    item_header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
                   'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    item_info = pd.read_csv(item_info_path, sep='|', names=item_header, encoding="ISO-8859-1")
    item_info = item_info.drop(columns=['title', 'video_release_date', 'IMDb_URL'])
    item_id = item_info[['item_id']]
    item_info = pd.get_dummies(item_info, columns=['release_date', 'unknown', 'Action', 'Adventure',
                   'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    # item_info = pd.concat([item_id, item_info], axis=1)
    item_dict = {}
    for index, row in item_info.iterrows():
        item_dict.setdefault(row['item_id'], list(item_info.iloc[index, 1:].values))

    user_ids = user_id['user_id'].values
    item_ids = item_id['item_id'].values
    item_feature_size = item_info.shape[1] - 1
    return user_ids, item_ids, user_dict, item_dict, user_feature_size, item_feature_size

def shuffle_batch(user_ids, item_ids, feature_user, feature_item, labels):
    length = len(user_ids)
    index = [i for i in range(length)]
    random.shuffle(index)
    user_ids_shuffle = [user_ids[i] for i in index]
    item_ids_shuffle = [item_ids[i] for i in index]
    feature_user_shuffle = [feature_user[i] for i in index]
    feature_item_shuffle = [feature_item[i] for i in index]
    labels_shuffle = [labels[i] for i in index]
    return user_ids_shuffle, item_ids_shuffle, feature_user_shuffle, feature_item_shuffle, labels_shuffle

def add_negative(data, user_dict, item_dict, user_negative, negative_num, batch_size, is_training):
    """"""
    user_ids, item_ids, feature_user, feature_item, labels_add = [], [], [], [], []

    for i, row in data.iterrows():

        user = row['user_id']
        item = row['item_id']
        user_ids.append(user)
        item_ids.append(item)
        feature_user.append(user_dict[user])
        feature_item.append(item_dict[item])
        labels_add.append(1)

        neg_samples = np.random.choice(user_negative[user], size=negative_num, replace=False).tolist()
        # print(neg_samples)
        if is_training:
            for k in neg_samples:
                user_ids.append(user)
                item_ids.append(k)
                feature_user.append(user_dict[user])
                feature_item.append(item_dict[k])
                labels_add.append(0)

        else:
            for k in neg_samples:
                user_ids.append(user)
                item_ids.append(k)
                feature_user.append(user_dict[user])
                feature_item.append(item_dict[k])
                labels_add.append(k)

    data_dict = dict([('user_ids', user_ids), ('item_ids', item_ids),  ('feature_user', feature_user), ('feature_item', feature_item), ('labels', labels_add)])
    # print(data_dict)
    dataset = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset = dataset.shuffle(100000).batch(batch_size)
    return dataset, data_dict


def load_data():
    user_ids, item_ids, user_dict, item_dict, user_feature_size, item_feature_size = user_item_data()


    header = ['user_id', 'item_id', 'rating', 'timestamp']
    base = pd.read_csv(base_path, sep='\t', names=header)
    base = base.drop(columns=['timestamp'])

    user_bought = {}
    for i, row in base.iterrows():
        u = row['user_id']
        t = row['item_id']
        user_bought.setdefault(u, [])
        user_bought[u].append(t)

    # 没有购买的item
    user_negative = {}
    for key in user_bought:
        user_negative[key] = list(set(item_ids) - set(user_bought[key]))

    # test = pd.read_csv(test_path, sep='\t', names=header)
    # test = test.drop(columns=['timestamp'])
    return base, user_ids, item_ids, user_dict, item_dict, user_bought, user_negative, user_feature_size, item_feature_size


# base, user_ids, item_ids, user_dict, item_dict, user_bought, user_negative, user_feature_size, item_feature_size = load_data()
#
# train_data = add_negative(base, user_dict, item_dict, user_negative, 4, 128, is_training=True)
# with tf.Session() as sess:
#     iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
