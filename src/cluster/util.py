import pandas as pd
import numpy as np
import random

user_info_path = '../../data/ml-100k/u.user'
item_info_path = '../../data/ml-100k/u.item'
base_path = '../../data/ml-100k/ua.base'
test_path = '../../data/ml-100k/ua.test'

def load_data():
    user_header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_info = pd.read_csv(user_info_path, sep='|', names=user_header)
    user_info['age'] = pd.cut(user_info['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                              labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '70-80', '80-90', '90-100', '100-150'])
    user_id = user_info['user_id']
    user_info = user_info.drop(columns=['user_id', 'zip_code'])
    user_info = pd.get_dummies(user_info, columns=['age', 'gender', 'occupation'])
    return np.array(user_id), np.array(user_info)


def read_rating_data():
    """"""
    data = dict()
    user_set = set()
    item_set = set()
    header = ['user_id', 'item_id', 'rating', 'time']
    base_info = pd.read_csv(base_path, sep='\t', names=header)
    base_info = base_info.drop(columns=['rating', 'time'])
    for base in np.array(base_info):
        user_set.add(base[0])
        item_set.add(base[1])
        if base[0] not in data:
            data.setdefault(base[0], set())
            data[base[0]].add(base[1])
        else:
            data[base[0]].add(base[1])
    return data, user_set, item_set
# load_data()
read_rating_data()