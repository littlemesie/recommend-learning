import numpy as np
import pandas as pd
import scipy.sparse as sp
import os

DATA_DIR = '../../data/ml-100k/u.data'
DATA_PATH = 'data/'
COLUMN_NAMES = ['user', 'item']
ITEM_CLIP = 300

class Dataset:

    def __init__(self):
        '''
        Constructor
        '''
        self.load_data()


    def map_index(self, item_set):

        i = 0
        items_map = {}
        for item in item_set:
            items_map[item] = i
            i += 1
        return items_map

    def load_data(self):
        full_data = pd.read_csv(DATA_DIR, sep='\t', header=None, names=COLUMN_NAMES, usecols=[0, 1],
                                dtype={0: np.int32, 1: np.int32}, engine='python')

        full_data.user = full_data['user'] - 1
        user_set = set(full_data['user'].unique())
        item_set = set(full_data['item'].unique())

        self.user_size = len(user_set)
        self.item_size = len(item_set)

        # self.user_map = self.map_index(user_set)
        # print(self.user_map)
        # full_data['user'] = full_data['user'].map(lambda x: self.user_map[x])

        self.item_map = self.map_index(item_set)
        full_data['item'] = full_data['item'].map(lambda x: self.item_map[x])

        index_item_set = set(full_data.item.unique())

        # {user:[items]}
        user_bought = {}
        for i in range(len(full_data)):
            u = full_data['user'][i]
            t = full_data['item'][i]
            if u not in user_bought:
                user_bought[u] = []
            user_bought[u].append(t)

        # 没有购买的item
        user_negative = {}
        for key in user_bought:
            user_negative[key] = list(index_item_set - set(user_bought[key]))

        # 划分train-test
        user_length = full_data.groupby('user').size().tolist()
        split_train_test = []
        for i in range(len(user_set)):
            for _ in range(user_length[i] - 1):
                split_train_test.append('train')
            split_train_test.append('test')

        full_data['split'] = split_train_test

        train_data = full_data[full_data['split'] == 'train'].reset_index(drop=True)
        test_data = full_data[full_data['split'] == 'test'].reset_index(drop=True)
        del train_data['split']
        del test_data['split']

        labels = np.ones(len(train_data), dtype=np.int32)

        train_features = train_data
        self.train_labels = labels.tolist()

        test_features = test_data

        self.test_labels = test_data['item'].tolist()
        train_data.sort_values("user", inplace=True)
        # print(train_data.shape)
        # test_data.sort_values("user", inplace=True)

        self.test_list, self.negative_list = self.get_test_list(test_data, user_negative)
        self.train_list = self.get_train_list(train_data)
        # batch_choice = 'user' 就不需要
        self.train_matrix = self.get_train_matrix(train_data)


    def get_test_list(self, test_data, user_negative):
        test_list = []
        negative_list = []
        for i in range(len(test_data)):
            negative = user_negative[test_data['user'][i]]
            test_list.append([test_data['user'][i], test_data['item'][i]])
            negative_list.append(negative if len(negative) < 100 else negative[:100])
        return test_list, negative_list

    def get_train_list(self, train_data):
        train_dict = {}
        for i in range(len(train_data)):

            user, item = int(train_data['user'][i]), int(train_data['item'][i])
            train_dict.setdefault(user, [])
            train_dict[user].append(item)

        return list(train_dict.values())

    def get_train_matrix(self, train_data):

        mat = sp.dok_matrix((self.user_size+1, self.item_size+1), dtype=np.float32)
        for i in range(len(train_data)):
            user, item = int(train_data['user'][i]), int(train_data['item'][i])
            mat[user, item] = 1.0
        return mat

if __name__ == '__main__':
    """"""
    d = Dataset()
