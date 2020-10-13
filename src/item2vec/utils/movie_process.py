# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import OrderedDict

user_info_path = '../../../data/ml-100k/u.user'
# item_info_path = '../../../data/ml-100k/u.item'
item_info_path = '../../data/ml-100k/u.item'
base_path = '../../../data/ml-100k/ua.base'
test_path = '../../../data/ml-100k/ua.test'

class MovieProcessor(object):

    def __init__(self):
        self.item_id, self.item_data = self.load_data()
        self.item_dict = self.get_item_dict()
        self.item_list = list(self.item_dict.keys())
        self.item_counts = list(self.item_dict.values())
        self.item_index_data = self.map_to_ix()

    def load_data(self):
        item_header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                       'Adventure',
                       'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                       'Horror',
                       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        item_info = pd.read_csv(item_info_path, sep='|', names=item_header, encoding="ISO-8859-1")
        item_id = item_info[['item_id']]
        item_info = item_info.drop(columns=['title', 'video_release_date', 'IMDb_URL'])
        year = lambda x: str(x).split('-')[-1]
        item_info['movie_year'] = item_info['release_date'].apply(year)
        item_info = item_info.drop(columns=['release_date', 'item_id'])
        item_info = pd.get_dummies(item_info, columns=['movie_year'])
        return item_id, item_info

    def get_item_dict(self):
        total = (self.item_data != 0).sum()
        d = {}
        for index in total.index:
            if index == 'item_id':
                continue
            d[index] = total[index]

        return OrderedDict(sorted(d.items(), reverse=True, key=lambda v: v[1]))

    def map_to_ix(self):
        headers = list(self.item_data.columns.values)
        item_index_data = []
        item_to_ix = dict(zip(self.item_list, range(len(self.item_list))))
        self.item_to_ix = item_to_ix

        for index, row in self.item_data.iterrows():
            item_index_data.append([item_to_ix[header] for header in headers if row[header]])

        return item_index_data

if __name__ == '__main__':
    # load_data()
    m = MovieProcessor()
    print(m.map_to_ix())