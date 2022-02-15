# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/2/15 下午2:55
@summary: 基于movie lens数据集实现Swing算法
"""
import random
from itertools import combinations

class SwingModel:
    def __init__(self):
        self.alpha = 0.5
        self.top_k = 20

    def get_dataset(self, filename, pivot=0.7):
        """load data"""
        train_u_items = dict()
        train_i_users = dict()
        test_u_items = dict()
        test_i_users = dict()
        for line in self.load_file(filename):
            user_id, item_id, rating, timestamp = line.split('\t')
            if(random.random() < pivot):
                train_u_items.setdefault(user_id, set())
                train_i_users.setdefault(item_id, set())

                train_u_items[user_id].add(item_id)
                train_i_users[item_id].add(user_id)
            else:
                test_u_items.setdefault(user_id, set())
                test_i_users.setdefault(item_id, set())
                test_u_items[user_id].add(item_id)
                test_i_users[item_id].add(user_id)
        return train_u_items, train_i_users, test_u_items, test_i_users

    def load_file(self, filename):
        """读文件，返回文件的每一行"""
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)

    def cal_similarity(self, u_items, i_users):
        """计算相似"""
        item_pairs = list(combinations(i_users.keys(), 2))
        print("item pairs length：{}".format(len(item_pairs)))
        item_sim_dict = dict()
        cnt = 0
        for (i, j) in item_pairs:
            cnt += 1
            # print(cnt)
            user_pairs = list(combinations(i_users[i] & i_users[j], 2))
            result = 0.0
            for (u, v) in user_pairs:
                result += 1 / (self.alpha + list(u_items[u] & u_items[v]).__len__())

            item_sim_dict.setdefault(i, dict())
            item_sim_dict[i][j] = result
            # print(item_sim_dict[i][j])

        # 排序
        new_item_sim_dict = dict()
        for item, sim_items in item_sim_dict.items():
            new_item_sim_dict[item] = sorted(sim_items.items(), key=lambda k: k[1], reverse=True)

        return new_item_sim_dict
