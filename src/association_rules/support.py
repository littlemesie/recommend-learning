"""
关联规则
"""
import random
import math
import os
import collections


class AssociationRules:
    # 初始化参数
    def __init__(self, support=0.4, K=3):
        # 找到相似的20部电影，为目标用户推荐10部电影
        self.n_sim_movie = 20
        self.n_rec_movie = 10

        self.train_set = {}
        self.item_set = {}
        self.move_cut_set = []
        self.support = support
        self.K = K

    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)


    def get_dataset(self, filename):
        """
        读文件得到“用户-电影”数据
        :param filename:
        :return:
        """
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split('\t')
            self.train_set.setdefault(user, [])
            self.train_set[user].append(movie)
            if movie in self.item_set:
                self.item_set[movie] += 1
            else:
                self.item_set.setdefault(movie, 1)

    def cut_tree(self):
        users_count = len(self.train_set)
        gt_support = {}
        lt_support = {}
        for movie, num in self.item_set.items():
            if (num * 1.0 / users_count) >= self.support:
                gt_support.setdefault(movie, num)
            else:
                lt_support.setdefault(movie, num)
        return gt_support, lt_support

    def get_combinations(self, data):
        """ 计算k个项的组合项集，利用递归的思想"""
        n = len(data)
        result = []
        for i in range(n - self.K + 1):
            if self.K > 1:
                newL = data[i + 1:]
                comb = self.get_combinations(newL)
                for item in comb:
                    item.insert(0, data[i])
                    result.append(item)
            else:
                result.append([data[i]])

        return result

    def move_cut(self):
        """ 获取k个元素的组合项集，除去k-1不符合支持度的子集（这个值通过剪枝得到）"""
        gt_support, lt_support = self.cut_tree()
        gt_movie = list(gt_support.keys())
        lt_movie = list(lt_support.keys())

        data_list = self.get_combinations(gt_movie)
        if len(data_list) == 0:
            data_list = []

        for i in lt_movie:
            for j in data_list:
                if set(list(i)).issubset(list(j)):
                    data_list.remove(j)

        self.move_cut_set = data_list

    def num_count(self):
        """计算组合项集中的元素在用户-物品倒排表当中出现的次数，主要用于计算支持度"""
        data_list = collections.OrderedDict()
        for user, movies in self.train_set.items():

            for i in self.move_cut_set:
                if set(list(i)).issubset(list(movies)):
                    keys = "、".join(list(i))
                    data_list.setdefault(i, 0)
                    data_list[i] += 1

if __name__ == '__main__':
    ar = AssociationRules()
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
    rating_file = base_path + 'ml-100k/u.data'
    ar.get_dataset(rating_file)
    ar.move_cut()
    print(ar.move_cut_set)




