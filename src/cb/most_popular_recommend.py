# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/7/31 22:21
@summary: 热门推荐之基于评分排序的推荐
"""
import os
import random
from operator import itemgetter
from utils.movielen_read import loadfile

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"

def load_data(train_rate=1):
    train_items_ratings = {}
    test_data = {}
    for line in loadfile(base_path + "ml-1m/ratings.dat"):
        arr = line.split("::")
        if random.random() < train_rate:
            train_items_ratings.setdefault(arr[1], set())
            train_items_ratings[arr[1]].add(arr[2])
        else:
            test_data.setdefault(arr[0], set())
            test_data[arr[0]].add(arr[1])

    return train_items_ratings, test_data

def calculate_weigh(items_ratings):
    """计算权重"""
    items_weigh = {}
    # 每个item的平均分及评分的次数
    items_means = {}
    sum_rating = 0
    sum_rating_times = 0
    for item, values in items_ratings.items():

        items_means.setdefault(item, list())
        sum = 0
        for value in values:
            sum += int(value)
        mean_rating = sum / len(values)
        sum_rating += mean_rating
        sum_rating_times += len(values)
        items_means[item].append(mean_rating)
        items_means[item].append(len(values))

    all_mean_rating = sum_rating / len(items_ratings)
    all_mean_rating_time = sum_rating_times / len(items_ratings)

    for item, values in items_means.items():
        weigh = (values[1] / (values[1] + all_mean_rating_time)) * values[0] + \
                (all_mean_rating / (values[1] + all_mean_rating_time)) * all_mean_rating
        items_weigh.setdefault(item, weigh)
    items_weigh = sorted(items_weigh.items(), key=itemgetter(1), reverse=True)
    print(items_weigh)

if __name__ == '__main__':
    train_items_ratings, test_data = load_data()
    calculate_weigh(train_items_ratings)
