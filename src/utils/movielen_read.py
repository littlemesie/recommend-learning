# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/5/4 20:41
@summary:
"""
import os
import random
import codecs

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"

def load_file(filename):
    """读文件，返回文件的每一行"""
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            # 去掉文件第一行的title
            if i == 0:
                continue
            yield line.strip('\r\n')

def loadfile(filename, encoding='UTF-8'):
    """
    根据文件名载入数据
    :param filename:
    :return:
    """
    with open(filename, "r",encoding=encoding) as f:
        for line in f:
            yield line


def read_rating_data( dataset='ml-100k', train_rate=0.8, seed=1):

    """
    载入评分数据
    @param path:  文件路径
    @param train_rate:   训练集所占整个数据集的比例，默认为0.8，表示所有的返回数据都是训练集
    @return: (训练集，测试集) list -- userId, movieId, rating
    """
    trainset = list()
    testset = list()
    user_set = set()
    item_set = set()
    random.seed(seed)
    path = None
    separator = None
    if dataset == 'ml-100k':
        path = base_path + 'ml-100k/u.data'
        separator = '\t'
    elif dataset == 'ml-1m':
        path = base_path + 'ml-1m/ratings.dat'
        separator = '::'
    for line in loadfile(filename=path):
        user, movie, rating, _ = line.split(separator)
        user_set.add(user)
        item_set.add(movie)
        if random.random() < train_rate:
            trainset.append([int(user), int(movie), int(rating)])
        else:
            testset.append([int(user), int(movie), int(rating)])
    return trainset, testset, user_set, item_set
