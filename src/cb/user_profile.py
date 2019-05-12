# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/5/11 10:06
@summary: 用户画像
"""
import os
import numpy as np
from utils.movielen_read import loadfile

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"

def judge_sex(sex):
    """性别"""
    if sex == "M":
        return [1]
    elif sex == "F":
        return [2]
    else:
        return [0]

def judge_age(age):
    """
    年龄
    :param age:
    :return:
    """

    if age >= 18 and age <= 24:
        ages = 2
    elif age > 24 and age <= 34:
        ages = 3
    elif age >= 35 and age <= 44:
        ages = 4
    elif age >= 45 and age <= 55:
        ages = 5
    elif age >= 56:
        ages = 6
    else:
        ages = 1

    return [ages]

def user_features():
    """用户特征"""
    users_feature = {}
    for line in loadfile(base_path + "ml-1m/users.dat",encoding="ISO-8859-1"):
        arr = line.split("::")
        sex_feature = judge_sex(arr[1])
        age_feature = judge_age(int(arr[2]))
        user_feature_list = sex_feature + age_feature + [int(arr[3])]
        users_feature.setdefault(arr[0],user_feature_list)
    return users_feature

def user_item_rating():
    """user-item"""
    user_item = {}
    for line in loadfile(base_path + "ml-1m/ratings.dat"):
        arr = line.split("::")
        uid = arr[0]
        user_item.setdefault(uid, [[], []])
        user_item[uid][0].append(arr[1])
        user_item[uid][1].append(arr[2])

    return user_item

if __name__ == '__main__':
    uf = user_item_rating()
    print(uf["1"])