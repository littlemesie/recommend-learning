# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/5/11 10:06
@summary: Item
"""
import re
import os
import numpy as np
from utils.movielen_read import loadfile

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"

def judge_year(year):
    """判断年份"""

    if year >= 2000:
        year_label = 1
    elif year >= 1990 and year < 2000:
        year_label = 2
    elif year >= 1980 and year < 1990:
        year_label = 3
    elif year >= 1970 and year < 1980:
        year_label = 4
    else:
        year_label = 0
    return [year_label]

def judge_genres(genres):
    """类型区分"""
    genres_lables = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
                   "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
                   "Western"]
    genres_list = np.zeros(len(genres_lables), dtype=int)
    for genre in genres:
        if genre in genres_lables:
            ind = genres_lables.index(genre)
            genres_list[ind] = 1
    return list(genres_list)

def item_features():
    """用户特征矩阵"""
    item_feature = {}
    for line in loadfile(base_path + "ml-1m/movies.dat", encoding="ISO-8859-1"):
        arr = line.split("::")
        year = int(re.findall("\d{4}", arr[1])[0])
        year_label = judge_year(year)
        genres = arr[2].split("|")
        genres_list = judge_genres(genres)
        item_feature_list = year_label + genres_list
        item_feature.setdefault(arr[0], item_feature_list)

    return item_feature

if __name__ == '__main__':
    count_2000 = 0
    count_1990 = 0
    count_1980 = 0
    count_1970 = 0
    other = 0

    item_feature = {}

    for line in loadfile(base_path + "ml-1m/movies.dat",encoding="ISO-8859-1"):
        arr = line.split("::")
        year = int(re.findall("\d{4}", arr[1])[0])
        year_label = judge_year(year)
        genres = arr[2].split("|")
        genres_list = judge_genres(genres)
        item_feature_list = year_label + genres_list
        item_feature.setdefault(arr[0], item_feature_list)
        if year >= 2000:
            count_2000 += 1
        elif year >= 1990 and year < 2000:
            count_1990 += 1
        elif year >= 1980 and year < 1990:
            count_1980 += 1
        elif year >= 1970 and year < 1980:
            count_1970 += 1
        else:
            other += 1
    print(count_2000,count_1990,count_1980,count_1970,other)
    print(item_feature)