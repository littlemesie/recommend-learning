# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/5/8 23:00
@summary:
"""

import numpy as np
from cb.item_profiles import item_features
from cb.user_profile import user_item_rating


def cos_measure(item_feature_vector, user_rated_items_matrix, rate=0.001):
    """
    计算item之间的余弦夹角相似度
    :param item_feature_vector: 待测量的item特征向量
    :param user_rated_items_matrix: 用户已评分的items的特征矩阵
    :return: 待计算item与用户已评分的items的余弦夹角相识度的向量
    """
    x_c = (item_feature_vector * user_rated_items_matrix.T) + rate
    mod_x = np.sqrt(item_feature_vector * item_feature_vector.T)
    mod_c = np.sqrt((user_rated_items_matrix * user_rated_items_matrix.T).diagonal())
    cos_xc = x_c / (mod_x * mod_c)

    return cos_xc


def cb_recommend_estimate(user, item_feature_matrix, user_item_matrix, N=10):
    """
    基于内容的推荐算法对item的评分进行估计
    :param user: 需要推荐的用户
    :param item_feature_matrix:
    :param user_item_matrix:
    :param N:
    :return:
    """
    # 用户有过评分的item
    user_rating_item = user_item_matrix[user][0]
    # 用户有过评分的item的特征矩阵
    user_rating_feature_matrix = np.matrix([item_feature_matrix[rate] for rate in user_rating_item])
    # 用户没有评分的矩阵
    # user_no_rating_feature_matrix = [value for item, value in item_feature_matrix.items() if item not in user_rating_item]

    for item, value in item_feature_matrix.items():
        if item not in user_rating_item:
            # 得到待计算item与用户已评分的items的余弦夹角相识度的向量
            cos_xc = cos_measure(np.matrix(item_feature_matrix[item]), user_rating_feature_matrix)
            # 计算uesr对该item的评分估计
            # rate_hat = estimate_rate(user_rating_feature_matrix, cos_xc)
            # print(user_rating_feature_matrix.shape, cos_xc.shape)
            cos_xc = np.mat(cos_xc).getA()
            print(cos_xc[0])
            break



if __name__ == '__main__':

    item_feature_matrix = item_features()
    user_item_matrix = user_item_rating()
    uid = "1"
    cb_recommend_estimate(uid,item_feature_matrix,user_item_matrix)