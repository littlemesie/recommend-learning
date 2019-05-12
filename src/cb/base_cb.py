# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/5/8 23:00
@summary:
"""

import numpy as np
import pandas as pd
from cb.item_profiles import item_features
from cb.user_profile import user_features
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


def estimate_rate(user_rated_vector, similarity):
    """
    估计用户对item的评分
    :param user_rated_vector: 用户已有item评分向量
    :param similarity: 待估计item和已评分item的相识度向量
    :return:用户对item的评分的估计
    """
    rate_hat = (user_rated_vector * similarity.T) / similarity.sum()
    return rate_hat[0, 0]


def comp_user_feature(user_rated_vector, item_feature_matrix):
    """
    根据user的评分来计算得到user的喜好特征
    :param user_rated_vector: user的评分向量
    :param item_feature_matrix: item的特征矩阵
    :return: user的喜好特征
    """
    #user评分的均值
    user_rating_mean = user_rated_vector.mean()
    # 分别得到user喜欢和不喜欢item的向量以及item对应的引索(以该user的评分均值来划分)
    user_like_item = user_rated_vector.loc[user_rated_vector >= user_rating_mean]
    user_unlike_item = user_rated_vector.loc[user_rated_vector < user_rating_mean]

    user_like_item_index = map(int, user_like_item.index.values)
    user_unlike_item_index = map(int, user_unlike_item.index.values)

    user_like_item_rating = np.matrix(user_like_item.values)
    user_unlike_item_rating = np.matrix(user_unlike_item.values)

    #得到user喜欢和不喜欢item的特征矩阵
    user_like_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_like_item_index, :].values)
    user_unlike_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_unlike_item_index, :].values)

    # print user_like_item, user_unlike_item

    #计算user的喜好特征向量，以其对item的评分作为权重
    weight_of_like = user_like_item_rating / user_like_item_rating.sum()
    weight_of_unlike = user_unlike_item_rating / user_unlike_item_rating.sum()

    #计算user的喜欢特征和不喜欢特征以及总特征
    user_like_feature = weight_of_like * user_like_item_feature_matrix
    user_unlike_feature = weight_of_unlike * user_unlike_item_feature_matrix
    user_feature_tol = user_like_feature - user_unlike_feature

    return user_feature_tol


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


def CB_recommend_top_K(user_feature, item_feature_matrix, K):
    """
    计算得到与user最相似的Top K个item推荐给user
    :param user_feature: 待推荐用户的对item的评分向量
    :param item_feature_matrix: 包含所有item的特征矩阵
    :param K: 推荐给user的item数量
    :return: 与user最相似的Top K个item的编号
    """

    # 得到user已评分和未评分的item向量
    user_rated_vector = user_feature.loc[user_feature > 0]
    user_unrated_vector = user_feature.loc[user_feature == 0]

    #未评分item的特征矩阵
    user_unrated_item_index = map(int, user_unrated_vector.index.values)
    user_unrated_item_feature_matrix = np.matrix(item_feature_matrix.loc[user_unrated_item_index, :].values)

    #user喜好总特征
    user_feature_tol = comp_user_feature(user_rated_vector, item_feature_matrix)
    #得到相似度并进行排序
    similarity = list(np.array(cos_measure(user_feature_tol, user_unrated_item_feature_matrix))[0])
    key = {'item_index': user_unrated_item_index,
           'similarity': similarity}
    item_sim_df = pd.DataFrame(key)
    item_sim_df.sort_values('similarity', ascending=False, inplace=True)

    return item_sim_df.iloc[:K, 0].values


if __name__ == '__main__':

    item_feature_matrix = item_features()
    user_item_matrix = user_item_rating()
    uid = "1"
    cb_recommend_estimate(uid,item_feature_matrix,user_item_matrix)