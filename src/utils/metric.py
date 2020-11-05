# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/5/8 22:48
@summary:
"""

import math
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score

def RMSE(records):
    """
    计算RMSE
    @param records: 预测评价与真实评价记录的一个list
    @return: RMSE
    """
    numerator = sum([(pred_rating - actual_rating)**2 for _,  _, pred_rating, actual_rating in records])
    denominator = float(len(records))
    return math.sqrt(numerator) / denominator


def MSE(records):
    """
    计算MSE
    @param records: 预测评价与真实评价记录的一个list
    @return: MSE
    """
    numerator = sum([abs(pred_rating - actual_rating) for _, _, pred_rating, actual_rating in records])
    denominator = float(len(records))
    return numerator / denominator


def precision(recommends, tests):
    """
    计算Precision
    :param recommends: dict
        给用户推荐的商品，recommends为一个dict，格式为 { userID : 推荐的物品 }
    :param tests: dict
        测试集，同样为一个dict，格式为 { userID : 实际发生事务的物品 }
    :return: float
        Precision
    """
    n_union = 0.
    recommend_sum = 0.
    for user_id, items in recommends.items():
        recommend_set = set(items)
        test_set = set(tests[user_id])
        n_union += len(recommend_set & test_set)
        recommend_sum += len(recommend_set)

    return n_union / recommend_sum


def recall(recommends, tests):
    """
    计算Recall
    @param recommends:  dict
        给用户推荐的商品，recommends为一个dict，格式为 { userID : 推荐的物品 }
    @param tests:  dict
        测试集，同样为一个dict，格式为 { userID : 实际发生事务的物品 }
    @return: Recall
    """
    n_union = 0.
    user_sum = 0.
    for user_id, items in recommends.items():
        recommend_set = set(items)
        test_set = set(tests[user_id])
        n_union += len(recommend_set & test_set)
        user_sum += len(test_set)

    return n_union / user_sum


def coverage(recommends, all_items):
    """
    计算覆盖率
    @param recommends : dict形式 { userID : Items }
    @param all_items :  所有的items，为list或set类型
    """
    recommend_items = set()
    for _, items in recommends.items():
        for item in items:
            recommend_items.add(item)
    return len(recommend_items) / len(all_items)


def popularity(item_popular, recommends):
    """
    计算流行度
    @param item_popular:  商品流行度　dict形式{ itemID : popularity}
    @param recommends :  dict形式 { userID : Items }
    @return: 平均流行度
    """
    popularity = 0.  # 流行度
    n = 0.
    for _, items in recommends.items():
        for item in items:
            popularity += math.log(1. + item_popular.get(item, 0.))
            n += 1
    return popularity / n

def hit(recommends, tests):
    recommend_cnt = defaultdict(int)
    test_cnt = defaultdict(int)
    for user_id, items in recommends.items():
        for item in items:
            recommend_cnt[item] = +1

    for user_id, items in tests.items():
        for item in items:
            test_cnt[item] = +1

    return sum(recommend_cnt.values()) / sum(test_cnt.values())

def eval_acc(pred_label, y):
    """accuracy"""
    acc = accuracy_score(y, pred_label.flatten())
    return acc

def eval_auc(pred_label, y):
    """auc"""
    auc = roc_auc_score(y, pred_label.flatten())
    return auc

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

def map_(recommends, tests):
    """
    Compute the mean average precision (MAP) of a list of ranked items
    """
    aps = []
    for user_id, items in recommends.items():
        hits = 0
        sum_precs = 0
        single_ap = 0
        ground_truth = tests.get(user_id, [])
        for n in range(len(items)):
            if items[n] in ground_truth:
                hits += 1
                sum_precs += hits / (n + 1.0)

        if len(ground_truth) > 0:
            single_ap = sum_precs / len(ground_truth)

        aps.append(single_ap)

    return np.mean(aps)
