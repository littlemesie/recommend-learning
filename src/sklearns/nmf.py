# coding:utf8
"""
NMF矩阵分解
"""
import math
import numpy as np
import pandas as pd
from operator import itemgetter
from sklearn.decomposition import NMF

def load_data(data_file="../../data/ml-100k/u.data"):
    # 加载数据集
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(data_file, sep='\t', names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    data_matrix = np.zeros((n_users, n_items))
    for line in df.itertuples():
        data_matrix[line[1] - 1, line[2] - 1] = line[3]

    item_popular = {}
    # 统计在所有的用户中，不同电影的总出现次数
    for i_index in range(n_items):
        if np.sum(data_matrix[:, i_index]) != 0:
            item_popular[i_index] = np.sum(data_matrix[:, i_index] != 0)

    return n_users, n_items, data_matrix, item_popular
def evaluate(prediction, item_popular, name, train_data_matrix):
    hit = 0
    rec_count = 0
    test_count = 0
    popular_sum = 0
    all_rec_items = set()
    for u_index in range(n_users):
        items = np.where(train_data_matrix[u_index, :] == 0)[0]
        pre_items = sorted(
            dict(zip(items, prediction[u_index, items])).items(),
            key=itemgetter(1),
            reverse=True)[:20]

        # 对比测试集和推荐集的差异 item, w
        for item, _ in pre_items:
            if item in items:
                hit += 1
            all_rec_items.add(item)

            # 计算用户对应的电影出现次数log值的sum加和
            if item in item_popular:
                popular_sum += math.log(1 + item_popular[item])

        rec_count += len(pre_items)
        test_count += len(items)

    precision = hit / (1.0 * rec_count)
    recall = hit / (1.0 * test_count)
    coverage = len(all_rec_items) / (1.0 * len(item_popular))
    popularity = popular_sum / (1.0 * rec_count)
    print('%s: precision=%.4f \t recall=%.4f \t coverage=%.4f \t popularity=%.4f' % (
        name, precision, recall, coverage, popularity))


def recommend(u_index, prediction, train_data_matrix):
    items = np.where(train_data_matrix[u_index, :] == 0)[0]
    pre_items = sorted(
        dict(zip(items, prediction[u_index, items])).items(),
        key=itemgetter(1),
        reverse=True)[:10]

    print('原始结果：', items)
    print('推荐结果：', [key for key, value in pre_items])

if __name__ == "__main__":
    n_users, n_items, data_matrix, item_popular = load_data()
    # 计算稀疏矩阵的最大k个奇异值/向量
    nmf = NMF(n_components=2)
    user_distribution = nmf.fit_transform(data_matrix)
    item_distribution = nmf.components_
    prediction = np.dot(user_distribution, item_distribution)
    print(prediction.shape)
    evaluate(prediction, item_popular, 'nmf', data_matrix)
    # 推荐结果
    recommend(1, prediction, data_matrix)