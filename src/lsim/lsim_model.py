import numpy as np
import random
import time
import pandas as pd
import math
import os
import operator
from concurrent.futures import ProcessPoolExecutor
import pyximport
pyximport.install()
from lsim import slim_util as slim


base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
class Data:
    def __init__(self, dataset='ml-100k'):
        """
        无上下文信息的隐性反馈数据集。
        :param dataset: 使用的数据集名字，当前有'ml-100k','ml-1m'
        """

        path = None
        separator = None
        if dataset == 'ml-100k':
            path = base_path + 'ml-100k/u.data'
            separator = '\t'
        elif dataset == 'ml-1m':
            path = base_path + 'ml-1m/ratings.dat'
            separator = '::'

        print('开始读取数据')

        # 从源文件读数据
        self.data = []
        for line in open(path, 'r'):
            data_line = line.split(separator)
            userID = int(data_line[0])
            movieID = int(data_line[1])
            # 无上下文信息的隐性反馈数据不需要评分和时间截
            #rating = int(data_line[2])
            #timestamp = int(data_line[3])
            self.data.append([userID, movieID])

        def compress(data, col):
            """
            压缩数据data第col列的数据。保证此列数字会从0开始连续出现，中间不会有一个不存在此列的数字。

            :param data: 二维列表数据
            :param col: 要压缩的列
            :return: 此列不同的数字个数（即此列最大数字加1）
            """
            e_rows = dict()  # 键是data数据第col列数据，值是一个存放键出现在的每一个行号的列表
            for i in range(len(data)):
                e = data[i][col]
                if e not in e_rows:
                    e_rows[e] = []
                e_rows[e].append(i)

            for rows, i in zip(e_rows.values(), range(len(e_rows))):
                for row in rows:
                    data[row][col] = i

            return len(e_rows)

        self.num_user = compress(self.data, 0)
        self.num_item = compress(self.data, 1)

        # 训练集和测试集
        self.train, self.test = self.__split_data()
        print('总共有{}条数据，训练集{}，测试集{}，用户{}，物品{}'.format(len(self.data), len(self.train), len(self.test), self.num_user, self.num_item))

    def __split_data(self):
        """
        将数据随机分成8份，1份作为测试集，7份作为训练集

        :return: 训练集和测试集
        """
        test = []
        train = []
        for user, item in self.data:
            if random.randint(1, 8) == 1:
                test.append([user, item])
            else:
                train.append([user, item])
        return train, test

class SLIM_Model:
    def __init__(self, data):
        """
        稀疏线性算法。
        :param data: 无上下文信息的隐性反馈数据集，包括训练集，测试集等
        """
        self.data = data

        print('稀疏线性算法')
        self.A = self.__user_item_matrix()  # 用户-物品行为矩阵

        self.alpha = None
        self.lam_bda = None
        self.max_iter = None  # 学习最大迭代次数
        self.tol = None  # 学习阈值
        self.N = None  # 每个用户最多推荐物品数量
        self.lambda_is_ratio = None  # lambda参数是否代表比例值

        self.W = None  # 系数集合
        self.recommendation = None

    def compute_recommendation(self, alpha=0.5, lam_bda=0.02, max_iter=1000, tol=0.0001, N=10, lambda_is_ratio=True):
        """
        开始计算推荐列表

        :param alpha: lasso占比（为0只有ridge-regression，为1只有lasso）
        :param lam_bda: elastic net系数
        :param max_iter: 学习最大迭代次数
        :param tol: 学习阈值
        :param N: 每个用户最多推荐物品数量
        :param lambda_is_ratio: lambda参数是否代表比例值。若为True，则运算时每列lambda单独计算；若为False，则运算时使用单一lambda的值
        """
        self.alpha = alpha
        self.lam_bda = lam_bda
        self.max_iter = max_iter
        self.tol = tol
        self.N = N
        self.lambda_is_ratio = lambda_is_ratio

        print('开始计算W矩阵（alpha=' + str(self.alpha) + ', lambda=' + str(self.lam_bda) + ', max_iter=' + str(
            self.max_iter) + ', tol=' + str(self.tol) + '）')
        self.W = self.__aggregation_coefficients()

        print('开始计算推荐列表（N=' + str(self.N) + '）')
        self.recommendation = self.__get_recommendation()

    def __user_item_matrix(self):
        A = np.zeros((self.data.num_user, self.data.num_item))
        for user, item in self.data.train:
            A[user, item] = 1
        return A

    def __aggregation_coefficients(self):
        group_size = 100  # 并行计算每组计算的行/列数
        n = self.data.num_item // group_size  # 并行计算分组个数
        starts = []
        ends = []
        for i in range(n):
            start = i * group_size
            starts.append(start)
            ends.append(start + group_size)
        if self.data.num_item % group_size != 0:
            starts.append(n * group_size)
            ends.append(self.data.num_item)
            n += 1

        print('进行covariance updates的预算')
        covariance_array = None
        with ProcessPoolExecutor() as executor:
            covariance_array = np.vstack(executor.map(slim.compute_covariance, [self.A] * n, starts, ends))
        slim.symmetrize_covariance(covariance_array)

        print('坐标下降法学习W矩阵')
        if self.lambda_is_ratio:
            with ProcessPoolExecutor() as executor:
                return np.hstack(executor.map(slim.coordinate_descent_lambda_ratio, [self.alpha] * n, [self.lam_bda] * n, [self.max_iter] * n, [self.tol] * n, [self.data.num_user] * n, [self.data.num_item] * n, [covariance_array] * n, starts, ends))
        else:
            with ProcessPoolExecutor() as executor:
                return np.hstack(executor.map(slim.coordinate_descent, [self.alpha] * n, [self.lam_bda] * n, [self.max_iter] * n, [self.tol] * n, [self.data.num_user] * n, [self.data.num_item] * n, [covariance_array] * n, starts, ends))

    def __recommend(self, user_AW, user_item_set):
        """
        给用户user推荐最多N个物品。

        :param user_AW: AW矩阵相乘的第user行
        :param user_item_set: 训练集用户user所有有过正反馈的物品集合
        :return: 推荐给本行用户的物品列表
        """
        rank = dict()
        for i in set(range(self.data.num_item)) - user_item_set:
            rank[i] = user_AW[i]
        return [items[0] for items in sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:self.N]]

    def __get_recommendation(self):
        """
        得到所有用户的推荐物品列表。

        :return: 推荐列表，下标i对应给用户i推荐的物品列表
        """
        # 得到训练集中每个用户所有有过正反馈物品集合
        train_user_items = [set() for u in range(self.data.num_user)]
        for user, item in self.data.train:
            train_user_items[user].add(item)

        AW = self.A.dot(self.W)

        # 对每个用户推荐最多N个物品
        recommendation = []
        for user_AW, user_item_set in zip(AW, train_user_items):
            recommendation.append(self.__recommend(user_AW, user_item_set))
        return recommendation

class Evaluation:
    def __init__(self, recommend_algorithm):
        """
        对推荐算法recommend_algorithm计算各种评测指标。

        :param recommend_algorithm: 推荐算法，包括推荐结果列表，数据集等
        """
        self.rec_alg = recommend_algorithm

        self.precision = None
        self.recall = None
        self.coverage = None
        self.popularity = None

    def evaluate(self):
        """
        评测指标的计算。
        """
        # 准确率和召回率
        self.precision, self.recall = self.__precision_recall()
        print('准确率 = ' + str(self.precision * 100) + "%  召回率 = " + str(self.recall * 100) + '%')

        # 覆盖率
        self.coverage = self.__coverage()
        print('覆盖率 = ' + str(self.coverage * 100) + '%')

        # 流行度
        self.popularity = self.__popularity()
        print('流行度 = ' + str(self.popularity))

    def __precision_recall(self):
        """
        计算准确率和召回率。

        :return: 准确率和召回率
        """
        # 得到测试集用户与其所有有正反馈物品集合的映射
        test_user_items = dict()
        for user, item in self.rec_alg.data.test:
            if user not in test_user_items:
                test_user_items[user] = set()
            test_user_items[user].add(item)

        # 计算准确率和召回率
        hit = 0
        all_ru = 0
        all_tu = 0
        for user, items in test_user_items.items():
            ru = set(self.rec_alg.recommendation[user])
            tu = items

            hit += len(ru & tu)
            all_ru += len(ru)
            all_tu += len(tu)
        return hit / all_ru, hit / all_tu

    def __coverage(self):
        """
        计算覆盖率

        :return: 覆盖率
        """
        recommend_items = set()
        for user in range(self.rec_alg.data.num_user):
            for item in self.rec_alg.recommendation[user]:
                recommend_items.add(item)
        return len(recommend_items) / self.rec_alg.data.num_item

    def __popularity(self):
        """
        计算新颖度（平均流行度）

        :return: 新颖度
        """
        item_popularity = [0 for i in range(self.rec_alg.data.num_item)]
        for user, item in self.rec_alg.data.train:
            item_popularity[item] += 1

        ret = 0
        n = 0
        for user in range(self.rec_alg.data.num_user):
            for item in self.rec_alg.recommendation[user]:
                ret += math.log(1 + item_popularity[item])
                n += 1
        return ret / n

if __name__ == '__main__':

    precisions = []
    recalls = []
    coverages = []
    popularities = []
    times = []

    data = Data()
    startTime = time.time()
    recommend = SLIM_Model(data)
    recommend.compute_recommendation()
    eva = Evaluation(recommend)
    eva.evaluate()
    times.append('%.3fs' % (time.time() - startTime))
    precisions.append('%.3f%%' % (eva.precision * 100))
    recalls.append('%.3f%%' % (eva.recall * 100))
    coverages.append('%.3f%%' % (eva.coverage * 100))
    popularities.append(eva.popularity)

    df = pd.DataFrame()
    df['precision'] = precisions
    df['recall'] = recalls
    df['coverage'] = coverages
    df['popularity'] = popularities
    df['time'] = times
    print(df)

    # recommend = SLIM_Model(Data())
    # recommend.compute_recommendation()
    # Evaluation(recommend).evaluate()