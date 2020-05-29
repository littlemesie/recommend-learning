import numpy as np
import random
from operator import itemgetter
from collections import OrderedDict
from utils.movielen_read import read_rating_data
from utils import metric

class SVD:
    def __init__(self, mat, K=20):
        self.mat = np.array(mat)
        self.K = K
        self.bi = OrderedDict()
        self.bu = OrderedDict()
        self.qi = OrderedDict()
        self.pu = OrderedDict()
        # self.avg = np.mean(self.mat[:, 2])
        self.avg = 2.5
        for i in range(self.mat.shape[0]):
            uid = self.mat[i, 0]
            iid = self.mat[i, 1]
            self.bi.setdefault(iid, 0)
            self.bu.setdefault(uid, 0)
            self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))

    def predict(self, uid, iid):
        """预测评分的函数"""
        # setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu，并设置初始值为0
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.K, 1)))
        self.pu.setdefault(uid, np.zeros((self.K, 1)))
        rating = self.avg + self.bi[iid] + self.bu[uid] + np.sum(self.qi[iid] * self.pu[uid])  # 预测评分公式
        # 由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    def train(self, steps=30, gamma=0.05, Lambda=0.15):  # 训练函数，step为迭代次数。
        print('train data size', self.mat.shape)
        for step in range(steps):
            print('step', step + 1, 'is running')
            # 随机梯度下降算法，kk为对矩阵进行随机排序
            KK = np.random.permutation(self.mat.shape[0])
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                eui = rating - self.predict(uid, iid)
                rmse += eui ** 2
                self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
                self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
                tmp = self.qi[iid]
                self.qi[iid] += gamma * (eui * self.pu[uid] - Lambda * self.qi[iid])
                self.pu[uid] += gamma * (eui * tmp - Lambda * self.pu[uid])
            # gamma = 0.93 * gamma
            print('rmse is', np.sqrt(rmse / self.mat.shape[0]))


    def recommend(self, user, N):
        """
        针对目标用户U，产生N个推荐
        """
        recommend_dict = {}
        items = list(self.bi.keys())

        bu_mat = self.bu[user] * np.ones((len(items), 1))
        qi_list = [q.flatten() for q in list(self.qi.values())]
        rating_mat = self.avg + np.mat(bu_mat).flatten() + np.mat(self.pu[user].flatten()) * np.mat(qi_list).T
        rating_list = np.array(rating_mat)[0]

        for i, item in enumerate(items):
            recommend_dict.setdefault(item, rating_list[i])
        return sorted(recommend_dict.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self, train_data, test_data, item_set):
        """
        产生推荐并通过准确率、召回率和覆盖率等进行评估
        :return:
        """
        print("Evaluation start ...")
        test_user_items = dict()
        test_uids = set()
        for user, item, _ in test_data:
            test_uids.add(user)
            if user not in test_user_items:
                test_user_items[user] = set()
            test_user_items[user].add(item)
        test_uids = list(test_uids)

        item_popularity = dict()

        for user, item, _ in train_data:
            if item in item_popularity:
                item_popularity[item] += 1
            else:
                item_popularity.setdefault(item, 1)

        recommed_dict = {}
        for uid in test_uids:
            recommeds = self.recommend(uid, 10)
            item_ids = [rec[0] for rec in recommeds]
            recommed_dict.setdefault(uid, item_ids)

        precision = metric.precision(recommed_dict, test_user_items)
        recall = metric.recall(recommed_dict, test_user_items)
        coverage = metric.coverage(recommed_dict, item_set)
        popularity = metric.popularity(item_popularity, recommed_dict)
        print("precision:{:.4f}, recall:{:.4f}, coverage:{:.4f}, popularity:{:.4f}".format(precision, recall, coverage,
                                                                                           popularity))

    def test(self, test_data):

        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating - self.predict(uid, iid)
            rmse += eui ** 2
        print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))



train_data, test_data, user_set, item_set = read_rating_data()
a = SVD(train_data, 50)
a.train(steps=30)
# l = a.recommend(1, 10)
a.test(test_data)
a.evaluate(train_data, test_data, item_set)

# precision:0.0106, recall:0.0050, coverage:0.5767, popularity:2.3052


