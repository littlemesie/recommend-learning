# coding = utf-8
import random
import math
from operator import itemgetter
from tianchi_mobile import read_data


class UserCF():
    # 初始化相关参数
    def __init__(self):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10个商品
        self.n_sim_user = 20
        self.n_rec_item = 10

        # 将数据集划分为训练集和测试集
        self.train_set = {}
        self.test_set = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.item_count = 0


    def get_dataset(self, file_name, pivot=0.8):
        """
        读文件得到“user-item”数据
        :param file_name:
        :param pivot:
        :return:
        """
        train_set_len = 0
        test_set_len = 0
        for line in read_data.load_file(file_name):
            arr = line.split(',')
            user = arr[0]
            item = arr[1]
            rating = arr[2]
            if random.random() < pivot:
                self.train_set.setdefault(user, {})
                self.train_set[user][item] = rating
                train_set_len += 1
            else:
                self.test_set.setdefault(user, {})
                self.test_set[user][item] = rating
                test_set_len += 1

        print('Split training_set and test_set success!')
        print('train_set = %s' % train_set_len)
        print('test_set = %s' % test_set_len)

    def calc_user_sim(self):
        """
        计算用户之间的相似度
        :return:
        """
        # 构建“item-user”倒排索引
        print('Building item-user table ...')
        item_user = {}
        for user, items in self.train_set.items():
            for item in items:
                if item not in item_user:
                    item_user[item] = set()
                item_user[item].add(user)
        print('Build item-user table success!')

        self.item_count = len(item_user)
        print('Total item number = %d' % self.item_count)
        print(item_user)
        print('Build user co-rated item matrix ...')
        for item, users in item_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        print('Build user co-rated items matrix success!')
        print(self.user_sim_matrix)
        # 计算相似性
        print('Calculating user similarity matrix ...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.train_set[u]) * len(self.train_set[v]))
        print('Calculate user similarity matrix success!')

    def recommend(self, user):
        """
        针对目标用户U，找到其最相似的K个用户，产生N个推荐
        :param user:
        :return:
        """
        K = self.n_sim_user
        N = self.n_rec_item
        rank = {}
        watched_items = self.train_set[user]

        # v=similar user, wuv=similar factor
        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for item in self.train_set[v]:
                if item in watched_items:
                    continue
                rank.setdefault(item, 0)
                rank[item] += wuv
        print(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N])
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]



    def evaluate(self):
        """
        产生推荐并通过准确率、召回率和覆盖率进行评估
        :return:
        """
        print("Evaluation start ...")
        N = self.n_rec_item
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_items = set()

        for i, user, in enumerate(self.train_set):
            test_items = self.test_set.get(user, {})
            rec_items = self.recommend(user)
            for item, w in rec_items:
                if item in test_items:
                    hit += 1
                all_rec_items.add(item)
            rec_count += N
            test_count += len(test_items)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_items) / (1.0 * self.item_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))


if __name__ == '__main__':
    base_path = "/Volumes/d/taobao/tianchi_mobile_recommend_train_item.csv"
    userCF = UserCF()
    userCF.get_dataset(base_path)
    userCF.calc_user_sim()
    userCF.evaluate()
    # userCF.recommend('10')
