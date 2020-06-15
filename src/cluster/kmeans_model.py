import random
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from cluster.util import load_data, read_rating_data
from utils import metric

class KMeansRecommend:
    def __init__(self):
        # 目标用户推荐10部电影
        self.n_rec_movie = 10
        self.data = {}
        # user和item集合
        self.user_set = set()
        self.item_set = set()

        self.user_ret = {}
        self.cluster_ret = {}


    def train(self):
        """"""
        self.data, self.user_set, self.item_set = read_rating_data()
        user_id, user_info = load_data()
        # 使用kmeans 进行聚类
        labels = KMeans(n_clusters=120, random_state=0).fit_predict(user_info)
        print("calinski_harabaz_score:{:.4f}, silhouette_score:{:.4f}".format(calinski_harabaz_score(user_info, labels), silhouette_score(user_info, labels)))

        for i, label in enumerate(labels):
            self.user_ret.setdefault(user_id[i], label)
            uid_items = self.data[user_id[i]]
            if label not in self.cluster_ret:
                self.cluster_ret.setdefault(label, set())
                self.cluster_ret[label] = uid_items
            else:
                items = self.cluster_ret[label] | uid_items
                self.cluster_ret[label] = items


    def recommend(self, user):
        """推荐"""
        N = self.n_rec_movie
        label = self.user_ret[user]
        items = self.cluster_ret[label]
        rec_movie = random.sample(items, N)
        return list(rec_movie)

    def evaluate(self):
        print('Evaluating start ...')
        test_user_items = self.data
        # 推荐
        recommed_dict = dict()
        for user in self.user_set:
            recommed = self.recommend(user)
            recommed_dict.setdefault(user, recommed)

        item_popularity = dict()
        for user, items in self.data.items():
            for item in items:
                if item in item_popularity:
                    item_popularity[item] += 1
                else:
                    item_popularity.setdefault(item, 1)

        precision = metric.precision(recommed_dict, test_user_items)
        recall = metric.recall(recommed_dict, test_user_items)
        coverage = metric.coverage(recommed_dict, self.item_set)
        popularity = metric.popularity(item_popularity, recommed_dict)
        print("precision:{:.4f}, recall:{:.4f}, coverage:{:.4f}, popularity:{:.4f}".format(precision, recall, coverage,
                                                                                           popularity))

if __name__ == '__main__':
    km = KMeansRecommend()
    km.train()
    km.evaluate()
    # rec = km.recommend(1)
    # print(rec)

    # n_clusters=50
    # calinski_harabaz_score: 101.1684, silhouette_score: 0.6761
    # precision:0.1247, recall:0.0130, coverage:0.8524, popularity:4.1485

    # n_clusters=80
    # calinski_harabaz_score:143.6128, silhouette_score:0.8463
    # precision:0.1619, recall:0.0169, coverage:0.8131, popularity:4.3114

    # n_clusters=100
    # calinski_harabaz_score:194.5926, silhouette_score:0.9061
    # precision:0.1744, recall:0.0182, coverage:0.8202, popularity:4.3032

    # n_clusters=120
    # calinski_harabaz_score:298.2859, silhouette_score:0.9335
    # precision:0.1984, recall:0.0207, coverage:0.8125, popularity:4.3404