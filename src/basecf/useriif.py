"""
    Ｕser-IIF算法
"""
import time
import os
import math
from collections import defaultdict
from basecf.usercf import UserBasedCF



class UserIIF(UserBasedCF):

    def user_similarity(self):
        """建立用户的协同过滤矩阵"""
        # 建立用户倒排表
        item_user = dict()
        for user, items in self.trainSet.items():
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user)

        # 建立用户协同过滤矩阵
        N = defaultdict(int)  # 记录用户购买商品数
        for item, users in item_user.items():
            for u in users:
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, defaultdict(int))
                    self.user_sim_matrix[u][v] += 1. / math.log(1 + len(item_user[item]))

        # 计算相关度
        for u, related_users in self.user_sim_matrix.items():
            for v, con_items_count in related_users.items():
                self.user_sim_matrix[u][v] = con_items_count / math.sqrt(N[u] * N[v])



if __name__ == '__main__':
    t1 = time.time()
    base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/"
    rating_file = base_path + 'ml-100k/u.data'
    user_iif = UserIIF()
    user_iif.get_dataset(rating_file)
    user_iif.user_similarity()
    user_iif.evaluate()
    print("time:{}".format(time.time() - t1))
    # precision:0.3486, recall:0.1320, coverage:0.2390, popularity:5.3135
    # time:9.83896279335022