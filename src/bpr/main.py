import numpy as np
import tensorflow as tf
import random
from collections import defaultdict
import faiss
from bpr.bpr_model import BPR

def load_data():
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open('../../data/ml-100k/u.data', 'r') as f:
        for line in f.readlines():
            u, i, _, _ = line.split("\t")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)


    print("max_u_id:", max_u_id)
    print("max_i_idL", max_i_id)

    return max_u_id, max_i_id, user_ratings

def generate_test(user_ratings):
    """
    对每一个用户u，在user_ratings中随机找到他评分过的一部电影i,保存在user_ratings_test，
    后面构造训练集和测试集需要用到。
    """
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test


def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512):
    """
    构造训练用的三元组
    对于随机抽出的用户u，i可以从user_ratings随机抽出，而j也是从总的电影集中随机抽出，当然j必须保证(u,j)不在user_ratings中

    """
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]

        j = random.randint(1, item_count)
        while j in user_ratings[u]:
            j = random.randint(1, item_count)

        t.append([u, i, j])

    return np.asarray(t)


def generate_test_batch(user_ratings,user_ratings_test,item_count):
    """
    对于每个用户u，它的评分电影i是我们在user_ratings_test中随机抽取的，它的j是用户u所有没有评分过的电影集合，
    比如用户u有1000部电影没有评分，那么这里该用户的测试集样本就有1000个
    """
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1,item_count + 1):
            if not(j in user_ratings[u]):
                t.append([u,i,j])
        yield np.asarray(t)

if __name__ == '__main__':

    user_count, item_count, user_ratings = load_data()
    user_ratings_test = generate_test(user_ratings)

    config = {
        'user_size': user_count, 'item_size': item_count, 'embedding_size': 32
    }
    model = BPR(**config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, 4):
            _batch_bprloss = 0
            for k in range(1, 5000):
                uij = generate_train_batch(user_ratings, user_ratings_test, item_count)
                _bpr_loss, _train_op = sess.run([model.loss, model.optimizer],
                                               feed_dict={model.user: uij[:, 0], model.item_i: uij[:, 1], model.item_j: uij[:, 2]})

                _batch_bprloss += _bpr_loss

            print("epoch:", epoch)
            print("bpr_loss:", _batch_bprloss / k)

            user_count = 0
            _auc_sum = 0.0

            for t_uij in generate_test_batch(user_ratings, user_ratings_test, item_count):
                _test_bpr_loss, _test__train_op = sess.run([model.loss, model.optimizer],
                                               feed_dict={model.user: t_uij[:, 0], model.item_i: t_uij[:, 1], model.item_j: t_uij[:, 2]}
                                               )
                user_count += 1
            print("test_loss: ", _test_bpr_loss)

        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print(v)
        user_embs = sess.run(model.user_emb_w)
        item_embs = sess.run(model.item_emb_w)
        print("user_emb: ", user_embs)
        print("user_emb shape: ", user_embs.shape)
        print("item_emb: ", item_embs)
        print("item_emb shape: ", item_embs.shape)

        index = faiss.IndexFlatIP(config['embedding_size'])
        # faiss.normalize_L2(item_embs)
        index.add(item_embs)
        # faiss.normalize_L2(user_embs)
        D, I = index.search(np.ascontiguousarray(user_embs), 10)

        print("以下是给用户0的推荐：")
        recommed = [x for x in I[0]]
        print(recommed)