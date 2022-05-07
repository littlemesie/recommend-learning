# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/3/23 下午1:56
@summary:
"""
import os
import faiss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from models.deepwalk import DeepWalk
from models.node2vec import Node2Vec

base_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data"

def load_data():
    """load data"""
    path = f"{base_path}/ml-100k/u.data"
    data_list = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            user, movie, rating, _ = line.split('\t')
            data_list.append([f"user_{user}", f"movie_{movie}", int(rating)])

    return data_list

def recommends(embeddings, top_N=10):
    """recommends"""
    item_embs = []
    item_map = {}
    i = 0
    for key in embeddings.keys():
        if 'movie' in key:
            ks = key.split('_')
            item_embs.append(embeddings[key])
            item_map[i] = ks[1]
            i += 1

    item_embs = np.array(item_embs, dtype='float32')
    num, vec_dim = item_embs.shape
    index = faiss.IndexFlatL2(vec_dim)
    index.add(item_embs)

    for key in embeddings.keys():
        if 'user' in key:
            ks = key.split('_')
            user_emb = np.array([embeddings[key]], dtype='float32')
            D, I = index.search(user_emb, top_N)
            recommend = [item_map[x] for x in I[0]]
            print(ks[1], recommend)

def train_deepwalk(data_list):
    """deepwalk"""
    G = nx.Graph()
    G.add_weighted_edges_from(data_list)

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, epochs=3)
    embeddings = model.get_embeddings()
    print(embeddings.keys())
    print(len(list(embeddings.keys())))
    print(embeddings['user_196'])
    # 推荐
    recommends(embeddings)
    # plt.subplot(121)
    # nx.draw_shell(G, with_labels=True, font_weight='bold')
    # plt.show()

def train_node2vec():
    """node2vec"""
    G = nx.Graph()
    G.add_weighted_edges_from(data_list)
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    model.train(window_size=5, epochs=3)
    embeddings = model.get_embeddings()
    print(embeddings.keys())
    print(len(list(embeddings.keys())))
    print(embeddings['user_196'])

if __name__ == '__main__':
    data_list = load_data()
    # deepwalk
    # train_deepwalk(data_list)

    # node2vec
    train_node2vec()