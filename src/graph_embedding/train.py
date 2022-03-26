# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2022/3/23 下午1:56
@summary:
"""
import os
import networkx as nx
import matplotlib.pyplot as plt
from models.deepwalk import DeepWalk

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

if __name__ == '__main__':
    data_list = load_data()
    G = nx.Graph()
    G.add_weighted_edges_from(data_list)

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, epochs=3)
    embeddings = model.get_embeddings()
    print(embeddings.keys())
    print(len(list(embeddings.keys())))
    print(embeddings['user_196'])
    # plt.subplot(121)
    # nx.draw_shell(G, with_labels=True, font_weight='bold')
    # plt.show()