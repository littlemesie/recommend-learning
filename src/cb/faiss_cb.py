import numpy as np
import faiss
from utils.item_util import item_features

class FaissCB:
    def __init__(self):
        self.n_sim_movie = 5

    def fit(self, item_matrix):
        num, vec_dim = item_matrix.shape
        # 创建索引
        self.faiss_index = faiss.IndexFlatL2(vec_dim)  # 使用欧式距离作为度量
        # 添加数据
        self.faiss_index.add(item_matrix)

    def predict(self, item_matrix):
        res_distance, res_index = self.faiss_index.search(item_matrix, self.n_sim_movie)
        print(res_index)

if __name__ == '__main__':
    fcb = FaissCB()
    items = item_features()
    item_matrix = []
    item_index_mapping = {}  # {item_matrix_index: item_id}
    index = 0
    for item_id, feature in items.items():
        item_matrix.append(feature)
        item_index_mapping[index] = int(item_id)
        index += 1

    item_matrix = np.array(item_matrix, dtype='float32')
    fcb.fit(item_matrix)
    fcb.predict(item_matrix)