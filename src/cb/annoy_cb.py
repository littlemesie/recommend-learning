import numpy as np
from annoy import AnnoyIndex
from utils.item_util import item_features

class AnnoyCB:
    def __init__(self, n_sim_movie=10, trees=10, model_name='angular'):
        self.n_sim_movie = n_sim_movie
        self.trees = trees
        self.model_name = model_name

    def fit(self, item_matrix):
        num, vec_dim = item_matrix.shape
        self.model = AnnoyIndex(vec_dim, self.model_name)
        for i, vec in enumerate(item_matrix):
            self.model .add_item(i, vec)
        self.model.build(self.trees)

    def predict(self, item_matrix):
        num, vec_dim = item_matrix.shape
        res_result = []
        for i in range(num):
            itmes = self.model.get_nns_by_item(i, self.n_sim_movie)
            res_result.append(itmes)

        return res_result
if __name__ == '__main__':
    acb = AnnoyCB()
    items = item_features()
    item_matrix = []
    item_index_mapping = {}  # {item_matrix_index: item_id}
    index = 0
    for item_id, feature in items.items():
        item_matrix.append(feature)
        item_index_mapping[index] = int(item_id)
        index += 1
    item_matrix = np.array(item_matrix)
    acb.fit(item_matrix)
    res_result = acb.predict(item_matrix)
    print(res_result)