# coding:utf8

import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse.linalg import svds

RATE_MATRIX = np.array([[5, 5, 3, 0, 5, 5], [5, 0, 4, 0, 4, 4],
                        [0, 3, 0, 5, 4, 5], [5, 4, 3, 3, 5, 5]])

nmf = NMF(n_components=2)
user_distribution = nmf.fit_transform(RATE_MATRIX)
item_distribution = nmf.components_
print(user_distribution)
print(item_distribution)


