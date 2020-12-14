import numpy as np
from sklearn.model_selection import StratifiedKFold

X = np.array([[1, 2, 3, 4],
              [11, 12, 13, 14],
              [21, 22, 23, 24],
              [31, 32, 33, 34],
              [41, 42, 43, 44],
              [51, 52, 53, 54],
              [61, 62, 63, 64],
              [71, 72, 73, 74]])

y = np.array([1, 1, 0, 0, 1, 1, 0, 0])

stratified_folder = StratifiedKFold(n_splits=2, shuffle=True, random_state=np.random)
for train_index, test_index in stratified_folder.split(X, y):
    print("Stratified Train Index:", X[train_index])
    print("Stratified Test Index:", X[test_index])
    print("Stratified y_train:", y[train_index])
