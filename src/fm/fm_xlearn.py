"""
fm算法
"""
import xlearn as xl
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from utils.ctr_data import load_data

def train_model(train_X, valid_X, train_y, valid_y):
    """训练模型"""

    fm_model = xl.FMModel(
        lr=0.02,
        reg_lambda=0.001,
        k=18,
        epoch=10,
        stop_window=4
    )

    fm_model.fit(train_X, train_y)
    y_pred = fm_model.predict(valid_X)
    fpr, tpr, thresholds = roc_curve(valid_y, np.array(y_pred))
    aucs = auc(fpr, tpr)
    print(aucs)

if __name__ == '__main__':
    train_path = '../../data/ctr/train.csv'
    test_path = '../../data/ctr/test.csv'
    train_X, valid_X, train_y, valid_y, test_y = load_data(train_path, test_path)
    train_model(train_X, valid_X, train_y, valid_y)