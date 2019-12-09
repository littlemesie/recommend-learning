"""
fm算法
"""
import xlearn as xl
import numpy as np
import pandas as pd
from utils.ctr_data import load_data

def train_model(train_X, valid_X, train_y, valid_y):
    """训练模型"""
    dtrain = xl.DMatrix(train_X, train_y)
    dtest = xl.DMatrix(valid_X)
    fm_model = xl.create_fm()
    fm_model.setTrain(dtrain)
    print(dtrain)

if __name__ == '__main__':
    train_path = '../../data/ctr/train.csv'
    test_path = '../../data/ctr/test.csv'
    train_X, valid_X, train_y, valid_y, test_y = load_data(train_path, test_path)
    train_model(train_X, valid_X, train_y, valid_y)