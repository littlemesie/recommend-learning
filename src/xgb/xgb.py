"""
xgb 简单实现
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import auc, roc_curve
from utils.ctr_data import load_data
from utils.model_util import load_model,save_model


def train_model(train_data, test_data, train_label):
    """模型训练"""
    dtrain = xgb.DMatrix(train_data, train_label)
    dtest = xgb.DMatrix(test_data)
    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'gamma': 0.1,
              'min_child_weight': 1.1,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'exact',
              'seed': 0,
              'nthread': 12
              }

    watchlist = [(dtrain, 'train')]
    model_path = "model/xgb.pkl"
    try:
        model = load_model(model_path)
    except:
        pass
        model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
        save_model(model_path, model)
    prob = np.array(model.predict(dtest))
    print(prob)


if __name__ == '__main__':
    train_path = '../../data/ctr/train.csv'
    test_path = '../../data/ctr/test.csv'
    train_data, test_data, train_label = load_data(train_path, test_path)
    train_model(train_data, test_data, train_label)