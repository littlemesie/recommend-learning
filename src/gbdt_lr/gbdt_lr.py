"""
GBDT_LR
基于lightgbm实现
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def load_data():
    print('Load data...')
    df_train = pd.read_csv('../../data/ctr/train.csv')
    # df_test = pd.read_csv('../../data/ctr/test.csv')

    NUMERIC_COLS = [
        "ps_reg_01", "ps_reg_02", "ps_reg_03",
        "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
    ]

    # print(df_train.head(10))
    # split data for train test
    train, test = train_test_split(df_train, test_size=0.2)
    X_train = train[NUMERIC_COLS]
    X_test = test[NUMERIC_COLS]
    y_train = train['target']
    y_test = test['target']

    # create dataset for lightgbm
    print(X_test)
    print(y_test)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    return X_train, y_train, X_test, y_test, lgb_train, lgb_test

def gbdt_lr_model():
    """gbdt_lr model"""
    #load data
    X_train, y_train, X_test, y_test, lgb_train, lgb_test = load_data()
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 64,
        'num_trees': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # number of leaves,will be used in feature transformation
    num_leaf = 64
    print('Start training...')
    # train model
    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_train)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')
    print('Start predicting...')
    # predict and get data on leaves, train data
    y_train_pred = gbm.predict(X_train, pred_leaf=True)

    print(np.array(y_train_pred).shape)
    print(y_train_pred[:10])

    print('Writing transformed train data')
    # N * num_tress * num_leafs
    transformed_train_matrix = np.zeros([len(y_train_pred), len(y_train_pred[0]) * num_leaf],
                                           dtype=np.int64)
    for i in range(0, len(y_train_pred)):
        temp = np.arange(len(y_train_pred[0])) * num_leaf + np.array(y_train_pred[i])
        transformed_train_matrix[i][temp] += 1

    # predict test data
    y_test_pred = gbm.predict(X_test, pred_leaf=True)
    print(y_test_pred)
    print('Writing transformed test data')
    transformed_test_matrix = np.zeros([len(y_test_pred), len(y_test_pred[0]) * num_leaf], dtype=np.int64)
    for i in range(0, len(y_test_pred)):
        temp = np.arange(len(y_test_pred[0])) * num_leaf + np.array(y_test_pred[i])
        transformed_test_matrix[i][temp] += 1

    # logestic model construction
    lm = LogisticRegression(penalty='l2', C=0.05)
    # fitting the data
    lm.fit(transformed_train_matrix, y_train)
    # Give the probabilty on each label
    y_pred_test = lm.predict_proba(transformed_test_matrix)

    print(y_pred_test[:, 1])

    fpr, tpr, thresholds = roc_curve(np.array(y_test), np.array(y_pred_test[:, 1]))
    aucs = auc(fpr, tpr)
    print(aucs)
    #
    # NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:, 1]) + (1-y_test)/2 * np.log(1 - y_pred_test[:, 1])))
    # print("Normalized Cross Entropy " + str(NE))

if __name__ == '__main__':
    # X_train, y_train, X_test, y_test, lgb_train, lgb_test = load_data()
    # print(lgb_train)
    gbdt_lr_model()

