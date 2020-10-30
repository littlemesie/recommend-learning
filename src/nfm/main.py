import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from utils.metric import gini_norm
from nfm import nfm_model
from utils import config_util as config
from utils.data_reader_util import FeatureDictionary, DataParser


def preprocess(df):
    cols = [c for c in df.columns if c not in ['id', 'target']]
    # df['missing_feat'] = np.sum(df[df[cols]==-1].values,axis=1)
    df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    return df

def load_data():
    """加载数据"""
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_test = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ['id', 'target']]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        return df

    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    cols = [c for c in df_train.columns if c not in ['id','target']]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = df_train[cols].values
    y_train = df_train['target'].values

    X_test = df_test[cols].values
    ids_test = df_test['id'].values

    cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return df_train, df_test, X_train, y_train, X_test, ids_test, cat_features_indices

def run(df_train, df_test, folds, params):
    """run model"""
    fd = FeatureDictionary(df_train=df_train, df_test=df_test, numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)

    data_parser = DataParser(feat_dict=fd)
    # Xi_train :列的序号
    # Xv_train :列的对应的值
    # y_train : label
    Xi_train, Xv_train, y_train = data_parser.parse(df=df_train, has_label=True)

    Xi_test, Xv_test, ids_test = data_parser.parse(df=df_test)

    # 特征数
    params['feature_size'] = fd.feat_dim
    # field size
    params['field_size'] = len(Xi_train[0])

    y_train_meta = np.zeros((df_train.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((df_test.shape[0], 1), dtype=float)

    _get = lambda x, l: [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):

        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        nfm = nfm_model.NFM(**params)
        nfm.fit(Xi_train_, Xv_train_, y_train_)

        # y_train_meta[valid_idx, 0] = nfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:, 0] += nfm.predict(Xi_test, Xv_test)

    return y_train_meta, y_test_meta




if __name__ == '__main__':
    pnn_params = {
        "embedding_size": 8,
        "deep_layers": [32, 32],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layer_activation": tf.nn.relu,
        "epoch": 10,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "verbose": True,
        "random_seed": config.RANDOM_SEED,
        "deep_init_size": 50,
        "use_inner": False

    }

    # load data
    dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()

    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))

    y_train_pnn, y_test_pnn = run(dfTrain, dfTest, folds, pnn_params)
    print(y_test_pnn)