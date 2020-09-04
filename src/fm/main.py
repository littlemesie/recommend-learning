import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from fm.fm_model_2 import FM
from utils import config_util as config
from utils.data_reader_util import FeatureDictionary, DataParser


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
    """运行deepfm"""
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
    print("build model...")
    dfm = FM(**params)

    _get = lambda x, l: [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

    test_pred = dfm.predict(Xi_test, Xv_test)
    print(test_pred)




if __name__ == '__main__':

    params = {
        "embedding_size": 32,
        "dropout_fm": [1.0, 1.0],
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "l2_reg": 0.01,
        "eval_metric": 'auc',
        "random_seed": config.RANDOM_SEED
    }

    # load data
    df_train, df_test, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()

    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))

    run(df_train, df_test, folds, params)

