"""
xgb 简单实现
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV
from utils.ctr_data import load_data
from utils.model_util import load_model,save_model

def tuning_model(train_X, train_y):
    """模型参数"""
    params = {
              # "n_estimators":[700, 900, 1000, 1200, 1500],
              # 'gamma': [0, 0.1, 0.2],
              # 'min_child_weight': [1.0, 1.1, 1.2, 1,3, 1.4, 1.5],
              # 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
              # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
              # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
              # 'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
              'eta': [0.01, 0.02, 0.03, 0.04, 0.05],
              }

    xlf = xgb.XGBClassifier(
        booster="gbtree",
        objective='rank:pairwise',
        eval_metric="auc",
        gamma=0.2,
        min_child_weight=1.2,
        max_depth=10,
        learning_rate=0.01,
        n_estimators=1500,
        subsample=0.7,
        nthread=12,
        colsample_bytree=0.5,
        colsample_bylevel=1,
        max_delta_step=0,
        eta=0.01,
        seed=0,
    )
    print(np.array(train_y))
    from sklearn import preprocessing
    import warnings
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    le = preprocessing.LabelEncoder()
    gsearch = GridSearchCV(xlf, param_grid=params, cv=3)
    gsearch.fit(np.array(train_X), np.array(train_y))

    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def train_model(train_X, valid_X, train_y, valid_y):
    """模型训练"""
    dtrain = xgb.DMatrix(train_X, train_y)
    dtest = xgb.DMatrix(valid_X)
    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'n_estimators': 1500,
              'gamma': 0.2,
              'min_child_weight': 1.2,
              'max_depth': 10,
              'lambda': 1,
              'subsample': 0.7,
              'colsample_bytree': 0.5,
              'colsample_bylevel': 1,
              'eta': 0.01,
              'seed': 0,
              }

    watchlist = [(dtrain, 'train')]
    model_path = "model/xgb.pkl"
    try:
        model = load_model(model_path)
    except:
        pass
        model = xgb.train(params, dtrain, evals=watchlist)
        save_model(model_path, model)
    prob = np.array(model.predict(dtest))
    fpr, tpr, thresholds = roc_curve(valid_y, prob)
    aucs = auc(fpr, tpr)
    print(aucs)
    return model

def test(model, test_y):
    """训练数据集"""

    dtest = xgb.DMatrix(test_y)
    prob = np.array(model.predict(dtest))
    print(prob)

if __name__ == '__main__':
    train_path = '../../data/ctr/train.csv'
    test_path = '../../data/ctr/test.csv'
    train_X, valid_X, train_y, valid_y, test_y = load_data(train_path, test_path)

    model = train_model(train_X, valid_X, train_y, valid_y)
    # tuning_model(train_X, train_y)