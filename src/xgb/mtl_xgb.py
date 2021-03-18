import pandas as pd
import numpy as np
from xgboost import DMatrix,train


# 训练参数
xgb_rank_params = {
    'bst:max_depth': 2,
    'bst:eta': 1,
    'silent': 1,
    'objective': 'rank:pairwise',
    'nthread': 4,
    'eval_metric': 'ndcg'
}

# 产生随机样本
# 一共2组*每组3条，6条样本，特征维数是2
n_group = 2
n_choice = 3
dtrain = np.random.uniform(0, 100, [n_group * n_choice, 2])

dtarget = np.array([np.random.choice([0, 1, 2], 3, False) for i in range(n_group)]).flatten()

# n_group用于表示从前到后每组各自有多少样本，前提是样本中各组是连续的，[3，3]表示一共6条样本中前3条是第一组，后3条是第二组
dgroup= np.array([n_choice for i in range(n_group)]).flatten()

# 构造Xgboost训练数据
xgbTrain = DMatrix(dtrain, label=dtarget)
xgbTrain.set_group(dgroup)

# 构造评测数据
dtrain_eval = np.random.uniform(0, 100, [n_group*n_choice, 2])
xgbTrain_eval = DMatrix(dtrain_eval, label=dtarget)
xgbTrain_eval .set_group(dgroup)
evallist = [(xgbTrain, 'train'), (xgbTrain_eval, 'eval')]

# 训练模型
rankModel = train(xgb_rank_params, xgbTrain, num_boost_round=20, evals=evallist)

# 测试模型
dtest = np.random.uniform(0, 100, [n_group*n_choice, 2])
dtestgroup = np.array([n_choice for i in range(n_group)]).flatten()
xgbTest = DMatrix(dtest)
xgbTest.set_group(dgroup)
print(rankModel.predict(xgbTest))