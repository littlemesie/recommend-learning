import numpy as np
from fm.fm_model import FM
from utils.ctr_data import load_data

EPOCH = 10
STEP_PRINT = 200

LEARNING_RATE = 0.02
BATCH_SIZE = 32
LAMDA = 0.001

VEC_DIM = 10

def run():
    train_path = '../../data/ctr/train.csv'
    test_path = '../../data/ctr/test.csv'
    train_X, valid_X, train_y, valid_y, test_data = load_data(train_path, test_path)
    FEAT_NUM = train_X.shape[1]

    config = {
        'feat_num': FEAT_NUM,
        'vec_dim': VEC_DIM,
        'lr': LEARNING_RATE,
        'lamda': LAMDA,
        'epoch': 10,
    }

    model = FM(**config)
    model.fit(np.array(train_X), np.array(train_y), np.array(valid_X), np.array(valid_y))
    test_pred = model.predict(np.array(valid_X), np.array(valid_y))
    print(test_pred)


run()
# Test acc:  0.962, Test auc: 0.5603