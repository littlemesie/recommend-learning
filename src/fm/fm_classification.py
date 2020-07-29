import tensorflow as tf
from fm.util import load_data, transform_data, get_val_test
from fm.fm_model import FM
from utils.metric import eval_acc, eval_auc

EPOCH = 10
STEP_PRINT = 200
STOP_STEP = 2000

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
LAMDA = 1e-3

VEC_DIM = 10


def run():
    base, test = load_data()
    FEAT_NUM = base.shape[1] - 1
    train_x, train_y = transform_data(base)
    valid_x, valid_y, test_x, test_y = get_val_test(test)
    config = {
        'feat_num': FEAT_NUM,
        'vec_dim': VEC_DIM,
        'lr': LEARNING_RATE,
        'lamda': LAMDA,
        'epoch': 10,
    }

    model = FM(**config)
    model.fit(train_x, train_y, valid_x, valid_y)
    test_pred = model.predict(test_x, test_y)
    print(test_pred)

run()
# Test acc: 0.6842, Test auc:  0.738