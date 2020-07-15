import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from fm.fm_model import FM
from utils.ctr_data import load_data, get_batch_data, get_test_data

def eval_acc(pred_label, y):
    acc = accuracy_score(y, pred_label.flatten())
    return acc

def eval_auc(y_hat, y):
    auc = roc_auc_score(y, y_hat.flatten())
    return auc

EPOCH = 10
STEP_PRINT = 100
STOP_STEP = 1000

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
LAMDA = 1e-3

VEC_DIM = 10

def run():
    train_path = '../../data/ctr/train.csv'
    test_path = '../../data/ctr/test.csv'
    train_X, valid_X, train_y, valid_y, test_data = load_data(train_path, test_path)
    FEAT_NUM = train_X.shape[1]

    model = FM(VEC_DIM, FEAT_NUM, LEARNING_RATE, LAMDA)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        step = 0
        stop = False
        print("====== let's train =====")
        for epoch in range(EPOCH):
            print("EPOCH: {}".format(epoch + 1))
            for x_batch, y_batch in get_batch_data(train_X, train_y, BATCH_SIZE):
                feed_dict = {model.x: x_batch, model.y: y_batch}
                sess.run(model.train_op, feed_dict=feed_dict)
                if step % STEP_PRINT == 0:

                    train_y_hat, train_pred_label, train_loss = sess.run([model.y_hat, model.pred_label, model.loss],
                                                                         feed_dict=feed_dict)
                    train_acc = eval_acc(train_pred_label, y_batch)

                    msg = 'Iter: {0:>6}, Train acc: {1:>6.4}'
                    print(msg.format(step, train_acc))
                step += 1
                if step > STOP_STEP:
                    print("No optimization for a long time, auto-stopping...")
                    stop = True
                    break
            if stop:
                break

        test_x, test_y = get_test_data(valid_X, valid_y)
        print("====== let's test =====")
        # saver.restore(sess=sess, save_path='./ckpt/best')
        test_y_hat, test_pred_label, test_loss = sess.run([model.y_hat, model.pred_label, model.total_loss],
                                                          feed_dict={model.x: test_x, model.y: test_y})
        test_acc, test_auc = eval_acc(test_pred_label, test_y), eval_auc(test_y_hat, test_y)
        msg = 'Test acc: {0:>6.4}, Test auc: {1:>6.4}'
        print(msg.format(test_acc, test_auc))

run()
# Test acc:  0.962, Test auc: 0.5603