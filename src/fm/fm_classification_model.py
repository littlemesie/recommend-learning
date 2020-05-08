import tensorflow as tf

from sklearn.metrics import accuracy_score, roc_auc_score
from fm.util import *
class FM(object):
    def __init__(self, vec_dim, feat_num, lr, lamda):
        self.vec_dim = vec_dim
        self.feat_num = feat_num
        self.lr = lr
        self.lamda = lamda

        self._build_graph()

    def _build_graph(self):
        self.add_input()
        self.inference()

    def add_input(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.feat_num], name='input_x')
        self.y = tf.placeholder(tf.float32, shape=[None], name='input_y')

    def inference(self):
        with tf.variable_scope('linear_part'):
            w0 = tf.get_variable(name='bias', shape=[1], dtype=tf.float32)
            self.W = tf.get_variable(name='linear_w', shape=[self.feat_num], dtype=tf.float32)
            self.linear_part = w0 + tf.reduce_sum(tf.multiply(self.x, self.W), axis=1)
        with tf.variable_scope('interaction_part'):
            self.V = tf.get_variable(name='interaction_w', shape=[self.feat_num, self.vec_dim], dtype=tf.float32)
            self.interaction_part = 0.5 * tf.reduce_sum(
                tf.square(tf.matmul(self.x, self.V)) - tf.matmul(tf.square(self.x), tf.square(self.V)),
                axis=1
            )
        self.y_logits = self.linear_part + self.interaction_part
        self.y_hat = tf.nn.sigmoid(self.y_logits)
        self.pred_label = tf.cast(self.y_hat > 0.5, tf.int32)
        self.loss = -tf.reduce_mean(self.y*tf.log(self.y_hat+1e-8) + (1-self.y)*tf.log(1-self.y_hat+1e-8))
        self.reg_loss = self.lamda*(tf.reduce_mean(tf.nn.l2_loss(self.W)) + tf.reduce_mean(tf.nn.l2_loss(self.V)))
        self.total_loss = self.loss + self.reg_loss

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)


EPOCH = 10
STEP_PRINT = 200
STOP_STEP = 2000

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
LAMDA = 1e-3

VEC_DIM = 10
base, test = loadData()
FEAT_NUM = base.shape[1]-1

def eval_acc(pred_label, y):
    acc = accuracy_score(y, pred_label.flatten())
    return acc
def eval_auc(y_hat, y):
    auc = roc_auc_score(y, y_hat.flatten())
    return auc

def getValTest():
    """split validation data/ test data"""
    for valTest_x, valTest_y in getBatchData(test, batch_size=8000):
        val_x, val_y = valTest_x[:4000], valTest_y[:4000]
        test_x, test_y = valTest_x[4000:], valTest_y[4000:]
        return val_x, val_y, test_x, test_y

def run():
    val_x, val_y, test_x, test_y = getValTest()

    model = FM(VEC_DIM, FEAT_NUM, LEARNING_RATE, LAMDA)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        step = 0
        best_metric = 0
        last_improved = 0
        stop = False
        print("====== let's train =====")
        for epoch in range(EPOCH):
            print("EPOCH: {}".format(epoch+1))
            for x_batch, y_batch in getBatchData(base, BATCH_SIZE):
                feed_dict = {model.x: x_batch, model.y:y_batch}
                sess.run(model.train_op, feed_dict=feed_dict)
                if step % STEP_PRINT == 0:

                    train_y_hat, train_pred_label, train_loss = sess.run([model.y_hat, model.pred_label, model.loss],
                                                                         feed_dict=feed_dict)
                    train_acc = eval_acc(train_pred_label, y_batch)
                    val_y_hat, val_pred_label, val_loss = sess.run([model.y_hat, model.pred_label, model.total_loss],
                                                                   feed_dict={model.x: val_x, model.y: val_y})
                    val_acc, val_auc = eval_acc(val_pred_label, val_y), eval_auc(val_y_hat, val_y)
                    improved_str = ''
                    if val_auc > best_metric:
                        best_metric = val_auc
                        last_improved = step
                        saver.save(sess=sess, save_path='./ckpt/best')
                        improved_str = '*'

                    msg = 'Iter: {0:>6}, Train acc: {1:>6.4}, Val acc: {2:>6.4}, Val auc: {3:>6.4}, Val loss: {4:6.6}, Flag: {5}'
                    print(msg.format(step, train_acc, val_acc, val_auc, val_loss, improved_str))
                step += 1
                if step-last_improved > STOP_STEP:
                    print("No optimization for a long time, auto-stopping...")
                    stop = True
                    break
            if stop:
                break

        print("====== let's test =====")
        saver.restore(sess=sess, save_path='./ckpt/best')
        test_y_hat, test_pred_label, test_loss = sess.run([model.y_hat, model.pred_label, model.total_loss],
                                                          feed_dict={model.x: test_x, model.y: test_y})
        test_acc, test_auc = eval_acc(test_pred_label, test_y), eval_auc(test_y_hat, test_y)
        msg = 'Test acc: {0:>6.4}, Test auc: {1:>6.4}'
        print(msg.format(test_acc, test_auc))

run()

# Iter:  59600, Train acc: 0.8125, Val acc:  0.684, Val auc: 0.7294, Val loss: 0.615628, Flag: *
# Iter:  59800, Train acc:  0.875, Val acc: 0.6665, Val auc: 0.7282, Val loss: 0.625017, Flag:
# Iter:  60000, Train acc: 0.9375, Val acc: 0.6767, Val auc: 0.7282, Val loss: 0.617686, Flag:
# Iter:  60200, Train acc:   0.75, Val acc: 0.6815, Val auc: 0.7277, Val loss: 0.614763, Flag:
# Iter:  60400, Train acc: 0.9062, Val acc:  0.681, Val auc: 0.7283, Val loss: 0.614414, Flag:
# Iter:  60600, Train acc: 0.6875, Val acc: 0.6853, Val auc: 0.7291, Val loss: 0.621548, Flag:
# Iter:  60800, Train acc:  0.625, Val acc:  0.679, Val auc: 0.7288, Val loss: 0.617327, Flag:
# Iter:  61000, Train acc: 0.7812, Val acc: 0.6835, Val auc: 0.7293, Val loss: 0.616952, Flag:
# Iter:  61200, Train acc: 0.8125, Val acc:  0.686, Val auc: 0.7292, Val loss: 0.614379, Flag:
# Iter:  61400, Train acc: 0.6562, Val acc:  0.688, Val auc: 0.7284, Val loss: 0.613859, Flag:
# Iter:  61600, Train acc: 0.6875, Val acc: 0.6725, Val auc: 0.7279, Val loss: 0.618824, Flag:
# No optimization for a long time, auto-stopping...
# ====== let's test =====
# Test acc: 0.6833, Test auc: 0.7369