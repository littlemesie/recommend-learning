import random
import numpy as np
import tensorflow as tf
from utils.metric import eval_auc, eval_acc

class FM(object):
    def __init__(self, feat_num, vec_dim=10, lr=0.001, lamda=0.001, epoch=10, batch_size=32,
                 step_print=200, loss_type="logloss", optimizer_type="adam"):
        self.feat_num = feat_num
        self.vec_dim = vec_dim
        self.lr = lr
        self.lamda = lamda
        self.epoch = epoch
        self.batch_size = batch_size
        self.step_print = step_print
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type

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
        self.out = tf.nn.sigmoid(self.y_logits)
        self.pred = tf.cast(self.out > 0.5, tf.int32)

        # loss
        if self.loss_type == "logloss":
            logs = tf.losses.log_loss(self.y, self.out)
            self.loss = tf.reduce_mean(logs)
        elif self.loss_type == "mse":
            self.loss = tf.losses.mean_squared_error(self.y, self.out)

        # optimizer
        if self.optimizer_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)
        elif self.optimizer_type == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        elif self.optimizer_type == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.95).minimize(
                self.loss)

        # init
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def shuffle_batch(self, x_batch, y_batch):
        assert len(x_batch) == len(y_batch)
        length = len(x_batch)
        index = [i for i in range(length)]
        random.shuffle(index)
        x_batch_shuffle = [x_batch[i] for i in index]
        y_batch_shuffle = [y_batch[i] for i in index]
        return x_batch_shuffle, y_batch_shuffle

    def get_batch_data(self, train_x, train_y):
        start = 0
        while start < train_x.shape[0]:
            end = start + self.batch_size
            x_batch = np.array(train_x[start: end, :])
            y_batch = train_y[start: end]

            start = end
            x_batch_shuffle, y_batch_shuffle = self.shuffle_batch(x_batch, y_batch)
            yield x_batch_shuffle, y_batch_shuffle

    def fit(self, train_x, train_y, valid_x, valid_y, best_metric=0.8):
            step = 0
            stop = False
            for epoch in range(self.epoch):
                print("EPOCH: {}".format(epoch + 1))
                for x_batch, y_batch in self.get_batch_data(train_x, train_y):
                    feed_dict = {self.x: x_batch, self.y: y_batch}
                    self.sess.run(self.optimizer, feed_dict=feed_dict)
                    if step % self.step_print == 0:

                        train_pred, train_pred_label, train_loss = self.sess.run([self.out, self.pred, self.loss],
                                                                                 feed_dict=feed_dict)
                        train_acc = eval_acc(train_pred_label, y_batch)
                        val_pred, val_pred_label, val_loss = self.sess.run([self.out, self.pred, self.loss],
                                                                           feed_dict={self.x: valid_x, self.y: valid_y})
                        val_acc, val_auc = eval_acc(val_pred_label, valid_y), eval_auc(val_pred, valid_y)

                        if val_auc > best_metric:
                            stop = True
                            break
                        msg = 'Iter: {0:>6}, Train acc: {1:>6.4}, Train loss: {4:6.6}, Val acc: {2:>6.4}, Val auc: {3:>6.4}, Val loss: {4:6.6},'
                        print(msg.format(step, train_acc, train_loss, val_acc, val_auc, val_loss))
                    step += 1
                if stop:
                    break


    def predict(self, test_x, test_y=np.array([])):
        test_pred, test_pred_label, test_loss = self.sess.run([self.out, self.pred, self.loss],
                                                              feed_dict={self.x: test_x, self.y: test_y})
        if test_y.any():
            test_acc, test_auc = eval_acc(test_pred_label, test_y), eval_auc(test_pred, test_y)
            msg = 'Test acc: {0:>6.4}, Test auc: {1:>6.4}'
            print(msg.format(test_acc, test_auc))
        return test_pred