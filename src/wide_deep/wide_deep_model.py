# -*- coding:utf-8 -*-
import random
import numpy as np
import tensorflow as tf
from wide_deep.util import load_data
from utils.metric import eval_auc, eval_acc

class WideDeep(object):
    def __init__(self, vec_dim=None, field_lens=None, dnn_layers=None, wide_lr=None, l1_reg=None, deep_lr=None,
                 epoch=10, batch_size=32, step_print=200, loss_type="logloss", optimizer_type="adam"):
        self.vec_dim = vec_dim
        self.field_lens = field_lens
        self.field_num = len(field_lens)
        self.dnn_layers = dnn_layers
        self.wide_lr = wide_lr
        self.l1_reg = l1_reg
        self.deep_lr = deep_lr

        self.epoch = epoch
        self.batch_size = batch_size
        self.step_print = step_print
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type

        assert isinstance(dnn_layers, list) and dnn_layers[-1] == 1
        self._build_graph()

    def _build_graph(self):
        self.add_input()
        self.inference()

    def add_input(self):
        self.x = [tf.placeholder(tf.float32, name='input_x_%d' % i) for i in range(self.field_num)]
        self.y = tf.placeholder(tf.float32, shape=[None], name='input_y')
        self.is_train = tf.placeholder(tf.bool)

    def inference(self):
        with tf.variable_scope('wide_part'):
            w0 = tf.get_variable(name='bias', shape=[1], dtype=tf.float32)
            linear_w = [tf.get_variable(name='linear_w_%d'%i, shape=[self.field_lens[i]], dtype=tf.float32) for i in range(self.field_num)]
            wide_part = w0 + tf.reduce_sum(
                tf.concat([tf.reduce_sum(tf.multiply(self.x[i], linear_w[i]), axis=1, keep_dims=True) for i in range(self.field_num)], axis=1),
                axis=1, keep_dims=True) # (batch, 1)
        with tf.variable_scope('dnn_part'):
            emb = [tf.get_variable(name='emb_%d'%i, shape=[self.field_lens[i], self.vec_dim], dtype=tf.float32) for i in range(self.field_num)]
            emb_layer = tf.concat([tf.matmul(self.x[i], emb[i]) for i in range(self.field_num)], axis=1) # (batch, F*K)
            x = emb_layer
            in_node = self.field_num * self.vec_dim
            for i in range(len(self.dnn_layers)):
                out_node = self.dnn_layers[i]
                w = tf.get_variable(name='w_%d' % i, shape=[in_node, out_node], dtype=tf.float32)
                b = tf.get_variable(name='b_%d' % i, shape=[out_node], dtype=tf.float32)
                in_node = out_node
                if out_node != 1:
                    x = tf.nn.relu(tf.matmul(x, w) + b)
                else:
                    self.y_logits = wide_part + tf.matmul(x, w) + b

        self.out = tf.nn.sigmoid(self.y_logits)
        self.pred = tf.cast(self.out > 0.5, tf.int32)

        # loss

        if self.loss_type == "logloss":
            logs = tf.losses.log_loss(self.y, self.out)
            self.loss = tf.reduce_mean(logs)
        elif self.loss_type == "mse":
            self.loss = tf.losses.mean_squared_error(self.y, self.out)

        self.global_step = tf.Variable(0, trainable=False)
        # wide_part optimizer
        wide_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_part')
        wide_part_optimizer = tf.train.FtrlOptimizer(learning_rate=self.wide_lr, l1_regularization_strength=self.l1_reg).\
            minimize(loss=self.loss, global_step=self.global_step, var_list=wide_part_vars)

        # dnn_part optimizer
        dnn_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dnn_part')
        if self.optimizer_type == "adam":
            dnn_part_optimizer = tf.train.AdamOptimizer(learning_rate=self.deep_lr, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss, global_step=None, var_list=dnn_part_vars)
        elif self.optimizer_type == "adagrad":
            dnn_part_optimizer = tf.train.AdagradOptimizer(learning_rate=self.deep_lr,
                                                       initial_accumulator_value=1e-8).minimize(self.loss, global_step=None, var_list=dnn_part_vars)
        elif self.optimizer_type == "gd":
            dnn_part_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.deep_lr).minimize(self.loss)
        elif self.optimizer_type == "momentum":
            dnn_part_optimizer = tf.train.MomentumOptimizer(learning_rate=self.deep_lr, momentum=0.95).minimize(
                self.loss, global_step=None, var_list=dnn_part_vars)
        else:
            dnn_part_optimizer = tf.train.AdamOptimizer(learning_rate=self.deep_lr, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss, global_step=None,
                                                                               var_list=dnn_part_vars)

        self.optimizer = tf.group(wide_part_optimizer, dnn_part_optimizer)

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

    def slice_by_field(self, single_sample):
        sample = []
        for ss in single_sample:
            index = 0
            tmp = []
            for feat_num in self.field_lens:
                tmp_ = []
                for i in range(feat_num):
                    tmp_.append(ss[index + i])
                tmp.append(tmp_)
                index += feat_num
            sample.append(tmp)
        return sample

    def get_batch_data(self, train_x, train_y):
        start = 0
        while start < train_x.shape[0]:
            end = start + self.batch_size
            single_sample = train_x[start: end, :]
            x_batch = self.slice_by_field(single_sample)
            y_batch = train_y[start: end]

            start = end
            x_batch_shuffle, y_batch_shuffle = self.shuffle_batch(x_batch, y_batch)

            yield x_batch_shuffle, y_batch_shuffle


    def get_feed_dict(self, x_batch, y_batch, is_train=True):
        feed_dict = {self.y: y_batch, self.is_train: is_train}
        for i in range(len(self.field_lens)):
            feed_dict[self.x[i]] = [[x for x in sample[i]] for sample in x_batch]
        return feed_dict

    def fit(self, train_x, valid_x, train_y, valid_y, best_metric=0.8):
        step = 0
        stop = False
        for epoch in range(self.epoch):
            print("EPOCH: {}".format(epoch + 1))
            for x_batch, y_batch in self.get_batch_data(train_x, train_y):
                feed_dict = self.get_feed_dict(x_batch, y_batch)
                self.sess.run(self.optimizer, feed_dict=feed_dict)
                if step % self.step_print == 0:

                    train_pred, train_pred_label, train_loss = self.sess.run([self.out, self.pred, self.loss],
                                                                         feed_dict=feed_dict)
                    train_acc = eval_acc(train_pred_label, y_batch)
                    val_feed_dict = self.get_feed_dict(self.slice_by_field(valid_x), valid_y, is_train=False)

                    val_pred, val_pred_label, val_loss = self.sess.run([self.out, self.pred, self.loss],
                                                                   feed_dict=val_feed_dict)
                    val_acc, val_auc = eval_acc(val_pred_label, valid_y), eval_auc(val_pred, valid_y)

                    if val_auc > best_metric:
                        stop = True
                        break
                    msg = 'Iter: {0:>6}, Train acc: {1:>6.4}, Train loss: {4:6.6}, Val acc: {2:>6.4}, Val auc: {3:>6.4}, Val loss: {4:6.6},'
                    print(msg.format(step, train_acc, train_loss, val_acc, val_auc, val_loss))
                    # print(step, train_acc)
                step += 1
            if stop:
                break

    def predict(self, test_x, test_y=None):
        feed_dict = self.get_feed_dict(self.slice_by_field(test_x), test_y, is_train=False)
        test_pred, test_pred_label, test_loss = self.sess.run([self.out, self.pred, self.loss],
                                                          feed_dict=feed_dict)
        if test_y.any():
            test_acc, test_auc = eval_acc(test_pred_label, test_y), eval_auc(test_pred, test_y)
            msg = 'Test acc: {0:>6.4}, Test auc: {1:>6.4}'
            print(msg.format(test_acc, test_auc))
        return test_pred


EPOCH = 10
STEP_PRINT = 200
STOP_STEP = 2000

WIDE_LEARNING_RATE = 1e-3
DEEP_LEARNING_RATE = 1e-4
L1_REG = 0.01

BATCH_SIZE = 32
VEC_DIM = 20
DNN_LAYERS = [256, 128, 128, 1]

def run():
    train_path = '../../data/ctr/train.csv'
    test_path = '../../data/ctr/test.csv'
    print("load data...")
    train_x, valid_x, train_y, valid_y, filed_lens = load_data(train_path)

    print("build model...")
    config = {
        'vec_dim': VEC_DIM,
        'field_lens': filed_lens,
        'dnn_layers': DNN_LAYERS,
        'wide_lr': WIDE_LEARNING_RATE,
        'l1_reg': L1_REG,
        'deep_lr': DEEP_LEARNING_RATE,
        'epoch': 10,
        'batch_size': 32
    }
    model = WideDeep(**config)
    print("lets train...")
    model.fit(train_x, valid_x, train_y, valid_y)
    # model.saver.save(sess=model.sess, save_path='./ckpt/best')
    # model.saver.restore(sess=model.sess, save_path='./ckpt/best')
    test_pred = model.predict(valid_x, valid_y)
    print("====== let's test =====")
    print(test_pred)

print("lets run...")
run()
# Test acc: 0.9635, Test auc: 0.5617

