from time import time
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics
from utils.util import shuffle_in_unison_scary

class FM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size, embedding_size=8, dropout_fm=[1.0, 1.0], epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer="adam", random_seed=2020, loss_type="logloss", eval_metric="auc",
                 l2_reg=0.0, step_print=200):
        assert eval_metric in ['auc', 'acc'], "eval_metric error"
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.dropout_fm = dropout_fm
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.step_print = step_print
        self.train_result, self.valid_result = [], []

        self._build_graph()

    def _build_graph(self):
        self.add_input()
        self.inference()

    def add_input(self):
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')

        self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
        self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')

    def inference(self):

        with tf.variable_scope('embeddings_weights'):
            self.weights = dict()
            self.weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings'
            )
            self.weights['feature_bias'] = tf.Variable(
                tf.random_normal([self.feature_size, 1], 0.0, 1.0), name='feature_bias')

            input_size = self.field_size + self.embedding_size
            glorot = np.sqrt(2.0 / (input_size + 1))
            self.weights['concat_projection'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
            self.weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        with tf.variable_scope('embeddings'):
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

        with tf.variable_scope('linear_part'):
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])

        with tf.variable_scope('second_part'):
            # sum-square-part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * k
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # squre-sum-part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])

        concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
        self.y_logits = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])

        self.out = tf.nn.sigmoid(self.y_logits)
        self.pred = tf.cast(self.out > 0.5, tf.int32)

        # loss
        if self.loss_type == "logloss":
            logs = tf.losses.log_loss(self.label, self.out)
            self.loss = tf.reduce_mean(logs)
        elif self.loss_type == "mse":
            self.loss = tf.losses.mean_squared_error(self.label, self.out)

        # optimizer
        if self.optimizer_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)
        elif self.optimizer_type == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        elif self.optimizer_type == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                self.loss)

        # init
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def evaluate(self, Xi, Xv, y):
        y_pred = self.predict(Xi, Xv)
        if self.eval_metric == 'auc':
            return metrics.roc_auc_score(y, y_pred)
        else:
            return metrics.accuracy_score(y, y_pred)

    def predict(self, Xi, Xv):
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.train_phase: False}

            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred


    def fit_on_batch(self, Xi, Xv,y):
        """train each batch data"""
        feed_dict = {
            self.feat_index: Xi,
            self.feat_value: Xv,
            self.label: y,
            self.dropout_keep_fm: self.dropout_fm,
            self.train_phase: True
        }

        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)

        return loss

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None, y_valid=None):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :return: None
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                loss = self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                print("epoch=%d, loss=%.4f" % (epoch + 1, loss))

            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
                print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]" % (epoch + 1, train_result, valid_result, time() - t1))
            else:
                print("[%d] train-result=%.4f [%.1f s]" % (epoch + 1, train_result, time() - t1))

