# -*- coding:utf-8 -*-
import time
import numpy as np
import random
import tensorflow as tf
from ncf import util

class NCF(object):
    def __init__(self, user_size, item_size, label_size, user_feature_size, item_feature_size, embed_size, loss_func,
                 optimizer_type, lr=0.001, initializer='Xavier', batch_size=128, epochs=10,
                 activation_func='ReLU', regularizer_rate=0, topK=10, dropout=0, step_print=200):
        """
        Important Arguments.

        embed_size: The final embedding size for users and items.
        optim: The optimization method chosen in this model.
        initializer: The initialization method.
        loss_func: Loss function, we choose the cross entropy.
        regularizer_rate: L2 is chosen, this represents the L2 rate.
        iterator: Input dataset.
        topk: For evaluation, computing the topk items.
        """
        self.user_size = user_size
        self.item_size = item_size
        self.label_size = label_size
        self.user_feature_size = user_feature_size
        self.item_feature_size = item_feature_size
        self.embed_size = embed_size
        self.lr = lr
        self.initializer = initializer
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.activation_func = activation_func
        self.regularizer_rate = regularizer_rate
        self.optimizer_type = optimizer_type
        self.topk = topK
        self.dropout = dropout
        self.epochs = epochs
        self.step_print = step_print
        self._build()


    def _build(self):
        self.add_input()
        self.inference()
        self.create_model()
        # self.eval()
        # self.summary()

        # init
        # self.saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def add_input(self):
        self.user_onehot = tf.zeros([self.label_size, self.user_feature_size], dtype=tf.float32, name='user_onehot')
        self.item_onehot = tf.zeros([self.label_size, self.item_feature_size], dtype=tf.float32, name='item_onehot')
        self.labels = tf.ones([self.label_size], dtype=tf.float32)
        # self.user_onehot = tf.placeholder(tf.float32, shape=[None, self.user_feature_size], name='user_onehot')
        # self.item_onehot = tf.placeholder(tf.float32, shape=[None, self.item_feature_size], name='item_onehot')
        # self.labels = tf.placeholder(tf.float32, shape=[None], name='input_labels')

    def inference(self):
        """ Initialize important settings """
        self.regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_rate)

        if self.initializer == 'Normal':
            self.initializer = tf.truncated_normal_initializer(stddev=0.01)
        elif self.initializer == 'Xavier_Normal':
            self.initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.initializer = tf.glorot_uniform_initializer()

        if self.activation_func == 'ReLU':
            self.activation_func = tf.nn.relu
        elif self.activation_func == 'Leaky_ReLU':
            self.activation_func = tf.nn.leaky_relu
        elif self.activation_func == 'ELU':
            self.activation_func = tf.nn.elu

        if self.loss_func == 'cross_entropy':
            self.loss_func = tf.nn.sigmoid_cross_entropy_with_logits

        if self.optimizer_type == 'SGD':
            self.optim = tf.train.GradientDescentOptimizer(self.lr, name='SGD')
        elif self.optimizer_type == 'RMSProp':
            self.optim = tf.train.RMSPropOptimizer(self.lr, decay=0.9, momentum=0.0, name='RMSProp')
        elif self.optimizer_type == 'Adam':
            self.optim = tf.train.AdamOptimizer(self.lr, name='Adam')


    def create_model(self):
        with tf.name_scope('embed'):
            self.user_embed_GMF = tf.layers.dense(inputs=self.user_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='user_embed_GMF')

            self.item_embed_GMF = tf.layers.dense(inputs=self.item_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='item_embed_GMF')

            self.user_embed_MLP = tf.layers.dense(inputs=self.user_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='user_embed_MLP')
            self.item_embed_MLP = tf.layers.dense(inputs=self.item_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='item_embed_MLP')



        with tf.name_scope("GMF"):
            self.GMF = tf.multiply(self.user_embed_GMF, self.item_embed_GMF, name='GMF')

        with tf.name_scope("MLP"):
            self.interaction = tf.concat([self.user_embed_MLP, self.item_embed_MLP],
                                         axis=-1, name='interaction')

            self.layer1_MLP = tf.layers.dense(inputs=self.interaction,
                                              units=self.embed_size * 2,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer1_MLP')
            self.layer1_MLP = tf.layers.dropout(self.layer1_MLP, rate=self.dropout)

            self.layer2_MLP = tf.layers.dense(inputs=self.layer1_MLP,
                                              units=self.embed_size,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer2_MLP')
            self.layer2_MLP = tf.layers.dropout(self.layer2_MLP, rate=self.dropout)

            self.layer3_MLP = tf.layers.dense(inputs=self.layer2_MLP,
                                              units=self.embed_size // 2,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer3_MLP')
            self.layer3_MLP = tf.layers.dropout(self.layer3_MLP, rate=self.dropout)

        with tf.name_scope('concatenation'):
            self.concatenation = tf.concat([self.GMF, self.layer3_MLP], axis=-1, name='concatenation')
            # self.concatenation = self.GMF


            self.logits = tf.layers.dense(inputs= self.concatenation,
                                          units=1,
                                          activation=None,
                                          kernel_initializer=self.initializer,
                                          kernel_regularizer=self.regularizer,
                                          name='predict')

            self.logits_dense = tf.reshape(self.logits, [-1])

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(self.loss_func(labels=self.labels, logits=self.logits_dense, name='loss'))
            # self.loss = tf.reduce_mean(self.loss_func(self.label, self.logits), name='loss')

        with tf.name_scope("optimzation"):
            self.optimizer = self.optim.minimize(self.loss)


    def eval(self):
        with tf.name_scope("evaluation"):
            self.item_replica = self.item_ids
            _, self.indice = tf.nn.top_k(tf.sigmoid(self.logits_dense), self.topk)


    def summary(self):
        """ Create summaries to write on tensorboard. """
        self.writer = tf.summary.FileWriter('./graphs/NCF', tf.get_default_graph())
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def shuffle_batch(self, user_ids, item_ids, feature_user, feature_item, labels):
        assert len(user_ids) == len(item_ids) or len(user_ids) == len(feature_user) or \
               len(user_ids) == len(feature_item) or len(user_ids) == len(labels)
        length = len(user_ids)
        index = [i for i in range(length)]
        random.shuffle(index)
        user_ids_shuffle = [user_ids[i] for i in index]
        item_ids_shuffle = [item_ids[i] for i in index]
        feature_user_shuffle = [feature_user[i] for i in index]
        feature_item_shuffle = [feature_item[i] for i in index]
        labels_shuffle = [labels[i] for i in index]
        return user_ids_shuffle, item_ids_shuffle, feature_user_shuffle, feature_item_shuffle, labels_shuffle

    def get_batch_data(self, data, batch_index):
        start = 0
        while start < len(data['user_ids']):
            end = start + self.batch_size
            user_ids_batch = data['user_ids'][start: end]
            item_ids_batch = data['item_ids'][start: end]
            feature_user_batch = np.array(data['feature_user'])[start: end, :]
            feature_item_batch = np.array(data['feature_item'])[start: end, :]
            labels_batch = data['labels'][start: end]
            start = end
            # user_ids_shuffle, item_ids_shuffle, feature_user_shuffle, feature_item_shuffle, labels_shuffle = \
            #     self.shuffle_batch(user_ids_batch, item_ids_batch, feature_user_batch, feature_item_batch, labels_batch)

            # yield user_ids_shuffle, item_ids_shuffle, feature_user_shuffle, feature_item_shuffle, labels_shuffle
            yield user_ids_batch, item_ids_batch, feature_user_batch, feature_item_batch, labels_batch

    def get_data(self, iterator):
        sample = iterator.get_next()
        self.user_ids= tf.cast(sample['user_ids'], tf.float32)
        self.item_ids = tf.cast(sample['item_ids'], tf.float32)
        self.user_onehot = tf.cast(sample['feature_user'], tf.float32)
        self.item_onehot = tf.cast(sample['feature_item'], tf.float32)
        self.labels = tf.cast(sample['labels'], tf.float32)

    def fit(self, data):
        """train model"""
        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        for epoch in range(self.epochs):
            print("EPOCH: {}".format(epoch + 1))
            self.sess.run(iterator.make_initializer(data))
            self.get_data(iterator)
            train_loss, optim = self.sess.run([self.loss, self.optimizer])
            print("train_loss: {}".format(train_loss))




    def predict(self):
        """predict result"""
        indice, item = self.sess.run([self.indice, self.item_replica])
        prediction = np.take(item, indice)
        user = self.sess.run(self.user_ids)
        return prediction, item, user[0]



if __name__ == '__main__':
    base, user_ids, item_ids, user_dict, item_dict, user_bought, user_negative, user_feature_size, item_feature_size = util.load_data()
    train_data, data_dict = util.add_negative(base, user_dict, item_dict, user_negative, 4, 128, is_training=True)

    config = {
        "user_size": len(user_ids),
        "item_size": len(item_ids),
        "label_size": len(data_dict['labels']),
        "user_feature_size": user_feature_size,
        "item_feature_size": item_feature_size,
        'batch_size': 128,
        # 'negative_num': 4,
        'embed_size': 16,
        "epochs": 10,
        'topK': 10,
        'optimizer_type': 'Adam',
        'initializer': 'Xavier',
        'loss_func': 'cross_entropy',
        'activation_func': 'ReLU',
        'regularizer_rate': 0.0,
        'lr': 0.001,
        'dropout': 0.0
    }


    # build model
    print("build model")
    model = NCF(**config)
    model.fit(train_data)
    # print(model.logits_dense)
