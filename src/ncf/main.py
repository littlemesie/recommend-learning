# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2020/1/5 21:10
@summary:
"""

import os
import time
import numpy as np
import tensorflow as tf

from ncf import data_reader
from ncf import ncf_model
from ncf import metrics
from utils import metric



FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 128, 'size of mini-batch.')
tf.flags.DEFINE_integer('negative_num', 4, 'number of negative samples.')
tf.flags.DEFINE_integer('test_neg', 99, 'number of negative samples for test.')
tf.flags.DEFINE_integer('embedding_size', 16, 'the size for embedding user and item.')
tf.flags.DEFINE_integer('epochs', 10, 'the number of epochs.')
tf.flags.DEFINE_integer('topK', 10, 'topk for evaluation.')
tf.flags.DEFINE_string('optim', 'Adam', 'the optimization method.')
tf.flags.DEFINE_string('initializer', 'Xavier', 'the initializer method.')
tf.flags.DEFINE_string('loss_func', 'cross_entropy', 'the loss function.')
tf.flags.DEFINE_string('activation', 'ReLU', 'the activation function.')
tf.flags.DEFINE_string('model_dir', 'model/', 'the dir for saving model.')
tf.flags.DEFINE_float('regularizer', 0.0, 'the regularizer rate.')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate.')
tf.flags.DEFINE_float('dropout', 0.0, 'dropout rate.')



def train(train_data, test_data, user_size, item_size):
    with tf.Session() as sess:
        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)

        model = ncf_model.NCF(FLAGS.embedding_size, user_size, item_size, FLAGS.lr,
                              FLAGS.optim, FLAGS.initializer, FLAGS.loss_func, FLAGS.activation,
                              FLAGS.regularizer, iterator, FLAGS.topK, FLAGS.dropout, is_training=True)

        model.build()

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters.")
            sess.run(tf.global_variables_initializer())


        count = 0
        for epoch in range(FLAGS.epochs):
            sess.run(model.iterator.make_initializer(train_data))
            model.is_training = True
            model.get_data()
            start_time = time.time()

            try:
                while True:
                    model.step(sess, count)
                    count += 1
            except tf.errors.OutOfRangeError:
                print("Epoch %d training " % epoch + "Took: " + time.strftime("%H: %M: %S",
                                                                              time.gmtime(time.time() - start_time)))

            sess.run(model.iterator.make_initializer(test_data))
            model.is_training = False
            model.get_data()
            start_time = time.time()
            HR ,MRR, NDCG = [], [], []
            # prediction, label = model.step(sess, None)
            try:
                while True:
                    prediction, label, user = model.step(sess, None)
                    # print(prediction)
                    # print(label)
                    label = int(label[0])
                    HR.append(metrics.hit(label, prediction))
                    MRR.append(metrics.mrr(label, prediction))
                    NDCG.append(metrics.ndcg(label, prediction))
            except tf.errors.OutOfRangeError:
                hr = np.array(HR).mean()
                mrr = np.array(MRR).mean()
                ndcg = np.array(NDCG).mean()
                print("Epoch %d testing  " % epoch + "Took: " + time.strftime("%H: %M: %S",
                                                                              time.gmtime(time.time() - start_time)))
                print("HR is %.3f, MRR is %.3f, NDCG is %.3f" % (hr, mrr, ndcg))

        ################################## SAVE MODEL ################################
        checkpoint_path = os.path.join(FLAGS.model_dir, "NCF.ckpt")
        model.saver.save(sess, checkpoint_path)

def test(train_data, test_data, user_size, item_size, user_bought, item_set, item_popularity):
    """测试"""
    with tf.Session() as sess:
        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)

        model = ncf_model.NCF(FLAGS.embedding_size, user_size, item_size, FLAGS.lr,
                              FLAGS.optim, FLAGS.initializer, FLAGS.loss_func, FLAGS.activation,
                              FLAGS.regularizer, iterator, FLAGS.topK, FLAGS.dropout, is_training=True)

        model.build()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("No model!")

        sess.run(model.iterator.make_initializer(test_data))
        model.is_training = False
        model.get_data()
        start_time = time.time()
        HR, MRR, NDCG = [], [], []
        recommed_dict = {}
        test_user_items = {}
        try:
            while True:
                prediction, items, user = model.step(sess, None)
                recommed_dict.setdefault(user, prediction)
                test_user_items.setdefault(user, user_bought[user])
                label = int(items[0])
                HR.append(metrics.hit(label, prediction))
                MRR.append(metrics.mrr(label, prediction))
                NDCG.append(metrics.ndcg(label, prediction))
        except tf.errors.OutOfRangeError:
            hr = np.array(HR).mean()
            mrr = np.array(MRR).mean()
            ndcg = np.array(NDCG).mean()

            print("HR is %.3f, MRR is %.3f, NDCG is %.3f" % (hr, mrr, ndcg))

            precision = metric.precision(recommed_dict, test_user_items)
            recall = metric.recall(recommed_dict, test_user_items)
            coverage = metric.coverage(recommed_dict, item_set)
            popularity = metric.popularity(item_popularity, recommed_dict)
            print("precision is %.3f, recall is %.3f, coverage is %.3f, popularity is %.3f" %
                  (precision, recall, coverage, popularity))

def main():
    ((train_features, train_labels),
     (test_features, test_labels),
     (user_size, item_size),
     (user_bought, user_negative),
     (user_set, item_set)) = data_reader.load_data()


    train_data = data_reader.train_input_fn(train_features, train_labels, FLAGS.batch_size,
                                            user_negative, FLAGS.negative_num)

    test_data = data_reader.eval_input_fn(test_features, test_labels,
                                        user_negative, FLAGS.test_neg)

    # train(train_data, test_data, user_size, item_size)
    item_popularity = dict()
    for item in train_features['item']:
        if item in item_popularity:
            item_popularity[item] += 1
        else:
            item_popularity.setdefault(item, 1)
    test(train_data, test_data, user_size, item_size, user_bought, item_set, item_popularity)

if __name__ == '__main__':
    main()
    """
    GFM:
    HR is 0.735, MRR is 0.374, NDCG is 0.459
    precision is 0.251, recall is 0.015, coverage is 0.458, popularity is 4.858
    NFM:
    HR is 0.741, MRR is 0.375, NDCG is 0.461
    precision is 0.211, recall is 0.012, coverage is 0.605, popularity is 4.509
    """