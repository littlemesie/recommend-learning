import os
import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.neg_num = 10

        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, embedding_dim], trainable=True)
            self.mid_embeddings_bias = tf.get_variable("bias_lookup_table", [n_mid], initializer=tf.zeros_initializer(),
                                                       trainable=False)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

    def build_sampled_softmax_loss(self, item_emb, user_emb):
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias,
                                                              tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb,
                                                              self.neg_num * self.batch_size, self.n_mid))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.lr: inps[4]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.mask: inps[1]
        })
        return user_embs

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)


class Model_DNN(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_DNN, self).__init__(n_mid, embedding_dim, hidden_size,
                                        batch_size, seq_len, flag="DNN")

        masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1)

        self.item_his_eb_mean = tf.reduce_sum(self.item_his_eb, 1) / (
                    tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        self.user_eb = tf.layers.dense(self.item_his_eb_mean, hidden_size, activation=None)
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)
