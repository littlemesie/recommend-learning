# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from itertools import combinations
from collections import deque

class BatchGenerator(object):

    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.ix = 0
        self.buffer = deque([])
        self._finish = False

    def next(self):
        if self._finish:
            return 'No data!'

        while len(self.buffer) < self.batch_size:
            items_list = self.data[self.ix]
            self.buffer.extend(combinations(items_list, 2))

            if self.ix == len(self.data) - 1:
                self._finish = True
                self.ix = 0
            else:
                self.ix += 1
        d = [self.buffer.popleft() for _ in range(self.batch_size)]

        d = np.array([[i[0], i[1]] for i in d])
        batch = d[:, 0]
        labels = d[:, 1]

        return batch, labels

    @property
    def finish(self):
        return self._finish

    def resume(self):

        self.ix = 0
        self._finish = False

    @property
    def current_percentage(self):
        return (self.ix / self.data.shape[0]) * 100

class Item2Vec(object):
    def __init__(self, processor, embedding_size, num_negatives, learning_rate, batch_size, epochs=10, step=0, save_path=None):
        self.vocab_size = len(processor.item_list)
        self.embed_dim = embedding_size
        self.num_negatives = num_negatives
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_path = save_path
        self.step = step

        self.item_counts = processor.item_counts

        self._init_graphs()

    def _init_graphs(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.batch = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
            self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
            true_logits, sampled_logits = self.forward(self.batch, self.labels)
            self.loss = self.nce_loss(true_logits, sampled_logits)
            self.train_op = self.optimize(self.loss)

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def forward(self, batch, labels):

        init_width = 0.5 / self.embed_dim
        embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim], -init_width, init_width), name='word_embedding')
        self.embed = embed

        softmax_w = tf.Variable(tf.zeros([self.vocab_size, self.embed_dim]), name="softmax_weights")
        softmax_b = tf.Variable(tf.zeros([self.vocab_size]), name="softmax_bias")

        labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64), [self.batch_size, 1])

        # Negative sampling
        sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=self.num_negatives,
                unique=True,
                range_max=self.vocab_size,
                distortion=0.75,
                unigrams=self.item_counts)

        # Embeddings for examples: [batch_size, embed_dim]
        example_emb = tf.nn.embedding_lookup(embed, batch)

        # Weights for labels: [batch_size, embed_dim]
        true_w = tf.nn.embedding_lookup(softmax_w, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(softmax_b, labels)

        # Weights for sampled ids: [batch_size, embed_dim]
        sampled_w = tf.nn.embedding_lookup(softmax_w, sampled_ids)
        # Biases for sampled ids: [batch_size, 1]
        sampled_b = tf.nn.embedding_lookup(softmax_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        sampled_b_vec = tf.reshape(sampled_b, [self.num_negatives])
        sampled_logits = tf.matmul(example_emb, sampled_w, transpose_b=True) + sampled_b_vec

        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)

        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self.batch_size

        return nce_loss_tensor

    def optimize(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss)

        return train_op

    @property
    def embeddings(self):
        return self.embed.eval(session=self.sess)

    def get_factors(self, embeddings, item_index_data):
        embedding_size = embeddings.shape[1]

        factors = []

        for i in item_index_data:
            if len(i) == 0:
                factors.append(list(np.full(embedding_size, 0)))
            else:
                factors.append(list(np.mean(embeddings[i], axis=0)))

        return np.array(factors)

    def get_norms(self, e):
        norms = np.linalg.norm(e, axis=-1)
        norms[norms == 0] = 1e-10
        return norms

    def calu_similar(self, embeddings, queryid, norms, N):
        scores = embeddings.dot(embeddings[queryid]) / norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best] / norms[queryid]), key=lambda x: -x[1])

    def similar_items(self, embeddings, item_index_data, queyid, N=10):

        e = self.get_factors(embeddings, item_index_data)
        norms = self.get_norms(e)
        result = []
        for index, score in self.calu_similar(e, queyid, norms, N):
            result.append((index, score))

        return result

    def fit(self, item_index_data):
        avg_loss = 0

        for epoch in range(self.epochs):
            print("epoch: {}".format(epoch))
            generator = BatchGenerator(self.batch_size, item_index_data)
            while not generator.finish:
                batch, labels = generator.next()
                feed_dict = {self.batch: batch, self.labels: labels}
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                avg_loss += loss
                self.step += 1
                print("loss: ", '{:.9f}'.format(loss))

        self.embed = self.embeddings
            # print("Cost: ", '{:.9f}'.format(avg_loss))
        # self.saver.save(self.sess, os.path.join(self.save_path, "model.ckpt"),global_step=self.step)

    def predict(self, item_index_data, queyid,  top_N=10):
        """"""
        return self.similar_items(self.embed, item_index_data, queyid, N=top_N)


if __name__ == '__main__':
    from item2vec.utils.movie_process import MovieProcessor
    config = {
        "embedding_size": 8,
        "num_negatives": 30,
        "learning_rate": 0.5,
        "batch_size": 64,
        "epochs": 10,
        "step": 0,
        "save_path": "result/",
    }
    processor = MovieProcessor()
    config['processor'] = processor
    model = Item2Vec(**config)
    model.fit(processor.item_index_data)
    result = model.predict(processor.item_index_data, 10)
    print(result)
    # generator = BatchGenerator(32, processor.item_index_data)
    # while not generator.finish:
    #     batch, labels = generator.next()
    #     print(batch, labels)
