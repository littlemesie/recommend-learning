import numpy as np
import pandas as pd
import tensorflow as tf

class NAIS:

    def __init__(self, user_size, item_size, pretrain=1, lr=0.01, embed_size=16, weight_size=16,
                 alpha=0, beta=0.5, data_alpha=0, verbose=1, regs=[1e-7,1e-7,1e-5], batch_choice="user",
                 activation=0, algorithm=0, train_loss=1):
        """
        pretrain: 0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.
        item_size: item size.
        lr: Learning rate.
        embed_size: Embedding size.
        weight_size: weight size.
        alpha: Index of coefficient of embedding vector
        beta: Index of coefficient of sum of exp(A)
        data_alpha: Index of coefficient of embedding vector
        verbose: Interval of evaluation.
        activation: Activation for ReLU, sigmoid, tanh.
        algorithm: 0 for NAIS_prod, 1 for NAIS_concat
        batch_choice: user: generate batches by user, fixed:batch_size: generate batches by batch size
        regs: Regularization for user and item embeddings.
        train_loss: Caculate training loss or nor
        """
        self.pretrain = pretrain
        self.item_size = item_size
        self.learning_rate = lr
        self.embedding_size = embed_size
        self.weight_size = weight_size
        self.alpha = alpha
        self.beta = beta
        self.data_alpha = data_alpha
        self.verbose = verbose
        self.activation = activation
        self.algorithm = algorithm
        self.batch_choice = batch_choice

        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.eta_bilinear = regs[2]
        self.train_loss = train_loss
        self.build_graph()

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])  # the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])  # the ground truth

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            trainable_flag = (self.pretrain != 2)
            self.c1 = tf.Variable(
                tf.truncated_normal(shape=[self.item_size, self.embedding_size], mean=0.0, stddev=0.01), name='c1',
                dtype=tf.float32, trainable=trainable_flag)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.item_size, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32, trainable=trainable_flag)
            self.bias = tf.Variable(tf.zeros(self.item_size), name='bias', trainable=trainable_flag)

            # Variables for attention
            if self.algorithm == 0:
                self.W = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.weight_size], mean=0.0,
                                                         stddev=tf.sqrt(
                                                             tf.div(2.0, self.weight_size + self.embedding_size))),
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:
                self.W = tf.Variable(tf.truncated_normal(shape=[2 * self.embedding_size, self.weight_size], mean=0.0,
                                                         stddev=tf.sqrt(tf.div(2.0, self.weight_size + (
                                                                     2 * self.embedding_size)))),
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.weight_size + self.embedding_size))), name='Bias_for_MLP', dtype=tf.float32,
                                 trainable=True)
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)

    def _attention_MLP(self, q_):
        with tf.name_scope("attention_MLP"):
            b = tf.shape(q_)[0]
            n = tf.shape(q_)[1]
            r = (self.algorithm + 1) * self.embedding_size

            MLP_output = tf.matmul(tf.reshape(q_, [-1, r]), self.W) + self.b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            if self.activation == 0:
                MLP_output = tf.nn.relu(MLP_output)
            elif self.activation == 1:
                MLP_output = tf.nn.sigmoid(MLP_output)
            elif self.activation == 2:
                MLP_output = tf.nn.tanh(MLP_output)

            A_ = tf.reshape(tf.matmul(MLP_output, self.h), [b, n])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            num_idx = tf.reduce_sum(self.num_idx, 1)
            mask_mat = tf.sequence_mask(num_idx, maxlen=n, dtype=tf.float32)  # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keep_dims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)

            return tf.reduce_sum(A * self.embedding_q_, 1)

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, self.user_input)  # (b, n, e)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_input)  # (b, 1, e)

            if self.algorithm == 0:
                self.embedding_p = self._attention_MLP(self.embedding_q_ * self.embedding_q)
            else:
                n = tf.shape(self.user_input)[1]
                self.embedding_p = self._attention_MLP(
                    tf.concat([self.embedding_q_, tf.tile(self.embedding_q, tf.stack([1, n, 1]))], 2))

            self.embedding_q = tf.reduce_sum(self.embedding_q, 1)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(self.num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            self.output = tf.sigmoid(
                self.coeff * tf.expand_dims(tf.reduce_sum(self.embedding_p * self.embedding_q, 1), 1) + self.bias_i)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                        self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_)) + \
                        self.eta_bilinear * tf.reduce_sum(tf.square(self.W))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()