import time
import tensorflow as tf
from dssm.data_process import MAX_SEQ_LEN
from dssm.data_process import Processor

class Dssm:
    def __init__(self, nwords, embedding_size=128, hidden_size_rnn=100, learning_rate=0.001, dropout=0.5,
                 optimizer_type="adam", loss_type="logloss", neg=4, query_bs=100, use_stack_rnn=False,
                 epochs=10, batch_size=32):
        self.nwords = nwords
        self.embedding_size = embedding_size
        self.hidden_size_rnn = hidden_size_rnn
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.dropout = dropout
        self.neg = neg
        self.query_bs = query_bs
        self.use_stack_rnn = use_stack_rnn
        self.epochs = epochs
        self.batch_size = batch_size
        self._init_graph()

    def _init_graph(self):
        self.add_input()
        self.inference()

    def add_input(self):
        # 预测时只用输入query即可，将其embedding为向量。
        self.query_batch = tf.placeholder(tf.int32, shape=[None, None], name='query_batch')
        self.doc_pos_batch = tf.placeholder(tf.int32, shape=[None, None], name='doc_positive_batch')
        self.doc_neg_batch = tf.placeholder(tf.int32, shape=[None, None], name='doc_negative_batch')
        self.query_seq_length = tf.placeholder(tf.int32, shape=[None], name='query_sequence_length')
        self.pos_seq_length = tf.placeholder(tf.int32, shape=[None], name='pos_seq_length')
        self.neg_seq_length = tf.placeholder(tf.int32, shape=[None], name='neg_sequence_length')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        self.on_train = tf.placeholder(tf.bool)

    def inference(self):
        with tf.name_scope('word_embeddings_layer'):
            _word_embedding = tf.get_variable(name="word_embedding_arr", dtype=tf.float32,
                                              shape=[self.nwords, self.embedding_size])
            self.query_embed = tf.nn.embedding_lookup(_word_embedding, self.query_batch, name='query_batch_embed')
            self.doc_pos_embed = tf.nn.embedding_lookup(_word_embedding, self.doc_pos_batch, name='doc_positive_embed')
            self.doc_neg_embed = tf.nn.embedding_lookup(_word_embedding, self.doc_neg_batch, name='doc_negative_embed')

        with tf.name_scope('RNN'):
            if self.use_stack_rnn:
                cell_fw = tf.contrib.rnn.GRUCell(self.hidden_size_rnn, reuse=tf.AUTO_REUSE)
                stacked_gru_fw = tf.contrib.rnn.MultiRNNCell([cell_fw], state_is_tuple=True)
                cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size_rnn, reuse=tf.AUTO_REUSE)
                stacked_gru_bw = tf.contrib.rnn.MultiRNNCell([cell_fw], state_is_tuple=True)
                (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(stacked_gru_fw, stacked_gru_bw)

            else:
                cell_fw = tf.contrib.rnn.GRUCell(self.hidden_size_rnn, reuse=tf.AUTO_REUSE)
                cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size_rnn, reuse=tf.AUTO_REUSE)
                # query
                (_, _), (query_output_fw, query_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                             self.query_embed,
                                                                                             sequence_length=self.query_seq_length,
                                                                                             dtype=tf.float32)
                query_rnn_output = tf.concat([query_output_fw, query_output_bw], axis=-1)
                query_rnn_output = tf.nn.dropout(query_rnn_output, self.dropout)
                # doc_pos
                (_, _), (doc_pos_output_fw, doc_pos_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                 self.doc_pos_embed,
                                                                                                 sequence_length=self.pos_seq_length,
                                                                                                 dtype=tf.float32)
                doc_pos_rnn_output = tf.concat([doc_pos_output_fw, doc_pos_output_bw], axis=-1)
                doc_pos_rnn_output = tf.nn.dropout(doc_pos_rnn_output, self.dropout)
                # doc_neg
                (_, _), (doc_neg_output_fw, doc_neg_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                 self.doc_neg_embed,
                                                                                                 sequence_length=self.neg_seq_length,
                                                                                                 dtype=tf.float32)
                doc_neg_rnn_output = tf.concat([doc_neg_output_fw, doc_neg_output_bw], axis=-1)
                doc_neg_rnn_output = tf.nn.dropout(doc_neg_rnn_output, self.dropout)

        with tf.name_scope('Merge_Negative_Doc'):
            # 合并负样本，tile。
            # doc_y = tf.tile(doc_positive_y, [1, 1])
            doc_y = tf.tile(doc_pos_rnn_output, [1, 1])

            for i in range(self.neg):
                for j in range(self.query_bs):
                    # slice(input_, begin, size)切片API
                    # doc_y = tf.concat([doc_y, tf.slice(doc_negative_y, [j * NEG + i, 0], [1, -1])], 0)
                    doc_y = tf.concat([doc_y, tf.slice(doc_neg_rnn_output, [j * self.neg + i, 0], [1, -1])], 0)

        with tf.name_scope('Cosine_Similarity'):
            # Cosine similarity
            # query_norm = sqrt(sum(each x^2))
            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_rnn_output), 1, True)), [self.neg + 1, 1])
            # doc_norm = sqrt(sum(each x^2))
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

            prod = tf.reduce_sum(tf.multiply(tf.tile(query_rnn_output, [self.neg + 1, 1]), doc_y), 1, True)
            norm_prod = tf.multiply(query_norm, doc_norm)

            # cos_sim_raw = query * doc / (||query|| * ||doc||)
            cos_sim_raw = tf.truediv(prod, norm_prod)
            # gamma = 20
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [self.neg + 1, self.query_bs])) * 20

        # 转化为softmax概率矩阵。
        self.prob = tf.nn.softmax(cos_sim)
        # 只取第一列，即正样本列概率。
        self.hit_prob = tf.slice(self.prob, [0, 0], [-1, 1])

        # loss
        if self.loss_type == "logloss":
            self.loss = -tf.reduce_sum(tf.log(self.hit_prob))
            # logs = tf.losses.log_loss(self.label, self.hit_prob)
            # self.loss = tf.reduce_mean(logs)

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

    def get_batch(slef, data_map, batch_index):
        query_in = data_map['query'][batch_index * slef.query_bs:(batch_index + 1) * slef.query_bs]
        query_len = data_map['query_len'][batch_index * slef.query_bs:(batch_index + 1) * slef.query_bs]
        doc_positive_in = data_map['doc_pos'][batch_index * slef.query_bs:(batch_index + 1) * slef.query_bs]
        doc_positive_len = data_map['doc_pos_len'][batch_index * slef.query_bs:(batch_index + 1) * slef.query_bs]
        doc_negative_in = data_map['doc_neg'][batch_index * slef.query_bs * slef.neg:(batch_index + 1) * slef.query_bs * slef.neg]
        doc_negative_len = data_map['doc_neg_len'][batch_index * slef.query_bs * slef.neg:(batch_index + 1) * slef.query_bs * slef.neg]
        label = data_map['label'][batch_index * slef.query_bs:(batch_index + 1) * slef.query_bs]
        # query_in, doc_positive_in, doc_negative_in = pull_all(query_in, doc_positive_in, doc_negative_in)
        return query_in, doc_positive_in, doc_negative_in, query_len, doc_positive_len, doc_negative_len, label

    def feed_dict(self, data_set, batch_index, on_training):
        query_in, doc_positive_in, doc_negative_in, query_seq_len, pos_seq_len, neg_seq_len, label = self.get_batch(data_set,
                                                                                                         batch_index)
        query_len = len(query_in)
        query_seq_len = [MAX_SEQ_LEN] * query_len
        pos_seq_len = [MAX_SEQ_LEN] * query_len
        neg_seq_len = [MAX_SEQ_LEN] * query_len * self.neg
        feed_dict = {
            self.query_batch: query_in,
            self.doc_pos_batch: doc_positive_in,
            self.doc_neg_batch: doc_negative_in,
            self.query_seq_length: query_seq_len,
            self.neg_seq_length: neg_seq_len,
            self.pos_seq_length: pos_seq_len,
            self.on_train: on_training,
            self.label: label
        }
        return feed_dict

    def fit(self, data_train):
        """"""
        train_len = int(len(data_train['query']) / self.query_bs) - 1
        batch_indexs = [i for i in range(0, train_len, self.batch_size)]
        for epoch in range(self.epochs):
            start = time.time()
            for batch_index in batch_indexs:
                loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=self.feed_dict(data_train, batch_index, True))
                print("epoch=%d, loss=%.4f" % (epoch + 1, loss))
            end = time.time()

    def predict(self, data_test):
        """预测"""
        test_len = int(len(data_test['query']) / self.query_bs) - 1
        # batch_indexs = [i for i in range(0, test_len, self.batch_size)]
        batch_indexs = [i for i in range(test_len)]
        for batch_index in batch_indexs:
            prob_, loss_v = self.sess.run([self.prob, self.loss], feed_dict=self.feed_dict(data_test, batch_index, False))
            print(prob_)


if __name__ == '__main__':
    """"""
    vocab_path = '../../data/dssm/vocab.txt'
    file_train = '/Users/mesie/Downloads/oppo_search_round1/train_100000.txt'
    file_vali = '/Users/mesie/Downloads/oppo_search_round1/vali_10000.txt'
    p = Processor(vocab_path)
    data_train = p.get_data(file_train)
    data_test = p.get_data(file_vali)

    config = {
        'nwords': p.nwords,
        'embedding_size': 100,
        'hidden_size_rnn': 100,
        'learning_rate': 0.001,
        'dropout': 0.5,
        'optimizer_type': "adam",
        'loss_type': "logloss",
        'neg': 4,
        "query_bs": 100,
        "use_stack_rnn": False,
        'epochs': 10,
        'batch_size': 32,
    }
    model = Dssm(**config)
    model.fit(data_train)
    model.predict(data_test)
