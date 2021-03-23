import tensorflow as tf

class BPR:
    def __init__(self, user_size, item_size, embedding_size, regulation_rate=0.0001):
        self.user_size = user_size
        self.item_size = item_size
        self.embedding_size = embedding_size
        self.regulation_rate = regulation_rate

        self._build_graph()

    def _build_graph(self):
        self.add_input()
        self.inference()

    def add_input(self):
        self.user = tf.placeholder(tf.int32, [None], name='user')
        self.item_i = tf.placeholder(tf.int32, [None], name='item_i')
        self.item_j = tf.placeholder(tf.int32, [None], name='item_j')

    def inference(self):
        with tf.variable_scope('embeddings_weights'):
            self.user_emb_w = tf.get_variable("user_emb_w", [self.user_size + 1, self.embedding_size],
                                         initializer=tf.random_normal_initializer(0, 0.1))
            self.item_emb_w = tf.get_variable("item_emb_w", [self.item_size + 1, self.embedding_size],
                                         initializer=tf.random_normal_initializer(0, 0.1))

        with tf.variable_scope('embeddings'):
            user_emb = tf.nn.embedding_lookup(self.user_emb_w, self.user)
            item_i_emb = tf.nn.embedding_lookup(self.item_emb_w, self.item_i)
            item_j_emb = tf.nn.embedding_lookup(self.item_emb_w, self.item_j)

        x = tf.reduce_sum(tf.multiply(user_emb, (item_i_emb - item_j_emb)), 1, keep_dims=True)


        l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(user_emb, user_emb)),
            tf.reduce_sum(tf.multiply(item_i_emb, item_i_emb)),
            tf.reduce_sum(tf.multiply(item_j_emb, item_j_emb))
        ])

        self.loss = self.regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))

        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)