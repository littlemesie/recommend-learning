import tensorflow as tf

class FM(object):
    def __init__(self, vec_dim, feat_num, lr, lamda):
        self.vec_dim = vec_dim
        self.feat_num = feat_num
        self.lr = lr
        self.lamda = lamda

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
        self.y_hat = tf.nn.sigmoid(self.y_logits)
        self.pred_label = tf.cast(self.y_hat > 0.5, tf.int32)
        self.loss = -tf.reduce_mean(self.y*tf.log(self.y_hat+1e-8) + (1-self.y)*tf.log(1-self.y_hat+1e-8))
        self.reg_loss = self.lamda*(tf.reduce_mean(tf.nn.l2_loss(self.W)) + tf.reduce_mean(tf.nn.l2_loss(self.V)))
        self.total_loss = self.loss + self.reg_loss

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)