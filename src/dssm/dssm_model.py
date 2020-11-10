import time
import tensorflow as tf

class Dssm:
    def __init__(self, embedding_size=8, learning_rate=0.001, optimizer="adam", loss_type="logloss"):
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_type = loss_type

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()


