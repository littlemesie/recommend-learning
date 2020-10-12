# -*- coding: utf-8 -*-
import time
import os
import logging
import joblib
import pandas as pd

from item2vec.item2vec_model import Item2Vec
from item2vec.utils.process import ItemNameProcessor
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logging.info('Start ... ')

config = {
    "embedding_size": 32,
    "num_negatives": 100,
    "learning_rate": 0.5,
    "batch_size": 256,
    "step": 0,
    "save_path": "result/",
}


start_time = time.time()
data = pd.read_csv("../../data/item2vec/data.csv")

processor = ItemNameProcessor(data, name_col='name')
config['processor'] = processor
with tf.Graph().as_default(), tf.Session() as session:
    config['session'] = session
    model = Item2Vec(**config)

    for epoch in range(30):
        model.train() # Process one epoch
        model.evaluate('手機')

        if (epoch + 1) % 5 == 0:
            embeds = model.embeddings
            processor.print_similar(embeds, 9, N=10)

        print('Finish {} epoch!'.format(epoch + 1))
        print('-'*10)

    embeds = model.embeddings
    processor.print_similar(embeds, 4839, N=10)
    processor.print_similar(embeds, 500, N=10)

    joblib.dump(embeds, os.path.join(config['save_path'], 'word_embeds.pkl'))
    joblib.dump(processor, os.path.join(config['save_path'], 'processor.pkl'))