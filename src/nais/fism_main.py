from time import time
import numpy as np
import tensorflow as tf
from nais.data_reader import Dataset
from nais.fism import FISM
from nais import batch_gen as batch
from nais import evaluate

def training(model, dataset, epochs, num_negatives):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # initialize for training batches
        batch_begin = time()
        batches = batch.shuffle(dataset, model.batch_choice, num_negatives)
        batch_time = time() - batch_begin

        num_batch = len(batches[1])
        batch_index = list(range(num_batch))

        # initialize the evaluation feed_dicts
        testDict = evaluate.init_evaluate_model(model, sess, dataset.test_list, dataset.negative_list,
                                                dataset.train_list)

        # train by epoch
        for epoch_count in range(epochs):

            train_begin = time()
            training_batch(batch_index, model, sess, batches)
            train_time = time() - train_begin

            if epoch_count % model.verbose == 0:

                if model.train_loss:
                    loss_begin = time()
                    train_loss = training_loss(model, sess, batches)
                    loss_time = time() - loss_begin
                else:
                    loss_time, train_loss = 0, 0

                eval_begin = time()
                (hits, ndcgs, losses) = evaluate.eval(model, sess, dataset.test_list, dataset.negative_list, testDict)
                hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                eval_time = time() - eval_begin

                print(
                    "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))

            batch_begin = time()
            batches = batch.shuffle(dataset, model.batch_choice, num_negatives)
            np.random.shuffle(batch_index)
            batch_time = time() - batch_begin


def training_batch(batch_index, model, sess, batches):
    for index in batch_index:
        user_input, num_idx, item_input, labels = batch.batch_gen(batches, index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None],
                     model.item_input: item_input[:, None],
                     model.labels: labels[:, None]}
        sess.run(model.optimizer, feed_dict)


def training_loss(model, sess, batches):
    train_loss = 0.0
    num_batch = len(batches[1])
    for index in range(num_batch):
        user_input, num_idx, item_input, labels = batch.batch_gen(batches, index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None],
                     model.item_input: item_input[:, None], model.labels: labels[:, None]}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch


if __name__ == '__main__':


    data = Dataset()
    config = {
        "user_size": data.user_size,
        "item_size": data.item_size,
        "lr": 0.01,
        "embed_size": 16,
        "alpha": 0,
        "verbose": 1,
        "regs": [1e-7, 1e-7],
        "batch_choice": "user"
    }
    model = FISM(**config)

    training(model, data, epochs=10, num_negatives=4)
