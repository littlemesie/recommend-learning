# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/5/19 19:42
@summary:
"""
import tensorflow as tf
import os
import time

# tf.config.set_soft_device_placement(False)
# tf.debugging.set_log_device_placement(False)
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import *
from esmm.esmm_model import CTCVRNet

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"


def train_model(cate_feature_dict, user_cate_feature_dict, item_cate_feature_dict, train_data, val_data):
    """
    model train and save as tf serving model
    :param cate_feature_dict: dict, categorical feature for data
    :param user_cate_feature_dict: dict, user categorical feature
    :param item_cate_feature_dict: dict, item categorical feature
    :param train_data: DataFrame, training data
    :param val_data: DataFrame, valdation data
    :return: None
    """
    ctcvr = CTCVRNet(cate_feature_dict)
    ctcvr_model = ctcvr.build(user_cate_feature_dict, item_cate_feature_dict)
    opt = optimizers.Adam(learning_rate=0.003, decay=0.0001)
    ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
                        metrics=[tf.keras.metrics.AUC()])

    # keras model save path
    # filepath = "esmm_best.h5"
    #
    # # call back function
    # checkpoint = ModelCheckpoint(
    #     filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # reduce_lr = ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
    # earlystopping = EarlyStopping(
    #     monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='auto')
    # callbacks = [checkpoint, reduce_lr, earlystopping]

    # load data
    ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train, \
    ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train, \
    cvr_item_numerical_feature_train, cvr_item_cate_feature_train, ctr_target_train, cvr_target_train = train_data

    ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val, \
    ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val, \
    cvr_item_numerical_feature_val, cvr_item_cate_feature_val, ctr_target_val, cvr_target_val = val_data

    # model train
    ctcvr_model.fit([ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train,
                     ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
                     cvr_item_numerical_feature_train,
                     cvr_item_cate_feature_train], [ctr_target_train, cvr_target_train], batch_size=256, epochs=50,
                    validation_data=(
                        [ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val,
                         ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
                         cvr_item_numerical_feature_val,
                         cvr_item_cate_feature_val], [ctr_target_val, cvr_target_val]), #callbacks=callbacks,
                    verbose=0,
                    shuffle=True)

    # load model and save as tf_serving model
    # saved_model_path = './esmm/{}'.format(int(time.time()))
    # ctcvr_model = tf.keras.models.load_model('esmm_best.h5')
    # tf.saved_model.save(ctcvr_model, saved_model_path)
