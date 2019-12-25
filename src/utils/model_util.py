# -*- coding:utf-8 -*-
import os
import pickle
from sklearn.externals import joblib

def save_model(model_path, data):
    """
    保存模型
    @param model_path:保存路径
    @param data:要保存的数据
    """
    parent_path = model_path[: model_path.rfind("/")]

    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    with open(model_path, "wb") as f:
        pickle.dump(data, f)


def load_model(model_path):
    """载入模型"""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data

def joblib_save(model_path, data):
    """"""
    parent_path = model_path[: model_path.rfind("/")]
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    joblib.dump(data, model_path)

def joblib_load(model_path):
    """"""
    model = joblib.load(model_path)
    return model