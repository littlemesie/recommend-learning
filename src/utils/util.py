# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2019/5/8 22:56
@summary:
"""
import os
import pickle
import time
import numpy as np


def save_file(filepath, data):
    """
    保存数据
    @param filepath:    保存路径
    @param data:    要保存的数据
    """
    parent_path = filepath[: filepath.rfind("/")]

    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_file(filepath):
    """载入二进制数据"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def open_text(filename,skip_row = 0):
    """打开文本文件

    :param filename: str
        文件名
    :param skip_row: int
         需要跳过的行数
    :return generator
        生成每一行的文本
    """
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip_row:
                continue
            yield line

def run_time(fn):
    """Decorator for calculating function runtime.Depending on the length of time,
    seconds, milliseconds, microseconds or nanoseconds are used.
    Arguments:
        fn {function}
    Returns:
        function
    """

    def inner():
        start = time.time()
        fn()
        ret = time.time() - start
        if ret < 1e-6:
            unit = "ns"
            ret *= 1e9
        elif ret < 1e-3:
            unit = "us"
            ret *= 1e6
        elif ret < 1:
            unit = "ms"
            ret *= 1e3
        else:
            unit = "s"
        print("Total run time is %.1f %s\n" % (ret, unit))
    return inner

def shuffle_in_unison_scary(a, b, c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)
