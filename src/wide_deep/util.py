import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
def load_data(train_path):
    need_cols = ["ps_reg_01", "ps_reg_02", "ps_reg_03",  "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15"]

    data = pd.read_csv(train_path)
    # train_data = data[need_cols]
    train_data = data.drop(['id', 'target'], axis=1)
    filed_lens = get_filed(train_data)

    train_label = data["target"]

    train_x, valid_x, train_y, valid_y = train_test_split(train_data, train_label, test_size=0.2)


    return np.array(train_x), np.array(valid_x), np.array(train_y), np.array(valid_y), filed_lens

def get_filed(train_data):
    """
    返回{'ps_reg_01': 0, 'ps_reg_02': 1, 'ps_reg_03': 2, 'ps_car_12': 3, 'ps_car_13': 4, 'ps_car_14': 5, 'ps_car_15': 6}
    """
    fileds = train_data.columns.values
    filed_lens = []
    for filed in fileds:
        filed_lens.append(1)

    return filed_lens
def slice_by_field (single_sample, field_lens):
    sample = []
    for ss in single_sample:
        index = 0
        tmp = []
        for feat_num in field_lens:
            tmp_ = []
            for i in range(feat_num):
                tmp_.append(ss[index + i])
            tmp.append(tmp_)
            index += feat_num
        sample.append(tmp)
    return sample

# train_path = '../../data/ctr/train.csv'

# train_x, valid_x, train_y, valid_y, filed_lens = load_data(train_path)
