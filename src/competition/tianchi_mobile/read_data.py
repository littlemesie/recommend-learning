# -*- coding:utf-8 -*-
import time
import numpy as np
import pandas as pd

base_path = "/Volumes/d/taobao/"

def load_file(filename):
    """读文件，返回文件的每一行"""
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            yield line.strip('\r\n')

def analyze_data():
    """
    分析数据
    10000个用户,
    click, collect,add-to-cart and payment 的数量：11550581 242556 343564 120205
    310582商品
    991个商品类别
    """
    # trian_user = pd.read_csv(base_path + "tianchi_mobile_recommend_train_user.csv")
    # uid_count = trian_user["user_id"].unique()
    # print(len(uid_count))
    trian_item = pd.read_csv(base_path + "tianchi_mobile_recommend_train_item.csv")
    # item_count = trian_item["item_id"].unique()
    item_category_count = trian_item["item_category"].unique()
    print(len(item_category_count))

    # click_count = 0
    # collect_count = 0
    # add_cart_count = 0
    # pay_count = 0
    # for behavior_type in trian_user['behavior_type']:
    #     if behavior_type == 1:
    #         click_count += 1
    #     elif behavior_type == 2:
    #         collect_count += 1
    #     elif behavior_type == 3:
    #         add_cart_count += 1
    #     elif behavior_type == 4:
    #         pay_count += 1
    # print(click_count, collect_count, add_cart_count, pay_count)

def split_data_v1(train_rate=1):
    """切分数据"""
    date = time.strftime("%Y-%m-%d", time.localtime())
    path = base_path + "tianchi_mobile_recommend_train_user.csv"
    for line in load_file(path):
        arr = line.split(",")

    # user_df = pd.read_csv(base_path + "tianchi_mobile_recommend_train_user.csv")
    # user_df.drop(['user_geohash', 'time'], axis=1)
    # num = int(len(user_df) * train_rate)
    # train_user = user_df[:num]
    # test_user = user_df[num:]
    # return train_user, test_user

if __name__ == '__main__':
    # analyze_data()
    # split_data_v1()
    train_data_matrix = np.zeros((2, 3))
    train_data_matrix[0,0] = 1
    print(train_data_matrix)