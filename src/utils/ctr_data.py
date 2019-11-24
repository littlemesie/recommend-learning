import pandas as pd

"""
处理数据成libsvm和libsvm格式
"""
NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
    # feature engineering
    "missing_feat", "ps_car_13_x_ps_reg_03",
]

IGNORE_COLS = [
    "id",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]


def gen_feat_dict(df):
    """"""
    tc = 0
    feat_dict = {}
    dfc = df.copy()
    if 'target' in dfc.columns:
        dfc.drop(['target'], axis=1, inplace=True)

    for col in dfc.columns:
        if col in IGNORE_COLS:
            continue
        if col in NUMERIC_COLS:
            feat_dict[col] = tc
            tc += 1

        else:
            us = df[col].unique()
            # print(col, ':', us)
            feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
            tc += len(us)
    print(feat_dict)
    return feat_dict

def parse_libsvm(feat_dict, df):
    """"""
    dfc = df.copy()
    for col in dfc.columns:
        if col in IGNORE_COLS:
            dfc.drop(col, axis=1, inplace=True)
            continue
        if col == 'target':
            continue
        if col in NUMERIC_COLS:
            for i, value in enumerate(dfc[col]):
                i_v = str(feat_dict[col]) + ":" + str(value)
                dfc[col][i] = i_v
                print(dfc[col][i])

        else:
            for i, value in enumerate(dfc[col]):

                i_v = str(feat_dict[col][value]) + ":" + str(value)
                dfc[col][i] = i_v
                print(dfc[col][i])

    return dfc

def parse_libffm(feat_dict, df):
    """"""
    feat_index = {}
    i = 0
    for key, value in feat_dict.items():
        feat_index[key] = i
        i += 1

    dfc = df.copy()
    for col in dfc.columns:
        if col in IGNORE_COLS:
            dfc.drop(col, axis=1, inplace=True)
            continue
        if col == 'target':
            continue
        if col in NUMERIC_COLS:
            for i, value in enumerate(dfc[col]):
                i_v = str(feat_index[col]) + ":" + str(feat_dict[col]) + ":" + str(value)
                dfc[col][i] = i_v
                print(dfc[col][i])

        else:
            for i, value in enumerate(dfc[col]):

                i_v = str(feat_index[col]) + ":" + str(feat_dict[col][value]) + ":" + str(value)
                dfc[col][i] = i_v
                print(dfc[col][i])

    return dfc

def generation_csv(data_path, new_path):
    """生成csv格式文件"""
    df_data = pd.read_csv(data_path)
    df_data.drop(['id'], axis=1, inplace=True)
    df_data.to_csv(new_path, index=False, header=None)

def generation_libsvm(data_path, new_path):
    """生成libsvm格式文件"""
    df_data = pd.read_csv(data_path)
    df_data.drop(['id'], axis=1, inplace=True)
    feat_dict = gen_feat_dict(df_data)
    df = parse_libsvm(feat_dict, df_data)
    df.to_csv(new_path, index=False, header=None)

def generation_libffm(data_path, new_path):
    """生成libsvm格式文件"""
    df_data = pd.read_csv(data_path)
    df_data.drop(['id'],axis=1,inplace=True)
    df_data.to_csv(new_path, index=False, header=None)


if __name__ == '__main__':
    data_path = '../../data/ctr/test.csv'
    new_path = 'test.csv'
    generation_libsvm(data_path, new_path)