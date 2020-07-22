import pandas as pd

class FeatureDictionary(object):
    def __init__(self, train_file=None, test_file=None, df_train=None, df_test=None, numeric_cols=None, ignore_cols=None):
        assert not ((train_file is None) and (df_train is None)), "trainfile or dfTrain at least one is set"
        assert not ((train_file is not None) and (df_train is not None)), "only one can be set"
        assert not ((test_file is None) and (df_test is None)), "testfile or dfTest at least one is set"
        assert not ((test_file is not None) and (df_test is not None)), "only one can be set"

        self.train_file = train_file
        self.test_file = test_file
        self.df_train = df_train
        self.df_test = df_test
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()




    def gen_feat_dict(self):
        """
        返回每个特征值对应的索引
        :return: {'missing_feat': 0, 'ps_car_01_cat': {10: 1, 11: 2, 7: 3, 6: 4, 9: 5, 5: 6, 4: 7, 8: 8, 3: 9, 0: 10, 2: 11, 1: 12, -1: 13}, 'ps_car_02_cat': {1: 14, 0: 15}}
        """
        if self.df_train is None:
            df_train = pd.read_csv(self.train_file)

        else:
            df_train = self.df_train

        if self.df_test is None:
            df_test = pd.read_csv(self.test_file)

        else:
            df_test = self.df_test

        df = pd.concat([df_train, df_test])

        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1

            else:
                us = df[col].unique()

                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)

        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        """
        :param infile:
        :param df:
        :param has_label:
        :return:  xi :列的序号， xv :列的对应的值， y : label
        """
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"


        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        if has_label:
            y = dfi['target'].values.tolist()
            dfi.drop(['id', 'target'], axis=1, inplace=True)
        else:
            ids = dfi['id'].values.tolist()
            dfi.drop(['id'], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()
        if has_label:
            return xi, xv, y
        else:
            return xi, xv, ids


