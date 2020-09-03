import pandas as pd

class FeatureDictionary(object):
    def __init__(self, df_train, df_test, numeric_cols=[], ignore_cols=[]):
        self.df_train = df_train
        self.df_test = df_test
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.feat_dict = {}
        self.gen_feat_dict()

    def gen_feat_dict(self):
        df = pd.concat([self.df_train, self.df_test])
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue

            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                # print(us)
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)

        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def get_xi_xv(self, dfi, dfv):
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

        return xi, xv

    def parse(self, df, has_label=False):
        """
        return:  xi :列的序号， xv :列的对应的值
        """
        dfi = df.copy()

        if has_label:
            y = dfi['target'].values.tolist()
            dfi.drop(['id', 'target'], axis=1, inplace=True)
            dfv = dfi.copy()
            xi, xv = self.get_xi_xv(dfi, dfv)
            return xi, xv, y
        else:
            ids = dfi['id'].values.tolist()
            dfi.drop(['id'], axis=1, inplace=True)
            dfv = dfi.copy()
            xi, xv = self.get_xi_xv(dfi, dfv)
            return xi, xv, ids



