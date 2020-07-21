import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
"""
基于sklearn进行数据预处理 —— 归一化/标准化/正则化
"""

def standard(train, test):
    """标准化"""
    ss = StandardScaler()
    ss.fit(train)
    prod = ss.transform(test)
    print(prod)

def normalized(train, test):
    """
    归一化
    对于方差非常小的属性可以增强其稳定性。
    """
    mms = MinMaxScaler()
    mms.fit(train)
    prod = mms.transform(test)
    print(prod)

def normal(train, test):
    """正则化"""
    model = Normalizer()
    model.fit(train)
    prod = model.transform(test)
    print(prod)

def pca():
    """pca降维"""
    # 信息保留90%
    X = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    test = [[5, 4, 4, 1]]
    pca = PCA(n_components=0.9)
    pca.fit(X)

    data = pca.transform(test)
    print(data)


if __name__ == '__main__':
    train = np.array(
        [[1], [2], [3]]
    )
    test = np.array(
        [[2]]
    )
    # standard(train, test)
    # normalized(train, test)
    # normal(train, test)
    pca()