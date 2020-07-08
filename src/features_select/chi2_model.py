import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target  #iris数据集

# 选择K个最好的特征，返回选择特征后的数据
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new)
# 获取选取索引的下标
indexs = SelectKBest(chi2, k=2).fit(X, y).get_support(indices=True)
print(type(X))
print(X[:, indexs])
print(indexs)