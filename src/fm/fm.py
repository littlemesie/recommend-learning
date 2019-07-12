# -*- coding: utf-8 -*-

from __future__ import division
from math import exp
from numpy import *
from random import normalvariate  # 正态分布
from sklearn import preprocessing
import numpy as np

# 参考 https://www.cnblogs.com/wkang/p/9588360.html

'''
    data : 数据的路径
    feature_potenital : 潜在分解维度数
    alpha ： 学习速率
    iter ： 迭代次数
    _w,_w_0,_v ： 拆分子矩阵的weight
    with_col : 是否带有columns_name
    first_col : 首列有价值的feature的index
'''


class fm(object):
    def __init__(self):
        self.data = None
        self.feature_potential = None
        self.alpha = None
        self.iter = None
        self._w = None
        self._w_0 = None
        self.v = None
        self.with_col = None
        self.first_col = None

    def min_max(self, data):
        self.data = data
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(self.data)

    def loadDataSet(self, data, with_col=True, first_col=2):

        self.first_col = first_col
        dataMat = []
        labelMat = []
        fr = open(data)
        self.with_col = with_col
        if self.with_col:
            N = 0
            for line in fr.readlines():
                # N=1时干掉列表名
                if N > 0:
                    currLine = line.strip().split()
                    lineArr = []
                    featureNum = len(currLine)
                    for i in range(self.first_col, featureNum):
                        lineArr.append(float(currLine[i]))
                    dataMat.append(lineArr)
                    labelMat.append(float(currLine[1]) * 2 - 1)
                N = N + 1
        else:
            for line in fr.readlines():
                currLine = line.strip().split()
                lineArr = []
                featureNum = len(currLine)
                for i in range(2, featureNum):
                    lineArr.append(float(currLine[i]))
                dataMat.append(lineArr)
                labelMat.append(float(currLine[1]) * 2 - 1)
        return mat(self.min_max(dataMat)), labelMat

    def sigmoid(self, inx):
        return 1.0 / (1 + exp(-inx))

    # 得到对应的特征weight的矩阵
    def fit(self, data, feature_potential=8, alpha=0.01, iter=100):
        # alpha是学习速率
        self.alpha = alpha
        self.feature_potential = feature_potential
        self.iter = iter
        # dataMatrix用的是mat, classLabels是列表
        dataMatrix, classLabels = self.loadDataSet(data)
        print('dataMatrix:',dataMatrix.shape)
        print('classLabels:',classLabels)
        k = self.feature_potential
        m, n = shape(dataMatrix)
        # 初始化参数
        w = zeros((n, 1))  # 其中n是特征的个数
        w_0 = 0.
        v = normalvariate(0, 0.2) * ones((n, k))
        for it in range(self.iter): # 迭代次数
            # 对每一个样本，优化
            for x in range(m):
                # 这边注意一个数学知识：对应点积的地方通常会有sum，对应位置积的地方通常都没有，详细参见矩阵运算规则，本处计算逻辑在：http://blog.csdn.net/google19890102/article/details/45532745
                # xi·vi,xi与vi的矩阵点积
                inter_1 = dataMatrix[x] * v
                # xi与xi的对应位置乘积   与   xi^2与vi^2对应位置的乘积    的点积
                inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # multiply对应元素相乘
                # 完成交叉项,xi*vi*xi*vi - xi^2*vi^2
                interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
                # 计算预测的输出
                p = w_0 + dataMatrix[x] * w + interaction
                print('classLabels[x]:',classLabels[x])
                print('预测的输出p:', p)
                # 计算sigmoid(y*pred_y)-1
                loss = self.sigmoid(classLabels[x] * p[0, 0]) - 1
                if loss >= -1:
                    loss_res = '正方向 '
                else:
                    loss_res = '反方向'
                # 更新参数
                w_0 = w_0 - self.alpha * loss * classLabels[x]
                for i in range(n):
                    if dataMatrix[x, i] != 0:
                        w[i, 0] = w[i, 0] - self.alpha * loss * classLabels[x] * dataMatrix[x, i]
                        for j in range(k):
                            v[i, j] = v[i, j] - self.alpha * loss * classLabels[x] * (
                                    dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
            print('the no %s times, the loss arrach %s' % (it, loss_res))
        self._w_0, self._w, self._v = w_0, w, v

    def predict(self, X):
        if (self._w_0 == None) or (self._w == None).any() or (self._v == None).any():
            raise NotFittedError("Estimator not fitted, call `fit` first")
        # 类型检查
        if isinstance(X, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")
        w_0 = self._w_0
        w = self._w
        v = self._v
        m, n = shape(X)
        result = []
        for x in range(m):
            inter_1 = mat(X[x]) * v
            inter_2 = mat(multiply(X[x], X[x])) * multiply(v, v)  # multiply对应元素相乘
            # 完成交叉项
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
            p = w_0 + X[x] * w + interaction  # 计算预测的输出
            pre = self.sigmoid(p[0, 0])
            result.append(pre)
        return result

    def accuracy(self, data):
        dataMatrix, classLabels = self.loadDataSet(data)
        w_0 = self._w_0
        w = self._w
        v = self._v
        m, n = shape(dataMatrix)
        allItem = 0
        error = 0
        result = []
        for x in range(m):
            allItem += 1
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # multiply对应元素相乘
            # 完成交叉项
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出
            pre = self.sigmoid(p[0, 0])
            result.append(pre)
            if pre < 0.5 and classLabels[x] == 1.0:
                error += 1
            elif pre >= 0.5 and classLabels[x] == -1.0:
                error += 1
            else:
                continue
        # print(result)
        value = 1 - float(error) / allItem
        return value


class NotFittedError(Exception):
    """
    Exception class to raise if estimator is used before fitting
    """
    pass


if __name__ == '__main__':
    fm()