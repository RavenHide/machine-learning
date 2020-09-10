# -*- coding: utf-8 -*-
"""
    logistic_regression
    ~~~~~~~

    Description. 逻辑线性回归

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-09-10
    :python version: 3.5
"""
import pandas
import numpy as np

def get_X_y_from_file(file_path):
    data = pandas.read_csv(file_path)

    X = data.values[:, 7:9].astype(float)

    y = list()
    result_key = data.keys()[len(data.keys()) - 1]
    for _ in data.get(result_key):
        if _ == '是':
            y.append(1)
        else:
            y.append(0)

    y = np.array(y)
    y = y.reshape(-1, 1)

    return X, y


def logistic_regression(filepath):
    """
    逻辑线性回归预测
    :param :
    :return:
    """
    X, y = get_X_y_from_file(filepath)

    # 初始化一个beta
    beta = np.random.randn(X.shape[1] + 1, 1) * 0.5 + 1

    beta = beta.reshape(-1, 1)
    # 定义步长
    learning_rate = 0.15
    max_iter_count = 100000

    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    for i in range(max_iter_count):
        # 计算梯度

        Z = np.dot(X_hat, beta)
        p1 = 1 / (1 + np.exp(-Z))
        grad = (X_hat * (p1 - y)).sum(0).reshape(-1, 1)

        beta = beta - learning_rate * grad

        if i % 100 == 0:
            print(_J_cost_(X_hat, y, beta))


def _J_cost_(X_hat, y, beta):

    Z = np.dot(X_hat, beta)

    Lbeta = -y * Z + np.log(1 + np.exp(Z))
    return Lbeta.sum()


if __name__ == '__main__':
    path = './../data/watermelon3_0_Ch.csv'
    logistic_regression(path)