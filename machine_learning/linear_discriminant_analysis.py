# -*- coding: utf-8 -*-
"""
    linear_discriminant_analysis
    ~~~~~~~

    Description. 线性判别分析

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-09-10
    :python version: 3.5
"""
import pandas
from numpy.linalg import inv
import numpy as np


def lda(file_path):
    """
    线性判别法
    :param file_path:
    :return:
    """
    # 求出均值向量 u0、u1，求出类内散度矩阵的逆矩阵
    data = pandas.read_csv(file_path)

    X0 = list()
    X1 = list()

    headers = data.keys()
    for i, row in data.iterrows():
        if row[9] == '是':
            X1.append([row[7], row[8]])
        else:
            X0.append([row[7], row[8]])

    X0 = np.array(X0, dtype=float)
    X1 = np.array(X1, dtype=float)

    u0 = X0.sum(axis=0) / X0.shape[0]
    u1 = X1.sum(axis=0) / X1.shape[0]

    Sw = np.cov(X0.transpose()) + np.cov(X1.transpose())

    w = np.dot(inv(Sw), ((u0 - u1).reshape(-1, 1)))
    # w = inv(Sw) * ((u0 - u1).reshape(-1, 1))
    print(w)
    # print(Sw)
    #
    # # 求Sw 的逆
    # print(inv(Sw))
    # print(Sw_r)
