# -*- coding: utf-8 -*-
"""
    neural_network
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-09-10
    :python version: 3.5
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def params_init_with_xavier(layer_dims, seed=16):
    """
    xavier参数初始化
    :param layer_dims:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    network_params_list = list()

    for i in range(len(layer_dims) - 1):
        weight_matrix = np.random.randn(layer_dims[i], layer_dims[i + 1])
        weight_matrix = weight_matrix * np.sqrt(1 / layer_dims[i])
        threshold_matrix = np.zeros((layer_dims[i + 1], 1))

        network_params_list.append(
            {
                'w': weight_matrix,
                't': threshold_matrix
            }
        )

    return network_params_list


if __name__ == '__main__':

    path = './../data/watermelon3_0_Ch.csv'

    network_params_list = list()
    layer_dims = [3, 1]

    data = pd.read_csv(path, index_col=0)
    columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    data = pd.get_dummies(data, columns=columns)
    data['好瓜'].replace(['是', '否'], [1, 0], inplace=True)

    layer_dims.insert(0, data.shape[1] - 1)

    network_params_list.extend(params_init_with_xavier(layer_dims))

    learning_rate = 0.01
    max_loop = 10000
    loss_list = list()
    for _ in range(max_loop):

        for i, x in data.iterrows():
            # 1 x l
            y = x.pop('好瓜').reshape(-1, 1)

            input_list = list()
            # 网络正向传播，输出 y'
            x_matrix = np.mat(x)

            # 1 x d
            y_hat = x_matrix
            for i, params_item in enumerate(network_params_list):
                weight_matrix = params_item['w']

                input = np.dot(y_hat, weight_matrix)
                activation_x = input - params_item['t'].T
                y_hat = 1 / (1 + np.exp(-activation_x))

                if i < len(network_params_list) - 1:
                    input_list.append(y_hat)

            # 计算损失函数
            # 交叉熵损失函数计算loss
            y_hat_vector = y_hat.T

            loss = -1 * (
                np.dot(y, np.log(y_hat_vector)) +
                np.dot((1 - y), np.log(1 - y_hat_vector))
            ) / y_hat_vector.shape[0]

            loss_list.extend(np.array(loss.flat))

            # 计算输出层神经元的梯度项
            # d, q, l, 各层神经元的总数

            # l x 1 矩阵
            g_vector = np.dot(
                np.diag(np.diag(np.dot(y_hat.T, (1 - y_hat))).flat),
                (y - y_hat).T
            )

            # 1 x q 矩阵
            b_vector = input_list[0]

            # q x 1 矩阵
            e_vector = np.dot(
                np.diag(np.dot(b_vector, np.diag((1 - b_vector).flat)).flat),
                np.dot(network_params_list[1]['w'], g_vector)
            )

            # 开始更新神经网络参数
            delta_W_1_matrix = learning_rate * np.dot(b_vector.T, g_vector.T)
            network_params_list[1]['w'] += delta_W_1_matrix
            delta_T_1_matrix = -1 * learning_rate * g_vector

            network_params_list[1]['t'] += delta_T_1_matrix

            delta_W_0_matrix = learning_rate * np.dot(
                x_matrix.T, e_vector.T
            )
            network_params_list[0]['w'] += delta_W_0_matrix

            delta_T_0_matrix = -1 * learning_rate * e_vector
            network_params_list[0]['t'] += delta_T_0_matrix

            # 计算隐层神经元的梯度项
            # e =


            # for j in range(len(network_params_list) -1, -1, -1):
            #     delta_w = learning_rate * g


            # print(g)

            #
            # 向后传播，调整network_params

    plt.plot(loss_list)
    plt.show()