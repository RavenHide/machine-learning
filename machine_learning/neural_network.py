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
        weight_matrix = np.random.randn(layer_dims[i + 1], layer_dims[i])
        weight_matrix = weight_matrix * np.sqrt(1 / layer_dims[i])
        threshold_matrix = np.zeros((1, layer_dims[i + 1]))

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
    max_loop = 1000
    for _ in range(max_loop):
        for i, x in data.iterrows():
            y = x.pop('好瓜').reshape(-1, 1)

            input_list = [x]
            # 网络正向传播，输出 y'
            y_hat = x
            for i, params_item in enumerate(network_params_list):
                weight_matrix = params_item['w']
                input = np.dot(y_hat, weight_matrix.transpose())
                activation_x = input - params_item['t']
                y_hat = 1 / (1 + np.exp(-activation_x))

                if i < len(network_params_list) - 1:
                    input_list.append(y_hat)

            print(input_list)

            g = y_hat * (1 - y_hat) * (y - y_hat)
            # for j in range(len(network_params_list) -1, -1, -1):
            #     delta_w = learning_rate * g


            # print(g)
            break
        break

            #
            # 向后传播，调整network_params



