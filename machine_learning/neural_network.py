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

import numpy as np
import pandas as pd
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
        a = np.sqrt(6 / (layer_dims[i] + layer_dims[i + 1]))
        # weight_matrix = np.random.randn(layer_dims[i], layer_dims[i + 1])
        weight_matrix = np.random.uniform(
            -a, a, (layer_dims[i], layer_dims[i+1])
        )
        threshold_matrix = np.zeros((layer_dims[i + 1], 1))

        network_params_list.append(
            {
                'w': weight_matrix,
                't': threshold_matrix
            }
        )

    return network_params_list


def stander_error_back_propagation(data, learning_rate, max_loop=100):
    """
    标准误差逆传播算法
    :param data:
    :param learning_rate:
    :param max_loop:
    :return:
    """
    layer_dims = [3, 1]
    layer_dims.insert(0, data.shape[1] - 1)
    network_params_list = params_init_with_xavier(layer_dims)
    loss_list = list()

    train_times = 0
    for _ in range(max_loop):
        train_times += 1

        loss_sum = 0
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

                input_vector = np.dot(y_hat, weight_matrix)

                activation_x = input_vector - params_item['t'].T

                y_hat = 1 / (1 + np.exp(-activation_x))

                if i < len(network_params_list) - 1:
                    input_list.append(y_hat)

            # 计算损失函数
            # 交叉熵损失函数计算loss
            # l x 1
            y_hat_vector = y_hat.T

            loss = -1 * (
                np.dot(y, np.log2(y_hat_vector)) +
                np.dot((1 - y), np.log2(1 - y_hat_vector))
            ) / y_hat_vector.shape[0]
            loss_sum += np.squeeze(loss.getA())

            # 计算输出层神经元的梯度项
            # d, q, l, 各层神经元的总数

            # l x 1 矩阵
            g_vector = np.dot(
                np.diag(np.dot(np.diag(y_hat_vector.flat), 1 - y_hat).flat),
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

        avg_loss = loss_sum / len(data)
        loss_list.append(avg_loss)

        if (_ % 50) == 0:
            print('第%s次训练: ' % train_times)
            print('损失: %s' % loss_list[-1])

    print('最终损失: %s' % loss_list[-1])
    return network_params_list, loss_list


def accumulated_error_back_propagation(data, learning_rate, max_loop=100):
    """
    累计误差逆传播算法
    :param data:
    :param learning_rate:
    :param max_loop:
    :return:
    """

    layer_dims = [3, 1]
    layer_dims.insert(0, data.shape[1] - 1)
    network_params_list = params_init_with_xavier(layer_dims)
    loss_list = list()

    train_times = 0
    for _ in range(max_loop):
        train_times += 1

        loss_sum = 0
        # 定义随机索引，命中索引时，才会更新神经网络的参数
        random_index = np.random.randint(1, len(data) + 1)

        hit_params = dict()
        for i, x in data.iterrows():

            # 1 x l
            y = x.pop('好瓜').reshape(-1, 1)

            input_list = list()
            # 网络正向传播，输出 y'
            x_matrix = np.mat(x)

            # 1 x d
            y_hat = x_matrix

            for j, params_item in enumerate(network_params_list):
                weight_matrix = params_item['w']

                input_vector = np.dot(y_hat, weight_matrix)

                activation_x = input_vector - params_item['t'].T

                y_hat = 1 / (1 + np.exp(-activation_x))

                if j < len(network_params_list) - 1:
                    input_list.append(y_hat)

            # 计算损失函数
            # 交叉熵损失函数计算loss
            # l x 1
            y_hat_vector = y_hat.T

            loss = -1 * (
                np.dot(y, np.log2(y_hat_vector)) +
                np.dot((1 - y), np.log2(1 - y_hat_vector))
            ) / y_hat_vector.shape[0]
            loss_sum += np.squeeze(loss.getA())

            if random_index != i:
                continue

            hit_params.update(
                {
                    'y': y,
                    'y_hat_vector': y_hat_vector,
                    'y_hat': y_hat,
                    'b_vector': input_list[0],
                    'x_matrix': x_matrix
                }
            )

        # 计算输出层神经元的梯度项
        # d, q, l, 各层神经元的总数

        # l x 1 矩阵
        g_vector = np.dot(
            np.diag(
                np.dot(
                    np.diag(hit_params['y_hat_vector'].flat),
                    1 - hit_params['y_hat']
                ).flat
            ),
            (hit_params['y'] - hit_params['y_hat']).T
        )

        # 1 x q 矩阵
        b_vector = hit_params['b_vector']

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
            hit_params['x_matrix'].T, e_vector.T
        )

        network_params_list[0]['w'] += delta_W_0_matrix

        delta_T_0_matrix = -1 * learning_rate * e_vector
        network_params_list[0]['t'] += delta_T_0_matrix

        avg_loss = loss_sum / len(data)
        loss_list.append(avg_loss)

        if (_ % 50) == 0:
            print('第%s次训练: ' % train_times)
            print('损失: %s' % loss_list[-1])

    print('最终损失: %s' % loss_list[-1])
    return network_params_list, loss_list


if __name__ == '__main__':

    path = './../data/watermelon3_0_Ch.csv'

    data = pd.read_csv(path, index_col=0)
    columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    data = pd.get_dummies(data, columns=columns)
    data['好瓜'].replace(['是', '否'], [1, 0], inplace=True)

    # 标准误差逆传播算法
    _, loss_list1 = stander_error_back_propagation(
        data, 0.1, max_loop=20000
    )

    # 累计误差逆传播算法
    _, loss_list2 = accumulated_error_back_propagation(
        data, 0.1, max_loop=20000
    )

    # plt.plot(loss_list)
    ax = plt.subplot()
    ax.plot(loss_list1, color='c', label='stander bp')
    ax.plot(loss_list2, color='lightcoral', label='accumalated bp')

    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')

    ax.legend(loc='upper right')
    ax.set_xlabel('num epochs')
    ax.set_ylabel('cost')
    plt.show()