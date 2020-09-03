# -*- coding: utf-8 -*-
"""
    playground
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-07-28
    :python version: 3.5
"""
from pprint import pprint

import numpy as np
import pandas
import sys
from numpy.linalg import inv
from pandas import DataFrame


def _J_cost_(X_hat, y, beta):

    Z = np.dot(X_hat, beta)

    Lbeta = -y * Z + np.log(1 + np.exp(Z))
    return Lbeta.sum()


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


def _build_decision_tree_2_(
        node, data, header_dict, classify_index, validation_data,
        algo_model=1
):
    """
    构建决策树2
    :param node:
    :param data:
    :param header_dict:
    :param classify_index:
    :param validation_data: 验证集
    :param algo_model:
            1 - 信息增益
            2 - 信息率
            3 - 基尼指数
    :return:
    """
    # 判断数据集是否是同一类型
    classification_count_dict, total = _get_attribute_value_count_dict_(
        data, classify_index
    )
    greater_than_0_count = 0
    hit_classify = None
    for classify, count in classification_count_dict.items():
        if count <= 0:
            continue

        greater_than_0_count += 1
        hit_classify = classify

    # 只有一个分类性
    if greater_than_0_count == 1:
        node['classify'] = hit_classify
        node['type'] = 'leaf'
        return None

    # 没有属性值时，选样本最多的类别作为结果
    if not header_dict:
        node['classify'] = hit_classify
        node['type'] = 'leaf'
        return None

    # 判断样本的当前的属性值是否都是一致
    one_attribute_value_count = 0
    for attribute_index in header_dict.keys():
        attribute_count_dict, _ = _get_attribute_value_count_dict_(
            data, attribute_index
        )
        if len(attribute_count_dict) != 1:
            continue

        one_attribute_value_count += 1

    # 所有样本的所有属性值相同
    if one_attribute_value_count == len(header_dict):
        node['classify'] = hit_classify
        node['type'] = 'leaf'
        return None

    hit_attribute_index, hit_attribute_value_dict = _get_a_best_attribute_(
        data, classify_index, header_dict, algo_model=algo_model
    )

    # todo 先进行预减枝
    node['attribute'] = header_dict.pop(hit_attribute_index)

    is_pre_pruning, hit_classify = _is_pre_pruning_(
        validation_data, classify_index, hit_attribute_value_dict,
        hit_attribute_index
    )

    # 不需要剪枝，直接标记为叶子节点
    if not is_pre_pruning:
        node['classify'] = hit_classify
        node['type'] = 'leaf'
        return None

    children_nodes = list()
    for attribute_value, sub_data in hit_attribute_value_dict.items():
        child_node = {
            'parent_attribute_value': attribute_value,
            'parent_attribute': node['attribute']
        }
        children_nodes.append(child_node)
        _build_decision_tree_2_(
            child_node, sub_data, header_dict, classify_index, validation_data
        )

    node['children_nodes'] = children_nodes

    return None


def _get_classify_result_(data, classify_index):
    """
    获取数据集分类结果
    :param data:
    :param classify_index:
    :return:
    """
    classification_count_dict, total = _get_attribute_value_count_dict_(
        data, classify_index
    )
    max_ratio = -1
    hit_classify = None
    hit_classify_count = 0

    for classify, classify_count in classification_count_dict.items():
        cur_ratio = classify_count / total
        if cur_ratio <= max_ratio:
            continue

        hit_classify = classify
        hit_classify_count = classify_count

    return hit_classify, hit_classify_count, total


def _is_pre_pruning_(
        validation_data, classify_index, attribute_value_dict,
        attribute_index
):
    """
    预剪枝
    :param validation_data: 验证集
    :param classify_index:
    :param attribute_value_dict:
    :param attribute_index:
    :return:
    """

    # 先确定预测类型
    hit_classify, _, _ = _get_classify_result_(
        attribute_value_dict, classify_index
    )

    total = len(validation_data)
    hit_classify_count = 0
    for i, _ in _read_data_(validation_data):
        if _[classify_index] != hit_classify:
            continue

        hit_classify_count += 1

    precision_before = hit_classify_count / total

    hit_count_after_pre_pruning = 0
    # 计算剪枝后的精度
    for attribute_value, sub_data in attribute_value_dict.items():
        sub_classify, _, _ = _get_classify_result_(
            sub_data, classify_index
        )

        for i, _ in _read_data_(validation_data):
            if _[attribute_index] != attribute_value:
                continue

            if _[classify_index] != sub_classify:
                continue

            hit_count_after_pre_pruning += 1

    precision_after = hit_count_after_pre_pruning / len(validation_data)

    if precision_before > precision_after:
        return False, hit_classify

    return True, None


def decision_tree(file_path):
    """
    决策树
    :param file_path:
    :return:
    """
    data = pandas.read_csv(file_path)

    validation_data = list()
    # 选择训练集和验证集
    # validation_data = data.sample(int(np.ceil(len(data) * 0.4)))
    #
    # exclude_number_set = {_['编号'] for i, _ in _read_data_(validation_data)}
    exclude_number_set = {4, 5, 8, 9, 11, 12, 13}
    train_data = list()
    for i, _ in _read_data_(data):
        if _['编号'] in exclude_number_set:
            validation_data.append(_)
            continue

        train_data.append(_)

    classify_index = 9
    # information_entropy, total, _ = _cal_information_entropy_(data, 9)

    headers = data.keys()
    header_dict = {_: headers[_] for _ in range(1, len(headers) - 3, 1)}
    root_node = {
        'name': 'root'
    }
    # _build_decision_tree_(root_node, data, header_dict, classify_index)
    _build_decision_tree_2_(
        root_node, train_data, header_dict, classify_index, validation_data

    )

    pprint(root_node)


def _read_data_(data):
    if isinstance(data, DataFrame):
        for i, _ in data.iterrows():
            yield i, _

    elif isinstance(data, list):
        for i, _ in enumerate(data):
            yield i, _
    else:
        return None


def _cal_information_entropy_(data, classify_index):
    """
    计算信息熵
    :param data:
    :param classify_index:类别对应的列编号
    :return:
    """

    classification_count_dict, total = _get_attribute_value_count_dict_(
        data, classify_index
    )
    # total = 0
    #
    # for i, _ in _read_data_(data):
    #     classify = _[classify_index]
    #     classification_count_dict.setdefault(classify, 0)
    #     classification_count_dict[classify] += 1
    #     total += 1

    information_entropy = 0
    for classification_count in classification_count_dict.values():
        p = classification_count / total
        information_entropy += p * np.log2(p)

    information_entropy = round(information_entropy, 3) * -1
    return information_entropy, total, classification_count_dict


def _get_attribute_value_count_dict_(data, col_index):
    """
    获取属性值数量的字典
    :param data:
    :param col_index:
    :return:
    """
    attribute_value_count_dict = dict()
    total = 0
    for i, _ in _read_data_(data):
        hit_key = _[col_index]
        attribute_value_count_dict.setdefault(hit_key, 0)
        attribute_value_count_dict[hit_key] += 1
        total += 1

    return attribute_value_count_dict, total


def _get_attribute_value_dict_(data, col_index):
    """
    获取属性值的字典
    :param data:
    :param col_index:
    :return:
    """
    attribute_value_dict = dict()
    for i, _ in _read_data_(data):
        hit_key = _[col_index]
        attribute_value_dict.setdefault(hit_key, list())
        attribute_value_dict[hit_key].append(_)

    return attribute_value_dict


def _cal_grain_(data, col_index, entropy, total, classify_col):
    """
    计算信息增益
    :param data:
    :param col_index:
    :param entropy
    :param total
    :return:
    """

    attribute_value_dict = _get_attribute_value_dict_(data, col_index)
    total_sub_entropy = 0

    for sub_data in attribute_value_dict.values():
        sub_entropy, sub_total, _ = _cal_information_entropy_(
            sub_data, classify_col
        )
        total_sub_entropy += sub_total / total * sub_entropy

    grain = entropy - total_sub_entropy
    return grain, attribute_value_dict


def _cal_gini_value_(data, classify_index):
    """
    计算基尼值
    :param data:
    :param classify_index:
    :return:
    """
    class_count_dict, total = _get_attribute_value_count_dict_(
        data, classify_index
    )

    gini_value = 1 - sum(lambda x: (x / total)**2, class_count_dict.values())

    return gini_value


def _cal_gini_index_(data, attribute_col_index, classify_index):
    """
    计算基尼指数
    :param data:
    :param attribute_col_index:
    :param classify_index:
    :return:
    """
    attribute_value_dict = _get_attribute_value_dict_(data, attribute_col_index)
    total = len(data)

    gini_index = 0
    for attribute_value, sub_data in attribute_value_dict.items():
        sub_gini_value = _cal_gini_value_(sub_data, classify_index)
        gini_index += len(sub_data) / total * sub_gini_value

    return gini_index, attribute_value_dict


def _get_a_best_attribute_(data, classify_index, header_dict, algo_model=1):
    """
    获取一个最佳的分类性属性
    :param data:
    :param classify_index:
    :param header_dict:
    :param algo_model:
    :return:
    """
    total = len(data)
    hit_attribute_index = -1
    hit_attribute_value_dict = dict()

    # 信息增益
    if algo_model == 1:
        # 计算信息熵
        entropy, _, _ = _cal_information_entropy_(data, classify_index)
        max_gain = -1
        # 选取一个最优的划分属性

        for attribute_index in header_dict.keys():
            gain, attribute_value_dict = _cal_grain_(
                data, attribute_index, entropy, total, classify_index
            )
            if gain < max_gain:
                continue

            max_gain = gain
            hit_attribute_index = attribute_index
            hit_attribute_value_dict = attribute_value_dict

    # 信息增益率
    elif algo_model == 2:
        raise NotImplementedError
    # 基尼指数
    elif algo_model == 3:
        min_gini = sys.maxsize
        for attribute_index in header_dict.keys():
            gini_index, attribute_value_dict = _cal_gini_index_(
                data, attribute_index, classify_index
            )

            if gini_index > min_gini:
                continue

            hit_attribute_index = attribute_index
            hit_attribute_value_dict = attribute_value_dict
            min_gini = gini_index
    else:
        raise NotImplementedError

    return hit_attribute_index, hit_attribute_value_dict


if __name__ == '__main__':
    path = './../data/watermelon3_0_Ch.csv'
    # logistic_regression(path)
    decision_tree(path)