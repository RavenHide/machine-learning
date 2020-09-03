# -*- coding: utf-8 -*-
"""
    kmp_search
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-06-23
    :python version: 3.5
"""


def kmp_match(match_str, patten_str):
    """
    字符串匹配
    :param match_str:
    :param patten_str:
    :return:
    """

    i, j = 0, 0
    next_array = create_next_array(patten_str)
    while i < len(match_str) and j < len(patten_str):
        if j == -1 or match_str[i] == patten_str[j]:
            i += 1
            j += 1
        else:
            j = next_array[j]

    # 没匹配到了
    if j != len(patten_str):
        return False

    return True


def create_next_array(patten_str):
    """
    生成模式字符串的next数组
    :param patten_str:
    :return:
    """

    next_array = [0] * len(patten_str)
    next_array[0] = -1
    i, j = 0, -1
    while i < len(patten_str):
        if j == -1 or patten_str[i] == patten_str[j]:

            i += 1
            j += 1

            if i >= len(patten_str):
                break

            next_array[i] = j

        else:
            j = next_array[j]

    return next_array

if __name__ == '__main__':
    print(kmp_match('abababca', 'bcb'))

