# -*- coding: utf-8 -*-
"""
    trie_tree
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-06-23
    :python version: 3.5
"""


def init_node(node_value):
    new_node = {
        'child_nodes_mapping': dict(),
        'value': node_value
    }

    return new_node


root_node = init_node(None)


def insert_str(string):
    """
    把字符串插入树中
    :param string:
    :return:
    """

    cur_node = root_node
    for char in string:
        child_node_mapping = cur_node['child_node_mapping']
        if char not in child_node_mapping:
            new_child_node = init_node(char)
            child_node_mapping[char] = new_child_node

        cur_node = child_node_mapping[char]

    return True


def search(search_str):
    if not isinstance(search_str, str) or not search_str:
        return False

    cur_node = root_node
    for char in search_str:
        child_node_mapping = cur_node['child_node_mapping']
        if char not in child_node_mapping:
            return False

        cur_node = child_node_mapping[char]

    return True
