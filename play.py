# -*- coding: utf-8 -*-
"""
    play
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2018-08-09
    :python version: 3.5
"""
from collections import OrderedDict
from pprint import pprint


def _init_from_words(words):
    """
    read from words and construct data structure
    """
    smap = {}

    for word in words:
        if isinstance(word, bytes):
            word = word.decode()

        tmp = smap
        size = len(word)

        for i in range(size):
            if word[i] in tmp:
                tmp = tmp[word[i]]
            else:
                tmp[word[i]] = {'end': 0}
                tmp = tmp[word[i]]

            # check if the last character
            if i == size - 1:
                tmp['end'] = 1

    return smap


def check_censor_word(i_string):
    """
    Return True when having sensitive word
    """
    string = i_string
    if isinstance(i_string, bytes):
        string = i_string.decode()

    for i, char in enumerate(string):
        if char in _smap:

            for word in match_censor_word(string[i:]):
                if word is None:
                    continue
                yield word, i


def match_censor_word(sub_str):
    """
    Almost the same as check_censor_word other than it just match from the
    start of given string, if not matched, will not check other chars.

    :param sub_str:
    :return: A generator of matched word or None
    """
    string = sub_str
    if isinstance(string, bytes):
        string = string.decode()

    word = ''
    tmp = _smap
    for _, char in enumerate(string):
        if char in tmp:
            word = '%s%s' % (word, char)
            tmp = tmp[char]
            # check if read the end
            if tmp['end'] == 1:
                yield word
        else:
            yield None
            break

_smap = None
if __name__ == '__main__':
    words = OrderedDict(
        {
            '空号': 1,
            '已停机': 2,
            '来电提醒': 3,
            '无法接听': 4,
            '关机': 5,
            '限制': 6,
            '无法接通': 7,
            '通话中': 8,
            '余额不足': 9
        }
    )
    _smap = _init_from_words(words=words)

    result = check_censor_word("无法接通")
    for key, index in result:
        print(key + " ----- " + str(index))
