# -*- coding: utf-8 -*-
"""
    read_jssip_log
    ~~~~~~~

    Description.

    :author: zhaoyouqing
    :copyright: (c) 2018, Tungee
    :date created: 2020-09-02
    :python version: 3.5
"""
from copy import deepcopy
from datetime import datetime
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib import ticker

if __name__ == '__main__':
    # filename = '/home/zhaoyouqing/Desktop/oepnsips-test-report/30-1000.log'
    filename = '/home/zhaoyouqing/Desktop/oepnsips-test-report/50-2000.log'
    first_time = None
    end_time = ''

    max_spent_time = -1
    total_spent_time = 0
    avg_spent_time = 0
    user_counter_map = dict()
    total_register_times = 0
    default_user = {
        '注册成功': 0,
        '反注册成功': 0,
        '注册耗时': 0
    }
    spent_time_list = list()

    with open(filename, 'r') as f:
        t_begin = None

        for _ in f:
            _ = _.strip()
            if not _:
                continue

            data = _.split(' ')

            if len(data) <= 5:
                continue

            user_id = data[4]
            key = data[5]

            if key not in default_user:
                continue

            if user_id not in user_counter_map:
                user_counter_map.setdefault(user_id, deepcopy(default_user))

            user_counter_map[user_id][key] += 1

            if key == '注册耗时':

                spent_time = int(data[6])
                spent_time_list.append(spent_time)
                total_register_times += 1
                total_spent_time += spent_time
                if spent_time > max_spent_time:
                    max_spent_time = spent_time

    pprint(user_counter_map)
    print('max_spent_time: ', max_spent_time, 'ms, avg_spent_time: ',  total_spent_time / total_register_times, 'ms')

    # 绘图
    plt.plot(spent_time_list, color='blue', lw=0.1)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%s ms'))
    # plt.ylim(0, 200)
    plt.ylabel('register time')
    plt.show()
